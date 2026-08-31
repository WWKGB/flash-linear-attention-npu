#!/usr/bin/env python3
# Copyright (c) 2026 Tianjin University, Ltd.
"""Transport and final-output precision checks for chunk_fwd_o A5.

Default (--transport): kernel launch + NPU sync only.
Use --precision to compare the final BSND output against a CPU reference.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

from fla_npu.ops.ascendc._runtime import (  # noqa: E402
    ACL_SUCCESS,
    _CallContext,
    _npu_device_guard,
    call_aclnn,
    current_stream_ptr,
    ensure_npu_tensor,
    runtime,
)

BT = 64
K = 128
V = 128


def _call_chunk_fwd_o(q, k, v, h, g, scale, *, cu_seqlens, chunk_indices, chunk_size=64):
    import torch

    # A5 output is sequence-major BSND: [B, T, HV, V].
    out = torch.empty((v.shape[0], v.shape[2], v.shape[1], v.shape[3]), dtype=v.dtype, device=v.device)
    aclnn_runtime = runtime()
    device = ensure_npu_tensor(out, "out").device
    ctx = _CallContext(aclnn_runtime, device)

    def build_args(c):
        return [
            c.tensor(q, "q"),
            c.tensor(k, "k"),
            c.tensor(v, "v"),
            c.tensor(h, "h"),
            c.tensor(g, "g"),
            c.int_array(cu_seqlens),
            c.int_array(chunk_indices),
            ctypes.c_double(float(scale)),
            ctypes.c_int64(int(chunk_size)),
            c.tensor(out, "out"),
        ]

    with _npu_device_guard(device):
        try:
            get_workspace = aclnn_runtime.symbol("aclnnChunkFwdOGetWorkspaceSize")
            launch = aclnn_runtime.symbol("aclnnChunkFwdO")
            get_workspace.restype = ctypes.c_int
            launch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            launch.restype = ctypes.c_int

            args = build_args(ctx)
            workspace_size = ctypes.c_uint64(0)
            executor = ctypes.c_void_p()
            ret = get_workspace(*args, ctypes.byref(workspace_size), ctypes.byref(executor))
            if ret != ACL_SUCCESS:
                raise RuntimeError(f"aclnnChunkFwdOGetWorkspaceSize failed: {ret}")

            workspace = torch.empty((int(workspace_size.value),), dtype=torch.uint8, device=device)
            workspace_ptr = ctypes.c_void_p(int(workspace.data_ptr()))
            ret = launch(
                workspace_ptr,
                ctypes.c_uint64(workspace_size.value),
                executor,
                ctypes.c_void_p(current_stream_ptr(device)),
            )
            if ret != ACL_SUCCESS:
                raise RuntimeError(f"aclnnChunkFwdO failed: {ret}")
        finally:
            ctx.destroy()
    torch.npu.synchronize()
    return out


def _cpu_output_ref(q, k, v, h, g, chunk_len: int, *, scale: float, use_exp2: bool = False):
    """Final-output reference aligned with the A5 kernel math."""
    import torch

    exp_fn = torch.exp2 if use_exp2 else torch.exp
    bt = BT
    valid = torch.arange(bt) < chunk_len
    g32 = g.float()[:chunk_len]
    if chunk_len < bt:
        g32 = torch.cat([g32, g32.new_zeros(bt - chunk_len)])

    gate_o = exp_fn(g32)
    gate_a = exp_fn(g32.unsqueeze(1) - g32.unsqueeze(0))
    causal = torch.arange(bt).unsqueeze(1) >= torch.arange(bt).unsqueeze(0)
    valid_2d = valid.unsqueeze(1) & valid.unsqueeze(0)
    mask = causal & valid_2d

    q_bf = torch.zeros(bt, K, dtype=torch.bfloat16)
    k_bf = torch.zeros(bt, K, dtype=torch.bfloat16)
    v_bf = torch.zeros(bt, V, dtype=torch.bfloat16)
    q_bf[:chunk_len] = q[:chunk_len].to(torch.bfloat16)
    k_bf[:chunk_len] = k[:chunk_len].to(torch.bfloat16)
    v_bf[:chunk_len] = v[:chunk_len].to(torch.bfloat16)
    h_bf = h.to(torch.bfloat16)
    a_raw = (q_bf @ k_bf.T).float()
    o_s_raw = (q_bf @ h_bf).float()
    a_prime = a_raw * gate_a * mask.float()
    o_s_prime = o_s_raw * gate_o.unsqueeze(1)
    # Stage4 keeps the Cube accumulator/Fixpipe result in FP32.  Build the
    # reference from the exact bf16 input values without rounding the GEMM
    # output back to bf16 first.
    o_l = a_prime.to(torch.bfloat16).float() @ v_bf.float()
    return ((o_s_prime + o_l) * scale).to(torch.bfloat16).float().numpy()


def _compare(name: str, ref: np.ndarray, got: np.ndarray, *, atol: float, rtol: float) -> bool:
    if ref.shape != got.shape:
        print(f"[FAIL] {name}: shape ref={ref.shape} got={got.shape}")
        return False
    if ref.dtype != got.dtype:
        ref = ref.astype(got.dtype, copy=False)
    bad = ~np.isclose(ref, got, rtol=rtol, atol=atol, equal_nan=True)
    nbad = int(bad.sum())
    if nbad == 0:
        print(f"[PASS] {name}")
        return True
    max_diff = np.max(np.abs(ref.astype(np.float64) - got.astype(np.float64)))
    print(f"[FAIL] {name}: {nbad}/{ref.size} mismatches, max_abs_diff={max_diff:.6g}")
    if bad.ndim == 2:
        bad_per_row = bad.sum(axis=1)
        bad_rows = np.flatnonzero(bad_per_row)
        if bad_rows.size:
            row_counts = [(int(row), int(bad_per_row[row])) for row in bad_rows]
            print(
                f"       bad_rows={int(bad_rows[0])}..{int(bad_rows[-1])}, "
                f"row_mismatches={row_counts[:8]}"
            )
    for index_row in np.argwhere(bad)[:5]:
        index = tuple(int(i) for i in index_row)
        print(f"       at {index} ref={ref[index]} got={got[index]}")
    return False


def run_case(*, seqlen=64, heads=None, hk=None, hv=None, use_exp2: bool = False, seed: int = 0,
             transport_only: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if heads is not None:
        if hk is not None or hv is not None:
            raise ValueError("pass either heads= or hk=/hv=, not both")
        hk = hv = heads
    if hk is None or hv is None:
        raise ValueError("hk and hv are required (or pass heads= for HK=HV)")
    if hv % hk != 0:
        raise ValueError(f"HV must be divisible by HK, got HK={hk} HV={hv}")
    g_ratio = hv // hk
    if g_ratio not in (1, 2, 3, 4):
        raise ValueError(f"G=HV/HK must be in [1,4], got {g_ratio}")

    B, HK, HV = 1, hk, hv
    scale = K ** -0.5
    chunk_size = 64
    num_chunks = (seqlen + chunk_size - 1) // chunk_size

    amplitude = 1.0 if transport_only else 0.1
    q = torch.randn(B, HK, seqlen, K, dtype=torch.bfloat16, device="npu") * amplitude
    k = torch.randn(B, HK, seqlen, K, dtype=torch.bfloat16, device="npu") * amplitude
    v = torch.randn(B, HV, seqlen, V, dtype=torch.bfloat16, device="npu") * amplitude
    h = torch.randn(B, HV, num_chunks, V, K, dtype=torch.bfloat16, device="npu") * amplitude
    g = torch.randn(B, HV, seqlen, dtype=torch.float32, device="npu")
    if not transport_only:
        g *= 0.1

    out = _call_chunk_fwd_o(
        q, k, v, h, g, scale, cu_seqlens=None, chunk_indices=None, chunk_size=chunk_size
    )
    if transport_only:
        return True

    ok = True
    for chunk_idx in range(num_chunks):
        token_begin = chunk_idx * chunk_size
        chunk_len = min(chunk_size, seqlen - token_begin)
        for hv in range(HV):
            q_pad = torch.zeros(BT, K, dtype=torch.bfloat16)
            k_pad = torch.zeros(BT, K, dtype=torch.bfloat16)
            g_pad = torch.zeros(BT, dtype=torch.float32)
            hk_idx = hv // g_ratio
            q_pad[:chunk_len] = q[0, hk_idx, token_begin : token_begin + chunk_len].cpu()
            k_pad[:chunk_len] = k[0, hk_idx, token_begin : token_begin + chunk_len].cpu()
            g_pad[:chunk_len] = g[0, hv, token_begin : token_begin + chunk_len].cpu()
            # Physical H is [V,K]; the math consumes H^T as [K,V].
            h_chunk = h[0, hv, chunk_idx].cpu().T.contiguous()
            v_pad = torch.zeros(BT, V, dtype=torch.bfloat16)
            v_pad[:chunk_len] = v[0, hv, token_begin : token_begin + chunk_len].cpu()
            ref = _cpu_output_ref(
                q_pad,
                k_pad,
                v_pad,
                h_chunk,
                g_pad,
                chunk_len,
                scale=scale,
                use_exp2=use_exp2,
            )
            tag = f"c{chunk_idx}/h{hv}"
            out_chunk = out[0, token_begin : token_begin + chunk_len, hv].cpu().float().numpy()
            ok &= _compare(
                f"O[{tag}]",
                ref[:chunk_len],
                out_chunk,
                atol=0.01,
                rtol=0.01,
            )
    return ok


DEFAULT_CASE_TIMEOUT_SEC = 60

DEFAULT_TRANSPORT_CASES = [
    ("G=1 T=64 exp2", dict(seqlen=64, heads=2, use_exp2=True, seed=1)),
    ("G=1 T=64 four-head ping-pong", dict(seqlen=64, heads=4, use_exp2=True, seed=4)),
    ("G=1 HK16 HV16 T=64", dict(seqlen=64, hk=16, hv=16, use_exp2=True, seed=10)),
    ("G=2 HK16 HV32 T=64", dict(seqlen=64, hk=16, hv=32, use_exp2=True, seed=11)),
    ("G=3 HK16 HV48 T=64", dict(seqlen=64, hk=16, hv=48, use_exp2=True, seed=12)),
]

DEFAULT_PRECISION_CASES = [
    ("G=1 T=64 exp2", dict(seqlen=64, heads=2, use_exp2=True, seed=1)),
    ("G=1 T=64 four-head ping-pong", dict(seqlen=64, heads=4, use_exp2=True, seed=4)),
    ("G=2 HK16 HV32 T=64", dict(seqlen=64, hk=16, hv=32, use_exp2=True, seed=11)),
]

ALL_NAMED_CASES = DEFAULT_TRANSPORT_CASES + [
    ("G=1 T=128 exp2", dict(seqlen=128, heads=2, use_exp2=True, seed=2)),
    ("G=1 T=256 exp2", dict(seqlen=256, heads=2, use_exp2=True, seed=3)),
    ("G=2 HK16 HV32 T=128", dict(seqlen=128, hk=16, hv=32, use_exp2=True, seed=13)),
    ("G=2 HK16 HV32 T=256", dict(seqlen=256, hk=16, hv=32, use_exp2=True, seed=14)),
    ("G=1 T=65 tail", dict(seqlen=65, heads=4, use_exp2=True, seed=15)),
    ("G=4 HK8 HV32 T=73 tail", dict(seqlen=73, hk=8, hv=32, use_exp2=True, seed=16)),
]

CASE_TIMEOUTS = {name: DEFAULT_CASE_TIMEOUT_SEC for name, _ in ALL_NAMED_CASES}


def _run_case_with_timeout(
    name: str,
    kwargs: dict,
    timeout_sec: float,
    *,
    transport_only: bool = False,
) -> bool:
    case_kwargs = dict(kwargs)
    if transport_only:
        case_kwargs["transport_only"] = True
    payload = json.dumps({"name": name, **case_kwargs})
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--single-json", payload]
    if transport_only:
        cmd.append("--transport")
    try:
        result = subprocess.run(cmd, timeout=timeout_sec)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name} exceeded {timeout_sec:.0f}s — subprocess killed")
        return False


def _run_single_from_json(payload: str) -> int:
    spec = json.loads(payload)
    name = spec.pop("name", "single")
    print(f"\n=== {name} ===")
    try:
        ok = run_case(**spec)
    except Exception as exc:
        print(f"[ERROR] {name}: {exc}")
        return 1
    if not ok:
        print(f"[FAIL] {name}")
        return 1
    print(f"[PASS] {name}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="chunk_fwd_o final-output precision checks")
    parser.add_argument("--single-json", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Run only named cases (substring match). Default: all.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_CASE_TIMEOUT_SEC,
        help=f"Per-case timeout in seconds (default {DEFAULT_CASE_TIMEOUT_SEC}).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first 2-head T=64 case.",
    )
    parser.add_argument(
        "--transport",
        action="store_true",
        help="Kernel launch + NPU sync only.",
    )
    parser.add_argument(
        "--precision",
        action="store_true",
        help="Compare the final BSND output against a CPU reference.",
    )
    args = parser.parse_args()
    if args.single_json is not None:
        sys.exit(_run_single_from_json(args.single_json))

    transport_only = args.transport or not args.precision
    pool = ALL_NAMED_CASES
    cases = list(DEFAULT_TRANSPORT_CASES if transport_only else DEFAULT_PRECISION_CASES)
    if args.smoke:
        cases = cases[:1]
    elif args.cases:
        selected = []
        for name, kwargs in pool:
            if any(token in name for token in args.cases):
                selected.append((name, kwargs))
        cases = selected
        if not cases:
            print(f"No cases matched: {args.cases}")
            sys.exit(2)

    failed = []
    for name, kwargs in cases:
        timeout_sec = CASE_TIMEOUTS.get(name, args.timeout)
        mode = "transport" if transport_only else "precision"
        print(f"\n=== {name} ({mode}, timeout={timeout_sec:.0f}s) ===")
        try:
            if not _run_case_with_timeout(
                name,
                kwargs,
                timeout_sec,
                transport_only=transport_only,
            ):
                failed.append(name)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")
            failed.append(name)

    if failed:
        print(f"\nFAILED: {failed}")
        sys.exit(1)
    if transport_only:
        print("\nAll transport checks passed.")
    else:
        print("\nAll final-output precision checks passed.")


if __name__ == "__main__":
    main()
