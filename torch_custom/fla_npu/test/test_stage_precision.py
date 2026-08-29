#!/usr/bin/env python3
# Copyright (c) 2026 Tianjin University, Ltd.
"""Stage-level precision check for chunk_fwd_o A5 (Init / S1 / S2).

Runs the NPU kernel, reads intermediate tensors from the user workspace dump
region, and compares against CPU stage references.
"""

from __future__ import annotations

import ctypes
import struct
import sys
from pathlib import Path

import ml_dtypes
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

# Dump layout must match chunk_fwd_o_a5_constants.h
DBG_MAGIC = 0xCF0DA5
BT = 64
K = 128
V = 128
DBG_HEADER_BYTES = 64
DBG_MASK_BYTES = 4 * 1024
DBG_GATE_O_BYTES = BT * 4
DBG_GATE_A_BYTES = BT * BT * 4
DBG_ARAW_BYTES = BT * BT * 2
DBG_OSRAW_BYTES = BT * V * 2
DBG_APRIME_BYTES = BT * BT * 2
DBG_OSPRIME_BYTES = BT * V * 4
DBG_MASK_OFF = 0
DBG_GATE_O_OFF = DBG_MASK_BYTES
DBG_GATE_A_OFF = DBG_GATE_O_OFF + DBG_GATE_O_BYTES
DBG_ARAW_OFF = DBG_GATE_A_OFF + DBG_GATE_A_BYTES
DBG_OSRAW_OFF = DBG_ARAW_OFF + DBG_ARAW_BYTES
DBG_APRIME_OFF = DBG_OSRAW_OFF + DBG_OSRAW_BYTES
DBG_OSPRIME_OFF = DBG_APRIME_OFF + DBG_APRIME_BYTES
DBG_SLOT_BYTES = ((DBG_OSPRIME_OFF + DBG_OSPRIME_BYTES + 511) // 512) * 512


def _call_chunk_fwd_o_with_workspace(q, k, v, h, g, scale, *, cu_seqlens, chunk_indices, chunk_size=64):
    import torch

    out = torch.empty_like(v)
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
    return out, workspace


def _parse_debug_header(user_ws: np.ndarray) -> dict:
    if user_ws.nbytes < DBG_HEADER_BYTES:
        raise RuntimeError("workspace too small for debug header")
    magic, version, sys_ws, dump_off, slot_bytes, header_bytes, chunk_num, v_num_head, bt, k_dim, v_dim = (
        struct.unpack_from("<IIIIIIqqqqq", user_ws, 0)
    )
    if magic != DBG_MAGIC:
        raise RuntimeError(f"debug header magic mismatch: got 0x{magic:08x}, expect 0x{DBG_MAGIC:08x}")
    return {
        "version": version,
        "sys_workspace_size": sys_ws,
        "debug_dump_offset": dump_off,
        "slot_bytes": slot_bytes,
        "header_bytes": header_bytes,
        "chunk_num": chunk_num,
        "v_num_head": v_num_head,
        "bt": bt,
        "k_dim": k_dim,
        "v_dim": v_dim,
    }


def _read_slot(user_ws: np.ndarray, slot_idx: int, slot_bytes: int) -> dict:
    base = DBG_HEADER_BYTES + slot_idx * slot_bytes
    end = base + slot_bytes
    if end > user_ws.nbytes:
        raise RuntimeError(f"slot {slot_idx} out of range: need {end}, have {user_ws.nbytes}")
    chunk = user_ws[base:end]
    mask = chunk[DBG_MASK_OFF : DBG_MASK_OFF + DBG_MASK_BYTES].view(np.uint8).reshape(BT, BT)
    gate_o = chunk[DBG_GATE_O_OFF : DBG_GATE_O_OFF + DBG_GATE_O_BYTES].view(np.float32).copy()
    gate_a = chunk[DBG_GATE_A_OFF : DBG_GATE_A_OFF + DBG_GATE_A_BYTES].view(np.float32).reshape(BT, BT).copy()
    a_raw = (
        chunk[DBG_ARAW_OFF : DBG_ARAW_OFF + DBG_ARAW_BYTES]
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
        .reshape(BT, BT)
        .copy()
    )
    o_s_raw = (
        chunk[DBG_OSRAW_OFF : DBG_OSRAW_OFF + DBG_OSRAW_BYTES]
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
        .reshape(BT, V)
        .copy()
    )
    a_prime = (
        chunk[DBG_APRIME_OFF : DBG_APRIME_OFF + DBG_APRIME_BYTES]
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
        .reshape(BT, BT)
        .copy()
    )
    o_s_prime = (
        chunk[DBG_OSPRIME_OFF : DBG_OSPRIME_OFF + DBG_OSPRIME_BYTES]
        .view(np.float32)
        .reshape(BT, V)
        .copy()
    )
    return {
        "mask": mask,
        "gate_o": gate_o,
        "gate_A": gate_a,
        "A_raw": a_raw,
        "O_s_raw": o_s_raw,
        "A_prime": a_prime,
        "O_s_prime": o_s_prime,
    }


def _cpu_stage_refs(q, k, h, g, chunk_len: int, *, use_exp2: bool = False):
    """Stage references aligned with kernel (h layout [K,V], exp by default)."""
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
    mask = (causal & valid_2d).numpy().astype(np.uint8)

    q_bf = torch.zeros(bt, K, dtype=torch.bfloat16)
    k_bf = torch.zeros(bt, K, dtype=torch.bfloat16)
    q_bf[:chunk_len] = q[:chunk_len].to(torch.bfloat16)
    k_bf[:chunk_len] = k[:chunk_len].to(torch.bfloat16)
    h_bf = h.to(torch.bfloat16)
    a_raw = (q_bf @ k_bf.T).float().numpy()
    o_s_raw = (q_bf @ h_bf).float().numpy()
    a_prime_fp32 = torch.from_numpy(a_raw) * gate_a * torch.from_numpy(mask).float()
    a_prime = a_prime_fp32.to(torch.bfloat16).float().numpy()
    o_s_prime = torch.from_numpy(o_s_raw) * gate_o.unsqueeze(1)

    return {
        "mask": mask,
        "gate_o": gate_o.numpy(),
        "gate_A": gate_a.numpy(),
        "A_raw": a_raw,
        "O_s_raw": o_s_raw,
        "A_prime": a_prime,
        "O_s_prime": o_s_prime.numpy(),
    }


def _compare(name: str, ref: np.ndarray, got: np.ndarray, *, atol: float, rtol: float) -> bool:
    if ref.shape != got.shape:
        print(f"[FAIL] {name}: shape ref={ref.shape} got={got.shape}")
        return False
    is_mask = name == "mask" or name.startswith("mask[")
    if ref.dtype != got.dtype and not is_mask:
        ref = ref.astype(got.dtype, copy=False)
    if is_mask:
        bad = ref != got
    else:
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


def _find_user_workspace(ws_np: np.ndarray) -> tuple[int, np.ndarray]:
    for off in range(0, max(1, ws_np.nbytes - 4), 4):
        if struct.unpack_from("<I", ws_np, off)[0] == DBG_MAGIC:
            user_ws = ws_np[off:]
            hdr = _parse_debug_header(user_ws)
            if off != hdr["sys_workspace_size"]:
                print(
                    f"warning: debug header at {off}, "
                    f"sys_workspace_size={hdr['sys_workspace_size']}"
                )
            return off, user_ws
    raise RuntimeError("debug header magic not found in workspace")


def run_case(*, seqlen=64, heads=2, use_exp2: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    B, HK, HV = 1, heads, heads
    scale = K ** -0.5
    chunk_size = 64
    num_chunks = (seqlen + chunk_size - 1) // chunk_size

    q = torch.randn(B, HK, seqlen, K, dtype=torch.bfloat16, device="npu")
    k = torch.randn(B, HK, seqlen, K, dtype=torch.bfloat16, device="npu")
    v = torch.randn(B, HV, seqlen, V, dtype=torch.bfloat16, device="npu")
    h = torch.randn(B, HV, num_chunks, V, K, dtype=torch.bfloat16, device="npu")
    g = torch.randn(B, HV, seqlen, dtype=torch.float32, device="npu")

    _, workspace = _call_chunk_fwd_o_with_workspace(
        q, k, v, h, g, scale, cu_seqlens=None, chunk_indices=None, chunk_size=chunk_size
    )

    ws_np = workspace.cpu().numpy()
    _, user_ws = _find_user_workspace(ws_np)
    hdr = _parse_debug_header(user_ws)

    ok = True
    refs = {}
    dumps = {}
    for chunk_idx in range(num_chunks):
        token_begin = chunk_idx * chunk_size
        chunk_len = min(chunk_size, seqlen - token_begin)
        for hv in range(HV):
            slot_idx = chunk_idx * HV + hv
            dump = _read_slot(user_ws, slot_idx, hdr["slot_bytes"])
            q_pad = torch.zeros(BT, K, dtype=torch.bfloat16)
            k_pad = torch.zeros(BT, K, dtype=torch.bfloat16)
            g_pad = torch.zeros(BT, dtype=torch.float32)
            q_pad[:chunk_len] = q[0, hv, token_begin : token_begin + chunk_len].cpu()
            k_pad[:chunk_len] = k[0, hv, token_begin : token_begin + chunk_len].cpu()
            g_pad[:chunk_len] = g[0, hv, token_begin : token_begin + chunk_len].cpu()
            # Physical H is [V,K]; the math consumes H^T as [K,V].
            h_chunk = h[0, hv, chunk_idx].cpu().T.contiguous()
            ref = _cpu_stage_refs(q_pad, k_pad, h_chunk, g_pad, chunk_len, use_exp2=use_exp2)
            refs[(chunk_idx, hv)] = ref
            dumps[(chunk_idx, hv)] = dump
            tag = f"c{chunk_idx}/h{hv}"
            ok &= _compare(f"mask[{tag}]", ref["mask"], dump["mask"], atol=0, rtol=0)
            ok &= _compare(f"gate_o[{tag}]", ref["gate_o"], dump["gate_o"], atol=1e-3, rtol=1e-2)
            ok &= _compare(f"gate_A[{tag}]", ref["gate_A"], dump["gate_A"], atol=1e-2, rtol=1e-2)
            ok &= _compare(
                f"A_raw[{tag}]",
                ref["A_raw"][:chunk_len, :chunk_len],
                dump["A_raw"][:chunk_len, :chunk_len],
                atol=0.05,
                rtol=0.01,
            )
            ok &= _compare(
                f"O_s_raw[{tag}]",
                ref["O_s_raw"][:chunk_len],
                dump["O_s_raw"][:chunk_len],
                atol=0.05,
                rtol=0.01,
            )
    if not ok and HV >= 2:
        for chunk_idx in range(num_chunks):
            for got_hv in range(HV):
                got = dumps[(chunk_idx, got_hv)]["O_s_raw"]
                scores = []
                for ref_hv in range(HV):
                    ref_value = refs[(chunk_idx, ref_hv)]["O_s_raw"]
                    close = int(np.isclose(got, ref_value, atol=0.05, rtol=0.01).sum())
                    scores.append((close, ref_hv))
                scores.sort(reverse=True)
                print(
                    f"[DIAG] c{chunk_idx}/h{got_hv} best O_s_raw reference: "
                    f"h{scores[0][1]} close={scores[0][0]}/{got.size}"
                )
    return ok


def main():
    cases = [
        ("G=1 T=64 exp2", dict(seqlen=64, heads=2, use_exp2=True, seed=1)),
        ("G=1 T=128 exp2", dict(seqlen=128, heads=2, use_exp2=True, seed=2)),
        ("G=1 T=256 exp2", dict(seqlen=256, heads=2, use_exp2=True, seed=3)),
        ("G=1 T=64 four-head ping-pong", dict(seqlen=64, heads=4, use_exp2=True, seed=4)),
    ]
    failed = []
    for name, kwargs in cases:
        print(f"\n=== {name} ===")
        try:
            if not run_case(**kwargs):
                failed.append(name)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")
            failed.append(name)

    if failed:
        print(f"\nFAILED: {failed}")
        sys.exit(1)
    print("\nAll stage precision checks passed.")


if __name__ == "__main__":
    main()
