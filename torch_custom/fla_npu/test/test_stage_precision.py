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
DBG_OSRAW_BYTES = BT * V * 4
DBG_MASK_OFF = 0
DBG_GATE_O_OFF = DBG_MASK_BYTES
DBG_GATE_A_OFF = DBG_GATE_O_OFF + DBG_GATE_O_BYTES
DBG_ARAW_OFF = DBG_GATE_A_OFF + DBG_GATE_A_BYTES
DBG_OSRAW_OFF = DBG_ARAW_OFF + DBG_ARAW_BYTES
DBG_SLOT_BYTES = ((DBG_OSRAW_OFF + DBG_OSRAW_BYTES + 511) // 512) * 512


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
        .view(np.float32)
        .reshape(BT, V)
        .copy()
    )
    return {"mask": mask, "gate_o": gate_o, "gate_A": gate_a, "A_raw": a_raw, "O_s_raw": o_s_raw}


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

    return {
        "mask": mask,
        "gate_o": gate_o.numpy(),
        "gate_A": gate_a.numpy(),
        "A_raw": a_raw,
        "O_s_raw": o_s_raw,
    }


def _compare(name: str, ref: np.ndarray, got: np.ndarray, *, atol: float, rtol: float) -> bool:
    if ref.shape != got.shape:
        print(f"[FAIL] {name}: shape ref={ref.shape} got={got.shape}")
        return False
    if ref.dtype != got.dtype and name != "mask":
        ref = ref.astype(got.dtype, copy=False)
    if name == "mask":
        bad = ref != got
    else:
        bad = ~np.isclose(ref, got, rtol=rtol, atol=atol, equal_nan=True)
    nbad = int(bad.sum())
    if nbad == 0:
        print(f"[PASS] {name}")
        return True
    max_diff = np.max(np.abs(ref.astype(np.float64) - got.astype(np.float64)))
    print(f"[FAIL] {name}: {nbad}/{ref.size} mismatches, max_abs_diff={max_diff:.6g}")
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


def run_case(*, seqlen=64, use_exp2: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    B, HK, HV = 1, 1, 1
    scale = K ** -0.5
    chunk_size = 64
    num_chunks = (seqlen + chunk_size - 1) // chunk_size

    q = torch.randn(B, HK, seqlen, K, dtype=torch.bfloat16, device="npu")
    k = torch.randn(B, HK, seqlen, K, dtype=torch.bfloat16, device="npu")
    v = torch.randn(B, HV, seqlen, V, dtype=torch.bfloat16, device="npu")
    h = torch.randn(B, HV, num_chunks, K, V, dtype=torch.bfloat16, device="npu")
    g = torch.randn(B, HV, seqlen, dtype=torch.float32, device="npu")

    _, workspace = _call_chunk_fwd_o_with_workspace(
        q, k, v, h, g, scale, cu_seqlens=None, chunk_indices=None, chunk_size=chunk_size
    )

    ws_np = workspace.cpu().numpy()
    _, user_ws = _find_user_workspace(ws_np)
    hdr = _parse_debug_header(user_ws)

    slot_idx = 0  # chunk 0, hv 0
    dump = _read_slot(user_ws, slot_idx, hdr["slot_bytes"])

    chunk_len = min(seqlen, chunk_size)
    q_cpu = q[0, 0, :chunk_len].cpu()
    if chunk_len < BT:
        q_pad = torch.zeros(BT, K, dtype=torch.bfloat16)
        k_pad = torch.zeros(BT, K, dtype=torch.bfloat16)
        h_chunk = h[0, 0, 0].cpu()
        g_pad = torch.zeros(BT, dtype=torch.float32)
        q_pad[:chunk_len] = q[0, 0, :chunk_len].cpu()
        k_pad[:chunk_len] = k[0, 0, :chunk_len].cpu()
        g_pad[:chunk_len] = g[0, 0, :chunk_len].cpu()
    else:
        q_pad = q[0, 0].cpu()
        k_pad = k[0, 0].cpu()
        g_pad = g[0, 0].cpu()
        h_chunk = h[0, 0, 0].cpu()

    ref = _cpu_stage_refs(q_pad, k_pad, h_chunk, g_pad, chunk_len, use_exp2=use_exp2)

    ok = True
    ok &= _compare("mask", ref["mask"], dump["mask"], atol=0, rtol=0)
    ok &= _compare("gate_o", ref["gate_o"], dump["gate_o"], atol=1e-3, rtol=1e-2)
    ok &= _compare("gate_A", ref["gate_A"], dump["gate_A"], atol=1e-2, rtol=1e-2)
    ok &= _compare(
        "A_raw",
        ref["A_raw"][:chunk_len, :chunk_len],
        dump["A_raw"][:chunk_len, :chunk_len],
        atol=0.05,
        rtol=0.01,
    )
    ok &= _compare(
        "O_s_raw",
        ref["O_s_raw"][:chunk_len],
        dump["O_s_raw"][:chunk_len],
        atol=0.05,
        rtol=0.01,
    )
    return ok


def main():
    cases = [
        ("T=64 exp2", dict(seqlen=64, use_exp2=True, seed=1)),
        ("T=48 tail exp2", dict(seqlen=48, use_exp2=True, seed=2)),
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
