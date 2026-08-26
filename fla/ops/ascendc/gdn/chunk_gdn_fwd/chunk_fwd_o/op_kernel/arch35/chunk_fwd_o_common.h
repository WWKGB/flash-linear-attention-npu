/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef CHUNK_FWD_O_ARCH35_COMMON_H
#define CHUNK_FWD_O_ARCH35_COMMON_H

#include <cstdint>

namespace GDN {

// Fixed tile geometry for the A5 L1<->UB path (v1).
constexpr int64_t CHUNK_FWD_O_A5_BT = 64;
constexpr int64_t CHUNK_FWD_O_A5_K = 128;
constexpr int64_t CHUNK_FWD_O_A5_V = 128;

// 4-chunk pipeline window (sync_idx / la_slot domain).
constexpr int64_t CHUNK_FWD_O_PIPELINE_WINDOW = 4;

// CrossCore hardware flag IDs (P7). Same pattern as chunk_gated_delta_rule_fwd_h / recompute_w_u_fwd.
// recompute_w_u_fwd uses 3/5; gdn_fwd_h uses 0-7 — chunk_fwd_o allocates from 8 upward.
constexpr uint32_t CHUNK_FWD_O_CC_CUBE_READY_BASE = 8;    // AIC -> AIV, sync_idx in [0,3]
constexpr uint32_t CHUNK_FWD_O_CC_APRIME_READY_BASE = 12; // AIV -> AIC
constexpr uint32_t CHUNK_FWD_O_CC_OL_READY_BASE = 16;     // AIC -> AIV

__aicore__ inline uint32_t ChunkFwdOCubeReadyFlagId(int64_t syncIdx)
{
    return CHUNK_FWD_O_CC_CUBE_READY_BASE + static_cast<uint32_t>(syncIdx);
}

__aicore__ inline uint32_t ChunkFwdOAPrimeReadyFlagId(int64_t syncIdx)
{
    return CHUNK_FWD_O_CC_APRIME_READY_BASE + static_cast<uint32_t>(syncIdx);
}

__aicore__ inline uint32_t ChunkFwdOOLReadyFlagId(int64_t syncIdx)
{
    return CHUNK_FWD_O_CC_OL_READY_BASE + static_cast<uint32_t>(syncIdx);
}

__aicore__ inline int64_t ChunkFwdOSyncIdx(int64_t chunkIdx)
{
    return chunkIdx % CHUNK_FWD_O_PIPELINE_WINDOW;
}

__aicore__ inline int64_t ChunkFwdOAivId(int64_t chunkIdx)
{
    return chunkIdx % 2;
}

__aicore__ inline int64_t ChunkFwdOUbSlot(int64_t chunkIdx)
{
    return (chunkIdx / 2) % 2;
}

__aicore__ inline int64_t ChunkFwdOLaSlot(int64_t chunkIdx)
{
    return chunkIdx % CHUNK_FWD_O_PIPELINE_WINDOW;
}

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_COMMON_H
