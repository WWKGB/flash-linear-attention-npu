/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef CHUNK_FWD_O_ARCH35_COMMON_H
#define CHUNK_FWD_O_ARCH35_COMMON_H

#include <cstdint>
#include "kernel_operator.h"
#include "../chunk_fwd_o_struct.h"
#include "../chunk_fwd_o_a5_constants.h"

namespace GDN {

// Fixed tile geometry: see chunk_fwd_o_a5_constants.h

// 4-chunk pipeline window (sync_idx / la_slot domain).
constexpr int64_t CHUNK_FWD_O_PIPELINE_WINDOW = 4;

// CrossCore hardware flag IDs (P7). Same pattern as chunk_gated_delta_rule_fwd_h / recompute_w_u_fwd.
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

struct ChunkFwdOChunkLoc {
    uint32_t batchIdx;
    uint32_t localChunkIdx;
    uint32_t globalChunkIdx;
    uint32_t tokenStart;
    uint32_t tokenEnd;
    uint32_t chunkLen;
};

__aicore__ inline void ChunkFwdOResolveChunkLoc(GM_ADDR cuSeqlens, GM_ADDR chunkOffsets,
                                                const ChunkFwdOTilingData &tiling, uint32_t loopIdx,
                                                ChunkFwdOChunkLoc &loc)
{
    const uint32_t chunkSize = static_cast<uint32_t>(tiling.chunkSize);
    if (tiling.isVariedLen == 0) {
        const uint32_t chunksPerBatch = static_cast<uint32_t>(tiling.numChunksPerBatch);
        loc.batchIdx = loopIdx / chunksPerBatch;
        loc.localChunkIdx = loopIdx % chunksPerBatch;
        loc.globalChunkIdx = loopIdx;
        loc.tokenStart = loc.localChunkIdx * chunkSize;
        const uint32_t seqlen = static_cast<uint32_t>(tiling.seqlen);
        loc.tokenEnd = loc.tokenStart + chunkSize > seqlen ? seqlen : loc.tokenStart + chunkSize;
    } else {
        AscendC::GlobalTensor<int64_t> cuSeqlensTensor;
        AscendC::GlobalTensor<int64_t> chunkOffsetsTensor;
        cuSeqlensTensor.SetGlobalBuffer((__gm__ int64_t *)cuSeqlens);
        chunkOffsetsTensor.SetGlobalBuffer((__gm__ int64_t *)chunkOffsets);
        const uint32_t seqIdx = static_cast<uint32_t>(chunkOffsetsTensor.GetValue(static_cast<int64_t>(loopIdx) * 2));
        loc.localChunkIdx = static_cast<uint32_t>(chunkOffsetsTensor.GetValue(static_cast<int64_t>(loopIdx) * 2 + 1));
        const uint32_t curSeqBegin = static_cast<uint32_t>(cuSeqlensTensor.GetValue(seqIdx));
        const uint32_t curSeqEnd = static_cast<uint32_t>(cuSeqlensTensor.GetValue(seqIdx + 1));
        loc.batchIdx = 0;
        loc.globalChunkIdx = loopIdx;
        loc.tokenStart = curSeqBegin + loc.localChunkIdx * chunkSize;
        loc.tokenEnd = loc.tokenStart + chunkSize > curSeqEnd ? curSeqEnd : loc.tokenStart + chunkSize;
    }
    loc.chunkLen = loc.tokenEnd - loc.tokenStart;
}

__aicore__ inline int64_t ChunkFwdOQKOffset(const ChunkFwdOTilingData &tiling, const ChunkFwdOChunkLoc &loc,
                                            int64_t hk)
{
    const int64_t row = static_cast<int64_t>(loc.batchIdx) * tiling.kNumHead + hk;
    return (row * tiling.seqlen + static_cast<int64_t>(loc.tokenStart)) * tiling.kHeadDim;
}

__aicore__ inline int64_t ChunkFwdOVOOffset(const ChunkFwdOTilingData &tiling, const ChunkFwdOChunkLoc &loc,
                                            int64_t hv)
{
    const int64_t row = static_cast<int64_t>(loc.batchIdx) * tiling.vNumHead + hv;
    return (row * tiling.seqlen + static_cast<int64_t>(loc.tokenStart)) * tiling.vHeadDim;
}

__aicore__ inline int64_t ChunkFwdOGOffset(const ChunkFwdOTilingData &tiling, const ChunkFwdOChunkLoc &loc,
                                           int64_t hv)
{
    const int64_t row = static_cast<int64_t>(loc.batchIdx) * tiling.vNumHead + hv;
    return row * tiling.seqlen + static_cast<int64_t>(loc.tokenStart);
}

__aicore__ inline int64_t ChunkFwdOHOffset(const ChunkFwdOTilingData &tiling, const ChunkFwdOChunkLoc &loc,
                                           int64_t hv)
{
    if (tiling.isVariedLen != 0) {
        return (hv * tiling.chunkNum + static_cast<int64_t>(loc.globalChunkIdx)) * tiling.kHeadDim * tiling.vHeadDim;
    }
    return (static_cast<int64_t>(loc.batchIdx) * tiling.vNumHead * tiling.numChunksPerBatch +
            hv * tiling.numChunksPerBatch + static_cast<int64_t>(loc.localChunkIdx)) *
           tiling.kHeadDim * tiling.vHeadDim;
}

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_COMMON_H
