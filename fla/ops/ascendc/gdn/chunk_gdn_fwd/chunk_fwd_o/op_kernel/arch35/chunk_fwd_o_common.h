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

// UB layout (bytes, single AIV). See chunk_fwd_o_ascendc-design.md §5.5.
constexpr uint32_t CHUNK_FWD_O_UB_MASK_BYTES = 4U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_GATE_BYTES = 64U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_CUBE_BASE = 68U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_ARAW_BYTES = 8U * 1024U;   // 64x64 bf16
constexpr uint32_t CHUNK_FWD_O_UB_OSRAW_BYTES = 32U * 1024U; // 64x128 fp32
constexpr uint32_t CHUNK_FWD_O_UB_ARAW_OFFSET = CHUNK_FWD_O_UB_CUBE_BASE;
constexpr uint32_t CHUNK_FWD_O_UB_OSRAW_OFFSET = CHUNK_FWD_O_UB_ARAW_OFFSET + CHUNK_FWD_O_UB_ARAW_BYTES;
constexpr uint32_t CHUNK_FWD_O_UB_OSPRIME_OFFSET = 116U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_OSPRIME_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_APRIME_FP32_OFFSET = 148U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_APRIME_FP32_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_STAGE3_END =
    CHUNK_FWD_O_UB_APRIME_FP32_OFFSET + CHUNK_FWD_O_UB_APRIME_FP32_BYTES;

// L1 scratch for S2 (KiB layout in design §Stage2).
constexpr uint32_t CHUNK_FWD_O_L1_Q_OFFSET = 0U;
constexpr uint32_t CHUNK_FWD_O_L1_K_OFFSET = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_H_OFFSET = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L0C_QH_OFFSET = 16U * 1024U;

// FixPipe -> UB handshake (mode 0x4, one AIC + two AIV subblocks).
// Use non-zero flag IDs (same band as chunk_kda_fwd_fwd_h direct UB: 6/7).
constexpr uint64_t CHUNK_FWD_O_CUBE_UB_FREE_FLAG = 6;
constexpr uint64_t CHUNK_FWD_O_CUBE_UB_READY_FLAG = 7;
constexpr uint64_t CHUNK_FWD_O_SUBBLOCK_FLAG_STRIDE = 16;

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

struct ChunkFwdODebugHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t sysWorkspaceSize;
    uint32_t debugDumpOffset;
    uint32_t slotBytes;
    uint32_t headerBytes;
    int64_t chunkNum;
    int64_t vNumHead;
    int64_t bt;
    int64_t kDim;
    int64_t vDim;
};

__aicore__ inline bool ChunkFwdODumpEnabled(const ChunkFwdOTilingData &tiling)
{
    return tiling.debugDumpOffset > 0 && tiling.debugDumpSlotBytes > 0;
}

__aicore__ inline int64_t ChunkFwdODumpSlotIndex(const ChunkFwdOTilingData &tiling, uint32_t loopIdx, int64_t hv)
{
    return static_cast<int64_t>(loopIdx) * tiling.vNumHead + hv;
}

__aicore__ inline void ChunkFwdOWriteDebugHeader(GM_ADDR userWorkspace, const ChunkFwdOTilingData &tiling)
{
    if (!ChunkFwdODumpEnabled(tiling)) {
        return;
    }
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        AscendC::GlobalTensor<uint32_t> hdrGm;
        hdrGm.SetGlobalBuffer((__gm__ uint32_t *)userWorkspace);
        hdrGm.SetValue(0, CHUNK_FWD_O_DBG_MAGIC);
        hdrGm.SetValue(1, CHUNK_FWD_O_DBG_VERSION);
        hdrGm.SetValue(2, static_cast<uint32_t>(tiling.debugDumpOffset));
        hdrGm.SetValue(3, static_cast<uint32_t>(tiling.debugDumpOffset));
        hdrGm.SetValue(4, static_cast<uint32_t>(tiling.debugDumpSlotBytes));
        hdrGm.SetValue(5, CHUNK_FWD_O_DBG_HEADER_BYTES);

        AscendC::GlobalTensor<int64_t> hdr64Gm;
        hdr64Gm.SetGlobalBuffer((__gm__ int64_t *)(userWorkspace + 24));
        hdr64Gm.SetValue(0, tiling.chunkNum);
        hdr64Gm.SetValue(1, tiling.vNumHead);
        hdr64Gm.SetValue(2, CHUNK_FWD_O_A5_BT);
        hdr64Gm.SetValue(3, tiling.kHeadDim);
        hdr64Gm.SetValue(4, tiling.vHeadDim);
    }
}

__aicore__ inline GM_ADDR ChunkFwdODumpSlotPtr(GM_ADDR userWorkspace, const ChunkFwdOTilingData &tiling,
                                               int64_t slotIdx)
{
    return userWorkspace + CHUNK_FWD_O_DBG_HEADER_BYTES + slotIdx * tiling.debugDumpSlotBytes;
}

template <typename T>
__aicore__ inline void ChunkFwdODumpUbToGm(GM_ADDR slotBase, uint32_t slotOff, AscendC::LocalTensor<T> &src,
                                           uint32_t count)
{
    AscendC::GlobalTensor<T> dstGm;
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(slotBase + slotOff));
    DataCopy(dstGm, src, count);
}

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_COMMON_H
