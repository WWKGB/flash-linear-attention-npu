/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 L1<->UB chunk_fwd_o path.
 *
 * Stage1/2: validated (precision passed); do not regress.
 * Stage2 (AIC): L0/L1 streamSlot ping-pong + per-head CrossCore (59e83dc / PR404).
 * Stage3 (AIV): per-head S1→WaitStage2→S3, mode=0x2 pair handshake and
 *   independent A-prime/V-to-MTE3 ping-pong, without SyncAll.
 * Stage4 (AIC): A-prime/V MTE2 prefetch + independent L0 ping-pong + FP32 Fixpipe to owner AIV.
 * Stage5 (AIV): owner fuses O_s-prime/O_l in FP32 and writes bf16 O through MTE3.
 */

#ifndef CHUNK_FWD_O_ARCH35_A5_H
#define CHUNK_FWD_O_ARCH35_A5_H

#include "kernel_operator.h"
#include "../chunk_fwd_o_struct.h"
#include "chunk_fwd_o_common.h"
#include "chunk_fwd_o_cube.h"
#include "chunk_fwd_o_vector.h"

namespace GDN {

template <typename GT, bool UseExp2>
class ChunkFwdOA5 {
public:
    static constexpr bool kEnableStage1 = true;
    static constexpr bool kEnableStage2 = true;
    static constexpr bool kEnableStage3 = true;
    static constexpr bool kEnableStage4 = true;
    static constexpr bool kEnableStage5 = true;

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g, GM_ADDR cuSeqlens,
                                GM_ADDR chunkOffsets, GM_ADDR o, GM_ADDR workspace,
                                const ChunkFwdOTilingData *tilingData)
    {
        q_ = q;
        k_ = k;
        v_ = v;
        h_ = h;
        g_ = g;
        cuSeqlens_ = cuSeqlens;
        chunkOffsets_ = chunkOffsets;
        o_ = o;
        workspace_ = workspace;
        tiling_ = *tilingData;
    }

    __aicore__ inline void Process()
    {
        const uint32_t aicCoreIdx = AscendC::GetBlockIdx();
        const uint32_t aicCoreNum = AscendC::GetBlockNum();
        const uint32_t aivCoreIdx = AscendC::GetBlockIdx() / 2U;
        const uint32_t aivCoreNum = AscendC::GetBlockNum();

        if ASCEND_IS_AIV {
            AscendC::TPipe pipe;
            ChunkFwdOA5VectorProcess<GT, UseExp2> vector(q_, k_, v_, h_, g_, cuSeqlens_, chunkOffsets_, o_,
                                                       workspace_);
            vector.Init(tiling_, &pipe);
            if (AscendC::GetSubBlockIdx() == 0) {
                vector.ProcessInit();
            }
            if constexpr (kEnableStage1) {
                RunStage1Aiv(vector, aivCoreIdx, aivCoreNum);
            }
            return;
        }

        if ASCEND_IS_AIC {
            if constexpr (kEnableStage2 || kEnableStage4) {
                ChunkFwdOA5CubeProcess cube(q_, k_, v_, h_, g_, cuSeqlens_, chunkOffsets_, o_, workspace_);
                cube.Init(tiling_);
                RunStage2Aic(cube, aicCoreIdx, aicCoreNum);
            }
        }
    }

private:
    __aicore__ inline void RunStage1Aiv(ChunkFwdOA5VectorProcess<GT, UseExp2> &vector, uint32_t coreIdx,
                                       uint32_t coreNum)
    {
        ChunkFwdOChunkLoc loc;
        for (uint32_t loopIdx = 0; loopIdx < static_cast<uint32_t>(tiling_.chunkNum); ++loopIdx) {
            ChunkFwdOResolveChunkLoc(cuSeqlens_, chunkOffsets_, tiling_, loopIdx, loc);
            const bool owns = (coreIdx == (loopIdx % coreNum));
            if (!owns) {
                continue;
            }
            const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            for (int64_t hvBase = 0; hvBase < tiling_.vNumHead; hvBase += tiling_.taskGroupSize) {
                const int64_t remaining = tiling_.vNumHead - hvBase;
                const int64_t taskCount = remaining < tiling_.taskGroupSize ? remaining : tiling_.taskGroupSize;
                vector.BeginStage1GroupPrefetch(loc, hvBase, taskCount, subBlockIdx);
                if constexpr (kEnableStage3) {
                    vector.BeginStage3Group();
                }
                // AIV per-head pipeline (PR404 V-side): owner computes; both subblocks
                // lock-step on headOffset. SignalStage1Ready / SignalStage2Consumed use
                // mode=0x2 (non-owner participates without compute). No SyncAll.
                // Finish S3 for head N before S1 reuses localSlot(N+2).
                for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
                    const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % 2);
                    const uint32_t localSlot = static_cast<uint32_t>(headOffset / 2);
                    const int64_t hv = hvBase + headOffset;
                    if (ownerSubBlock == subBlockIdx) {
                        vector.ProcessStage1Head(loopIdx, loc, hvBase, headOffset, taskCount, subBlockIdx);
                    }
                    vector.SignalStage1Ready(static_cast<uint32_t>(headOffset));
                    if constexpr (kEnableStage2) {
                        vector.WaitStage2Ready(static_cast<uint32_t>(headOffset));
                        if (ownerSubBlock == subBlockIdx) {
                            if constexpr (kEnableStage3) {
                                const int64_t hk = hv / tiling_.hvPerHk;
                                vector.ProcessStage3(loopIdx, loc, hk, hv, localSlot,
                                                     static_cast<uint32_t>(headOffset));
                            }
                        }
                        if constexpr (kEnableStage4) {
                            vector.SignalStage3Ready(static_cast<uint32_t>(headOffset));
                        }
                        vector.SignalStage2Consumed(static_cast<uint32_t>(headOffset));
                    }
                }
                if constexpr (kEnableStage3) {
                    vector.FinishStage3Group();
                }
                if constexpr (kEnableStage4) {
                    for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
                        const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % 2);
                        const uint32_t localSlot = static_cast<uint32_t>(headOffset / 2);
                        const int64_t hv = hvBase + headOffset;
                        vector.WaitStage4Ready(static_cast<uint32_t>(headOffset));
                        if (ownerSubBlock == subBlockIdx) {
                            vector.DumpStage4Result(loopIdx, hv, localSlot);
                            if constexpr (kEnableStage5) {
                                vector.ProcessStage5(loc, hv, localSlot);
                            }
                        }
                    }
                }
                if constexpr (kEnableStage2 || kEnableStage4) {
                    vector.ReleaseStage2Group();
                }
            }
        }
    }

    __aicore__ inline void RunStage2Aic(ChunkFwdOA5CubeProcess &cube, uint32_t coreIdx, uint32_t coreNum)
    {
        ChunkFwdOChunkLoc loc;
        for (uint32_t loopIdx = 0; loopIdx < static_cast<uint32_t>(tiling_.chunkNum); ++loopIdx) {
            ChunkFwdOResolveChunkLoc(cuSeqlens_, chunkOffsets_, tiling_, loopIdx, loc);
            const bool owns = (coreIdx == (loopIdx % coreNum));
            if (!owns) {
                continue;
            }
            for (int64_t hvBase = 0; hvBase < tiling_.vNumHead; hvBase += tiling_.taskGroupSize) {
                const int64_t remaining = tiling_.vNumHead - hvBase;
                const int64_t taskCount = remaining < tiling_.taskGroupSize ? remaining : tiling_.taskGroupSize;
                if constexpr (kEnableStage2) {
                    cube.ProcessStage2Group(loopIdx, loc, hvBase, taskCount);
                }
                if constexpr (kEnableStage4) {
                    cube.ProcessStage4Group(loopIdx, loc, hvBase, taskCount);
                }
                cube.WaitStage2GroupRelease();
            }
        }
    }

    GM_ADDR q_ = nullptr;
    GM_ADDR k_ = nullptr;
    GM_ADDR v_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR g_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkOffsets_ = nullptr;
    GM_ADDR o_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    ChunkFwdOTilingData tiling_{};
};

template <typename GT, bool UseExp2>
__aicore__ inline void ChunkFwdOA5Dispatch(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                           GM_ADDR cuSeqlens, GM_ADDR chunkOffsets, GM_ADDR o, GM_ADDR workspace,
                                           const ChunkFwdOTilingData *tilingData)
{
    ChunkFwdOA5<GT, UseExp2> op;
    op.Init(q, k, v, h, g, cuSeqlens, chunkOffsets, o, workspace, tilingData);
    op.Process();
}

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_A5_H
