/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 L1<->UB chunk_fwd_o path.
 *
 * Bring-up status (WIP):
 * - Stage1: UB layout §5.3 + streamSlot prefetch/compute aligned (ProcessStage1Group).
 * - Stage2: streamSlot ping-pong wiring pending; enable after S1/S2 handshake validated.
 * - Stage3/4/5: not started.
 * - Risk: kernel may hang at synchronize until Stage2 streamSlot is fully integrated.
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
    static constexpr bool kEnableStage1 = false;
    static constexpr bool kEnableStage2 = false;
    static constexpr bool kEnableStage3 = false;

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
        const uint32_t aivCoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
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
            if constexpr (kEnableStage2) {
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
                vector.ProcessStage1Group(loopIdx, loc, hvBase, taskCount, subBlockIdx);
                vector.SignalStage1GroupDone();
                // Stage2 consumer loop: per-head IDs allow AIC Stage2(head N)
                // to overlap AIV Stage1(head N+1) without flag reuse.
                for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
                    vector.WaitStage2Ready(static_cast<uint32_t>(headOffset));
                    const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % 2);
                    if (ownerSubBlock == subBlockIdx) {
                        const uint32_t localSlot = static_cast<uint32_t>(headOffset / 2);
                        const int64_t hv = hvBase + headOffset;
                        vector.DumpStage1Result(loopIdx, loc, hv, localSlot);
                        vector.DumpStage2ARaw(loopIdx, hv, localSlot);
                        vector.DumpStage2OSRaw(loopIdx, hv, localSlot);
                        if constexpr (kEnableStage3) {
                            const int64_t hk = hv / tiling_.hvPerHk;
                            vector.ProcessStage3(loopIdx, loc, hk, hv);
                        }
                    }
                    vector.SignalStage2Consumed(static_cast<uint32_t>(headOffset));
                }
                vector.ReleaseStage2Group();
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
                for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
                    if (headOffset >= 2) {
                        cube.WaitStage2Consumed(static_cast<uint32_t>(headOffset - 2));
                    }
                    cube.WaitStage1Ready(static_cast<uint32_t>(headOffset));
                    const int64_t hv = hvBase + headOffset;
                    const int64_t hk = hv / tiling_.hvPerHk;
                    const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % 2);
                    const uint32_t localSlot = static_cast<uint32_t>(headOffset / 2);
                    cube.ProcessStage2(
                        loopIdx, loc, hk, hv, ownerSubBlock, localSlot, static_cast<uint32_t>(headOffset));
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
