/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 L1<->UB chunk_fwd_o path.
 * Bring-up scheduler: run only Stage1 and Stage2 with a paired AIC/AIV
 * direct-UB handshake. Later stages stay disabled until these two run.
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
    static constexpr bool kEnableStage2 = true;

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
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t aicCoreIdx = AscendC::GetBlockIdx();
        const uint32_t aicCoreNum = AscendC::GetBlockNum();
        const uint32_t aivCoreIdx = AscendC::GetBlockIdx() / subBlockNum;
        const uint32_t aivCoreNum = AscendC::GetBlockNum() / subBlockNum;

        if ASCEND_IS_AIV {
            AscendC::TPipe pipe;
            ChunkFwdOA5VectorProcess<GT, UseExp2> vector(q_, k_, v_, h_, g_, cuSeqlens_, chunkOffsets_, o_,
                                                       workspace_);
            vector.Init(tiling_, &pipe);
            if (AscendC::GetSubBlockIdx() == 0) {
                vector.ProcessInit();
            }
            RunStage1Aiv(vector, aivCoreIdx, aivCoreNum);
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
            for (int64_t hv = 0; hv < tiling_.vNumHead; ++hv) {
                const int64_t hk = hv / tiling_.hvPerHk;
                if (owns && AscendC::GetSubBlockIdx() == 0) {
                    vector.ProcessStage1(loopIdx, loc, hk, hv);
                }
                if (owns && kEnableStage2) {
                    vector.PrepareCubeUbForStage2();
                    vector.WaitStage2Ready();
                    vector.DumpStage2ARaw(loopIdx, hv);
                    vector.DumpStage2OSRaw(loopIdx, hv);
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
            for (int64_t hv = 0; hv < tiling_.vNumHead; ++hv) {
                const int64_t hk = hv / tiling_.hvPerHk;
                if (owns) {
                    cube.ProcessStage2(loopIdx, loc, hk, hv);
                }
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
