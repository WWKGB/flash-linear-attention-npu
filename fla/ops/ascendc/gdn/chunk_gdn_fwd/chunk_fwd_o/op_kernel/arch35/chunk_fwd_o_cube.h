/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef CHUNK_FWD_O_ARCH35_CUBE_H
#define CHUNK_FWD_O_ARCH35_CUBE_H

#include "kernel_operator.h"
#include "../chunk_fwd_o_struct.h"
#include "chunk_fwd_o_common.h"

namespace GDN {

using namespace AscendC;

template <typename GT>
class ChunkFwdOA5CubeProcess {
public:
    __aicore__ inline ChunkFwdOA5CubeProcess(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                             GM_ADDR cuSeqlens, GM_ADDR chunkOffsets, GM_ADDR o,
                                             GM_ADDR workspace)
        : q_(q), k_(k), v_(v), h_(h), g_(g), cuSeqlens_(cuSeqlens), chunkOffsets_(chunkOffsets), o_(o),
          workspace_(workspace)
    {
    }

    __aicore__ inline void Init(const ChunkFwdOTilingData &tiling)
    {
        tiling_ = tiling;
    }

    __aicore__ inline void ProcessStage2(uint32_t loopIdx, const ChunkFwdOChunkLoc &loc, int64_t hk, int64_t hv)
    {
        (void)loopIdx;
        (void)loc;
        (void)hk;
        (void)hv;
        // P3 placeholder: C1 A_raw + C2 O_s_raw, FixPipe -> UB.
    }

    __aicore__ inline void ProcessStage4(uint32_t loopIdx, const ChunkFwdOChunkLoc &loc, int64_t hk, int64_t hv)
    {
        (void)loopIdx;
        (void)loc;
        (void)hk;
        (void)hv;
        // P5 placeholder: C3 O_l = scale * (A' @ V), FixPipe -> UB.
    }

private:
    GM_ADDR q_;
    GM_ADDR k_;
    GM_ADDR v_;
    GM_ADDR h_;
    GM_ADDR g_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkOffsets_;
    GM_ADDR o_;
    GM_ADDR workspace_;
    ChunkFwdOTilingData tiling_{};
};

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_CUBE_H
