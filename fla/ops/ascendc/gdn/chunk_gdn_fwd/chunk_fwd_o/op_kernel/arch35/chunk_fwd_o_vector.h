/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef CHUNK_FWD_O_ARCH35_VECTOR_H
#define CHUNK_FWD_O_ARCH35_VECTOR_H

#include "kernel_operator.h"
#include "../chunk_fwd_o_struct.h"
#include "chunk_fwd_o_common.h"

namespace GDN {

using namespace AscendC;

// P1 · Init causal mask (kernel-level, once per launch).
__simd_vf__ inline void InitCausalMask(__ubuf__ uint8_t *maskAddr, int64_t seqlen, int64_t chunkSize)
{
    (void)maskAddr;
    (void)seqlen;
    (void)chunkSize;
}

// P2 · Stage1 gate precompute.
__simd_vf__ inline void Stage1Gate(__ubuf__ float *gateOAddr, __ubuf__ float *gateAAddr,
                                   __ubuf__ float *gAddr, int64_t chunkLen, bool useExp2)
{
    (void)gateOAddr;
    (void)gateAAddr;
    (void)gAddr;
    (void)chunkLen;
    (void)useExp2;
}

// P4 · Stage3 gate application + causal mask; A' -> L1, O_s' -> UB.
__simd_vf__ inline void Stage3GateMask(__ubuf__ float *aRawAddr, __ubuf__ float *oSRawAddr,
                                       __ubuf__ float *gateOAddr, __ubuf__ float *gateAAddr,
                                       __ubuf__ uint8_t *maskAddr, __ubuf__ float *oSPrimeAddr,
                                       int64_t chunkLen)
{
    (void)aRawAddr;
    (void)oSRawAddr;
    (void)gateOAddr;
    (void)gateAAddr;
    (void)maskAddr;
    (void)oSPrimeAddr;
    (void)chunkLen;
}

// P6 · Stage5 fuse and write back.
template <typename OutputT>
__simd_vf__ inline void Stage5Fuse(__ubuf__ float *oSPrimeAddr, __ubuf__ float *oLAddr, __ubuf__ OutputT *oOutAddr,
                                 float scale, int64_t chunkLen)
{
    (void)oSPrimeAddr;
    (void)oLAddr;
    (void)oOutAddr;
    (void)scale;
    (void)chunkLen;
}

template <typename InputT, typename GT>
class ChunkFwdOA5VectorProcess {
public:
    __aicore__ inline ChunkFwdOA5VectorProcess(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                               GM_ADDR cuSeqlens, GM_ADDR chunkOffsets, GM_ADDR o,
                                               GM_ADDR workspace)
        : q_(q), k_(k), v_(v), h_(h), g_(g), cuSeqlens_(cuSeqlens), chunkOffsets_(chunkOffsets), o_(o),
          workspace_(workspace)
    {
    }

    __aicore__ inline void Init(const ChunkFwdOTilingData &tiling, TPipe *pipe)
    {
        (void)pipe;
        tiling_ = &tiling;
    }

    __aicore__ inline void ProcessInit()
    {
        // P1 placeholder: materialize m_A into UB fixed region.
    }

    __aicore__ inline void ProcessStage1(int64_t chunkIdx)
    {
        (void)chunkIdx;
        // P2 placeholder.
    }

    __aicore__ inline void ProcessStage3(int64_t chunkIdx)
    {
        (void)chunkIdx;
        // P4 placeholder.
    }

    __aicore__ inline void ProcessStage5(int64_t chunkIdx)
    {
        (void)chunkIdx;
        // P6 placeholder.
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
    const ChunkFwdOTilingData *tiling_ = nullptr;
};

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_VECTOR_H
