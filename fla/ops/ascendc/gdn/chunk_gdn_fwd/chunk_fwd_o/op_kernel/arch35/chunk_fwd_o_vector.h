/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef CHUNK_FWD_O_ARCH35_VECTOR_H
#define CHUNK_FWD_O_ARCH35_VECTOR_H

#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"
#include "kernel_utils/vector/regbase.hpp"
#include "../chunk_fwd_o_struct.h"
#include "chunk_fwd_o_common.h"

namespace GDN {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr float CHUNK_FWD_O_LN2 = 0.69314718055994530941723212145818f;
constexpr uint32_t CHUNK_FWD_O_UB_ALIGN = 32;
constexpr uint32_t CHUNK_FWD_O_VEC_MASK_OFFSET = 0;
constexpr uint32_t CHUNK_FWD_O_VEC_GATE_O_OFFSET = CHUNK_FWD_O_UB_MASK_BYTES;
constexpr uint32_t CHUNK_FWD_O_VEC_GATE_A_OFFSET =
    CHUNK_FWD_O_VEC_GATE_O_OFFSET + CHUNK_FWD_O_A5_BT * sizeof(float);
constexpr uint32_t CHUNK_FWD_O_VEC_G_FP32_OFFSET =
    CHUNK_FWD_O_VEC_GATE_A_OFFSET + CHUNK_FWD_O_A5_BT * CHUNK_FWD_O_A5_BT * sizeof(float);
constexpr uint32_t CHUNK_FWD_O_VEC_G_INPUT_OFFSET =
    CHUNK_FWD_O_VEC_G_FP32_OFFSET + CHUNK_FWD_O_A5_BT * sizeof(float);

__simd_callee__ inline void LoadGateFloatPair(RegTensor<float> &zero, RegTensor<float> &one, __ubuf__ float *src)
{
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zero, one, src);
}

__simd_callee__ inline void StoreGateFloatPair(__ubuf__ float *dst, RegTensor<float> &zero, RegTensor<float> &one,
                                              MaskReg &maskF32)
{
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst, zero, one, maskF32);
}

// P1 · causal mask m_A[C,C] in uint8, lower-triangular with i >= j.
__aicore__ inline void BuildCausalMaskU8(LocalTensor<uint8_t> &mask, uint32_t chunkLen)
{
    const uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
    for (uint32_t row = 0; row < bt; ++row) {
        for (uint32_t col = 0; col < bt; ++col) {
            const uint8_t valid = (row < chunkLen && col < chunkLen && row >= col) ? 1 : 0;
            mask.SetValue(row * bt + col, valid);
        }
    }
}

// P2 · gate_o[C] = exp_fn(g), gate_A[C,C] = exp_fn(g[i]-g[j]).
template <bool UseExp2>
__simd_vf__ inline void Stage1Gate64VF(__ubuf__ float *gateOAddr, __ubuf__ float *gateAAddr, __ubuf__ float *gAddr,
                                       uint16_t chunkLen)
{
    (void)chunkLen;
    constexpr uint16_t kBt = static_cast<uint16_t>(CHUNK_FWD_O_A5_BT);
    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> gZeroReg;
    RegTensor<float> gOneReg;
    RegTensor<float> gateZeroReg;
    RegTensor<float> gateOneReg;
    RegTensor<float> gRowReg;
    RegTensor<float> rowGateZeroReg;
    RegTensor<float> rowGateOneReg;

    LoadGateFloatPair(gZeroReg, gOneReg, gAddr);

    if constexpr (UseExp2) {
        Muls(gateZeroReg, gZeroReg, CHUNK_FWD_O_LN2, maskFull32);
        Muls(gateOneReg, gOneReg, CHUNK_FWD_O_LN2, maskFull32);
    } else {
        Adds(gateZeroReg, gZeroReg, 0.0f, maskFull32);
        Adds(gateOneReg, gOneReg, 0.0f, maskFull32);
    }
    ExpFloatTwoReg(gateZeroReg, gateOneReg, gateZeroReg, gateOneReg, maskFull32);
    StoreGateFloatPair(gateOAddr, gateZeroReg, gateOneReg, maskFull32);

    for (uint16_t row = 0; row < kBt; ++row) {
        LoadIn<float, true>(gRowReg, gAddr + row);
        SubFloatTwoReg(rowGateZeroReg, rowGateOneReg, gRowReg, gRowReg, gZeroReg, gOneReg, maskFull32);
        if constexpr (UseExp2) {
            Muls(rowGateZeroReg, rowGateZeroReg, CHUNK_FWD_O_LN2, maskFull32);
            Muls(rowGateOneReg, rowGateOneReg, CHUNK_FWD_O_LN2, maskFull32);
        }
        ExpFloatTwoReg(rowGateZeroReg, rowGateOneReg, rowGateZeroReg, rowGateOneReg, maskFull32);
        StoreGateFloatPair(gateAAddr + static_cast<uint32_t>(row) * kBt, rowGateZeroReg, rowGateOneReg, maskFull32);
    }
}

// P4 · Stage3 gate application + causal mask (stub until P4).
__simd_vf__ inline void Stage3GateMask(__ubuf__ float *aRawAddr, __ubuf__ float *oSRawAddr,
                                       __ubuf__ float *gateOAddr, __ubuf__ float *gateAAddr,
                                       __ubuf__ uint8_t *maskAddr, __ubuf__ float *oSPrimeAddr, int64_t chunkLen)
{
    (void)aRawAddr;
    (void)oSRawAddr;
    (void)gateOAddr;
    (void)gateAAddr;
    (void)maskAddr;
    (void)oSPrimeAddr;
    (void)chunkLen;
}

// P6 · Stage5 fuse (stub until P6).
__simd_vf__ inline void Stage5Fuse(__ubuf__ float *oSPrimeAddr, __ubuf__ float *oLAddr, __ubuf__ bfloat16_t *oOutAddr,
                                   float scale, int64_t chunkLen)
{
    (void)oSPrimeAddr;
    (void)oLAddr;
    (void)oOutAddr;
    (void)scale;
    (void)chunkLen;
}

template <typename GT, bool UseExp2>
class ChunkFwdOA5VectorProcess {
public:
    __aicore__ inline ChunkFwdOA5VectorProcess(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                               GM_ADDR cuSeqlens, GM_ADDR chunkOffsets, GM_ADDR o, GM_ADDR workspace)
        : q_(q), k_(k), v_(v), h_(h), g_(g), cuSeqlens_(cuSeqlens), chunkOffsets_(chunkOffsets), o_(o),
          workspace_(workspace)
    {
    }

    __aicore__ inline void Init(const ChunkFwdOTilingData &tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;

        const uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
        pipe_->InitBuffer(
            ubBuf_, CHUNK_FWD_O_UB_OSRAW_OFFSET + CHUNK_FWD_O_UB_OSRAW_BYTES);
        mte2ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
    }

    __aicore__ inline void ProcessInit()
    {
        if (AscendC::GetSubBlockIdx() == 0) {
            ChunkFwdOWriteDebugHeader(workspace_, tiling_);
        }
    }

    __aicore__ inline void PrepareCubeUbForStage2()
    {
        SetCubeUbFreeAiv();
    }

    __aicore__ inline void WaitStage2Ready()
    {
        WaitCubeUbReadyAiv();
    }

    __aicore__ inline void DumpStage2Slice(uint32_t loopIdx, int64_t hv)
    {
        if (!ChunkFwdODumpEnabled(tiling_)) {
            return;
        }
        constexpr uint32_t rowsPerSubBlock = CHUNK_FWD_O_A5_BT / 2;
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const int64_t slotIdx = ChunkFwdODumpSlotIndex(tiling_, loopIdx, hv);
        GM_ADDR slotBase = ChunkFwdODumpSlotPtr(workspace_, tiling_, slotIdx);

        LocalTensor<bfloat16_t> aRaw =
            resource_.ubBuf.GetBufferByByte<bfloat16_t>(CHUNK_FWD_O_UB_ARAW_OFFSET);
        LocalTensor<bfloat16_t> oSRaw =
            resource_.ubBuf.GetBufferByByte<bfloat16_t>(CHUNK_FWD_O_UB_OSRAW_OFFSET);
        const uint32_t aRowBytes = CHUNK_FWD_O_A5_BT * sizeof(bfloat16_t);
        const uint32_t oRowBytes = CHUNK_FWD_O_A5_V * sizeof(bfloat16_t);
        ChunkFwdODumpUbToGm(
            slotBase, CHUNK_FWD_O_DBG_ARAW_OFF + subBlockIdx * rowsPerSubBlock * aRowBytes,
            aRaw, rowsPerSubBlock * CHUNK_FWD_O_A5_BT);
        ChunkFwdODumpUbToGm(
            slotBase, CHUNK_FWD_O_DBG_OSRAW_OFF + subBlockIdx * rowsPerSubBlock * oRowBytes,
            oSRaw, rowsPerSubBlock * CHUNK_FWD_O_A5_V);
    }

    __aicore__ inline void ProcessStage1(uint32_t loopIdx, const ChunkFwdOChunkLoc &loc, int64_t hk, int64_t hv)
    {
        (void)loopIdx;
        (void)hk;

        GlobalTensor<GT> gGm;
        gGm.SetGlobalBuffer((__gm__ GT *)g_);
        const int64_t gOffset = ChunkFwdOGOffset(tiling_, loc, hv);

        const uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
        LocalTensor<GT> gLocal =
            ubBuf_.GetWithOffset<GT>(bt, CHUNK_FWD_O_VEC_G_INPUT_OFFSET);
        DataCopyPad(gLocal, gGm[gOffset],
                    {1, static_cast<uint32_t>(loc.chunkLen * sizeof(GT)), 0, 0, 0},
                    {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        LocalTensor<float> gFp32 =
            ubBuf_.GetWithOffset<float>(bt, CHUNK_FWD_O_VEC_G_FP32_OFFSET);
        if constexpr (std::is_same<GT, float>::value) {
            DataCopy(gFp32, gLocal, loc.chunkLen);
        } else {
            Cast(gFp32, gLocal, RoundMode::CAST_NONE, loc.chunkLen);
        }
        if (loc.chunkLen < static_cast<uint32_t>(CHUNK_FWD_O_A5_BT)) {
            Duplicate(gFp32[loc.chunkLen], static_cast<float>(0), static_cast<int32_t>(CHUNK_FWD_O_A5_BT - loc.chunkLen));
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> mask =
            ubBuf_.GetWithOffset<uint8_t>(bt * bt, CHUNK_FWD_O_VEC_MASK_OFFSET);
        BuildCausalMaskU8(mask, loc.chunkLen);
        LocalTensor<float> gateO =
            ubBuf_.GetWithOffset<float>(bt, CHUNK_FWD_O_VEC_GATE_O_OFFSET);
        LocalTensor<float> gateA =
            ubBuf_.GetWithOffset<float>(bt * bt, CHUNK_FWD_O_VEC_GATE_A_OFFSET);
        AscendC::VF_CALL<Stage1Gate64VF<UseExp2>>(
            (__ubuf__ float *)gateO.GetPhyAddr(), (__ubuf__ float *)gateA.GetPhyAddr(),
            (__ubuf__ float *)gFp32.GetPhyAddr(), static_cast<uint16_t>(loc.chunkLen));
        PipeBarrier<PIPE_V>();
        DumpStage1(loc, loopIdx, hv, mask, gateO, gateA);
    }

    __aicore__ inline void ProcessStage3(uint32_t loopIdx, const ChunkFwdOChunkLoc &loc, int64_t hk, int64_t hv)
    {
        (void)loopIdx;
        (void)loc;
        (void)hk;
        (void)hv;
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        // Serial scheduling: S2 completes before this stage via SyncAll.
        WaitCubeUbReadyAiv();
        SetCubeUbFreeAiv();
    }

    __aicore__ inline void ProcessStage5(uint32_t loopIdx, const ChunkFwdOChunkLoc &loc, int64_t hk, int64_t hv)
    {
        (void)loopIdx;
        (void)loc;
        (void)hk;
        (void)hv;
    }

private:
    __aicore__ inline void DumpStage1(const ChunkFwdOChunkLoc &loc, uint32_t loopIdx, int64_t hv,
                                      LocalTensor<uint8_t> &mask, LocalTensor<float> &gateO,
                                      LocalTensor<float> &gateA)
    {
        (void)loc;
        if ASCEND_IS_AIV {
            if (ChunkFwdODumpEnabled(tiling_) && AscendC::GetSubBlockIdx() == 0) {
                SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
                WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
                const int64_t slotIdx = ChunkFwdODumpSlotIndex(tiling_, loopIdx, hv);
                GM_ADDR slotBase = ChunkFwdODumpSlotPtr(workspace_, tiling_, slotIdx);
                const uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
                ChunkFwdODumpUbToGm(slotBase, CHUNK_FWD_O_DBG_MASK_OFF, mask, bt * bt);
                ChunkFwdODumpUbToGm(slotBase, CHUNK_FWD_O_DBG_GATE_O_OFF, gateO, bt);
                ChunkFwdODumpUbToGm(slotBase, CHUNK_FWD_O_DBG_GATE_A_OFF, gateA, bt * bt);
            }
        }
    }

    __aicore__ inline void SetCubeUbFreeAiv()
    {
        CrossCoreSetFlag<0x4, PIPE_V>(CHUNK_FWD_O_CUBE_UB_FREE_FLAG);
    }

    __aicore__ inline void WaitCubeUbReadyAiv()
    {
        CrossCoreWaitFlag<0x4, PIPE_V>(CHUNK_FWD_O_CUBE_UB_READY_FLAG);
    }

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
    TPipe *pipe_ = nullptr;

    TBuf<TPosition::VECCALC> ubBuf_;
    TEventID mte2ToVEvent_ = 0;
    TEventID vToMte3Event_ = 0;
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource_;
};

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_VECTOR_H
