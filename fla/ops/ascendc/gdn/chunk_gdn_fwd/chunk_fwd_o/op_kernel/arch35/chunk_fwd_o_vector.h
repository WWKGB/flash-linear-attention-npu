/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef CHUNK_FWD_O_ARCH35_VECTOR_H
#define CHUNK_FWD_O_ARCH35_VECTOR_H

#include "kernel_operator.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "kernel_utils/vector/regbase.hpp"
#include "../chunk_fwd_o_struct.h"
#include "chunk_fwd_o_common.h"

namespace GDN {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr float CHUNK_FWD_O_LN2 = 0.69314718055994530941723212145818f;
constexpr CastTrait CHUNK_FWD_O_FP32_TO_B16_PACK = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_ROUND,
};

__simd_callee__ inline void LoadGateFloatPair(RegTensor<float> &zero, RegTensor<float> &one, __ubuf__ float *src)
{
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zero, one, src);
}

__simd_callee__ inline void StoreGateFloatPair(__ubuf__ float *dst, RegTensor<float> &zero, RegTensor<float> &one,
                                              MaskReg &maskF32)
{
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst, zero, one, maskF32);
}

__simd_vf__ inline void PadGateInput64VF(__ubuf__ float *gAddr, uint16_t validRows)
{
    RegTensor<float> gReg;
    RegTensor<float> zeroReg;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t validCount = static_cast<uint32_t>(validRows);
    MaskReg validMask = UpdateMask<float>(validCount);
    LoadAlign(gReg, gAddr);
    Duplicate(zeroReg, 0.0f, fullMask);
    Select(gReg, gReg, zeroReg, validMask);
    StoreAlign(gAddr, gReg, fullMask);
}

template <bool UseExp2>
__simd_vf__ inline void Stage1Gate64VF(__ubuf__ float *gateAAddr, __ubuf__ float *gAddr, uint16_t chunkLen)
{
    (void)chunkLen;
    constexpr uint16_t kBt = static_cast<uint16_t>(CHUNK_FWD_O_A5_BT);
    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> gZeroReg;
    RegTensor<float> gOneReg;
    RegTensor<float> gRowReg;
    RegTensor<float> rowGateZeroReg;
    RegTensor<float> rowGateOneReg;

    LoadGateFloatPair(gZeroReg, gOneReg, gAddr);

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

__simd_vf__ inline void Stage3Gate64VF(__ubuf__ bfloat16_t *aPrimeAddr, __ubuf__ float *oSPrimeAddr,
                                      __ubuf__ float *aRawAddr, __ubuf__ float *oSRawAddr,
                                      __ubuf__ float *gateAAddr, __ubuf__ float *gateOAddr,
                                      uint16_t validRows)
{
    constexpr uint16_t kBt = static_cast<uint16_t>(CHUNK_FWD_O_A5_BT);
    constexpr uint16_t kTilesPerRow = static_cast<uint16_t>(CHUNK_FWD_O_A5_V / CHUNK_FWD_O_A5_BT);
    RegTensor<float> aRawReg;
    RegTensor<float> gateAReg;
    RegTensor<float> aPrimeReg;
    RegTensor<bfloat16_t> aPrimeBf16Reg;
    RegTensor<float> oSRawReg;
    RegTensor<float> gateOVectorReg;
    RegTensor<float> gateORowReg;
    RegTensor<float> zeroReg;
    RegTensor<uint32_t> rowIndexReg;
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg lowerMask;
    uint32_t lowerCount = 0;
    Duplicate(zeroReg, 0.0f, floatMask);
    LoadAlign(gateOVectorReg, gateOAddr);

    for (uint16_t row = 0; row < kBt; ++row) {
        const uint32_t matrixOffset = static_cast<uint32_t>(row) * kBt;
        lowerCount = row < validRows ? static_cast<uint32_t>(row + 1) : 0U;
        lowerMask = UpdateMask<float>(lowerCount);
        LoadAlign(aRawReg, aRawAddr + matrixOffset);
        LoadAlign(gateAReg, gateAAddr + matrixOffset);
        Mul(aPrimeReg, aRawReg, gateAReg, floatMask);
        Select(aPrimeReg, aPrimeReg, zeroReg, lowerMask);
        Cast<bfloat16_t, float, CHUNK_FWD_O_FP32_TO_B16_PACK>(aPrimeBf16Reg, aPrimeReg, floatMask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(
            aPrimeAddr + matrixOffset, aPrimeBf16Reg, floatMask);

        Duplicate(rowIndexReg, static_cast<uint32_t>(row), floatMask);
        Gather(gateORowReg, gateOVectorReg, rowIndexReg);
        for (uint16_t tile = 0; tile < kTilesPerRow; ++tile) {
            const uint32_t offset =
                static_cast<uint32_t>(row) * static_cast<uint16_t>(CHUNK_FWD_O_A5_V) +
                static_cast<uint32_t>(tile) * kBt;
            if (row < validRows) {
                LoadAlign(oSRawReg, oSRawAddr + offset);
                Mul(oSRawReg, oSRawReg, gateORowReg, floatMask);
                StoreAlign(oSPrimeAddr + offset, oSRawReg, floatMask);
            } else {
                StoreAlign(oSPrimeAddr + offset, zeroReg, floatMask);
            }
        }
    }
}

__simd_vf__ inline void Stage5Fuse64VF(__ubuf__ bfloat16_t *oOutAddr, __ubuf__ float *oSPrimeAddr,
                                       __ubuf__ float *oLAddr, float scale, uint16_t validRows)
{
    constexpr uint16_t kV = static_cast<uint16_t>(CHUNK_FWD_O_A5_V);
    constexpr uint16_t kTilesPerRow = static_cast<uint16_t>(CHUNK_FWD_O_A5_V / CHUNK_FWD_O_A5_BT);
    RegTensor<float> oSPrimeReg;
    RegTensor<float> oLReg;
    RegTensor<float> oOutReg;
    RegTensor<bfloat16_t> oOutBf16Reg;
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t row = 0; row < validRows; ++row) {
        for (uint16_t tile = 0; tile < kTilesPerRow; ++tile) {
            const uint32_t offset = static_cast<uint32_t>(row) * kV +
                                    static_cast<uint32_t>(tile) * CHUNK_FWD_O_A5_BT;
            LoadAlign(oSPrimeReg, oSPrimeAddr + offset);
            LoadAlign(oLReg, oLAddr + offset);
            Add(oOutReg, oSPrimeReg, oLReg, floatMask);
            Muls(oOutReg, oOutReg, scale, floatMask);
            Cast<bfloat16_t, float, CHUNK_FWD_O_FP32_TO_B16_PACK>(oOutBf16Reg, oOutReg, floatMask);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(oOutAddr + offset, oOutBf16Reg, floatMask);
        }
    }
}

template <typename GT, bool UseExp2>
class ChunkFwdOA5VectorProcess {
public:
    using ArchTag = Catlass::Arch::Ascend950;

    static constexpr bool kEnableStage3Compute = true;
    static constexpr uint32_t kStreamBankCount = CHUNK_FWD_O_STREAM_BANK_COUNT;
    static constexpr uint32_t kLocalSlotCount = 2U;

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
        oGm_.SetGlobalBuffer((__gm__ bfloat16_t *)o_);

        pipe_->InitBuffer(ubBuf_, CHUNK_FWD_O_UB_TOTAL_BYTES);
        for (uint32_t bankIdx = 0; bankIdx < kStreamBankCount; ++bankIdx) {
            mte2ToV_[bankIdx] = pipe_->AllocEventID<HardEvent::MTE2_V>();
            vToMte3Stream_[bankIdx] = pipe_->AllocEventID<HardEvent::V_MTE3>();
            mte3ToVStream_[bankIdx] = pipe_->AllocEventID<HardEvent::MTE3_V>();
            mte3ToMte2_[bankIdx] = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
            // No previous MTE3 owns either Stage3 output slot on the first group.
            SetFlag<HardEvent::MTE3_V>(mte3ToVStream_[bankIdx]);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[bankIdx]);
        }
        for (uint32_t slotIdx = 0; slotIdx < kLocalSlotCount; ++slotIdx) {
            gateReadyEvent_[slotIdx] = pipe_->AllocEventID<HardEvent::V_MTE2>();
        }
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
    }

    __aicore__ inline void WaitStage2Ready()
    {
        Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
    }

    __aicore__ inline void SignalStage3Ready()
    {
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
    }

    __aicore__ inline void WaitStage4Ready()
    {
        Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
    }

    __aicore__ inline void ReleaseStage2Group()
    {
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
    }

    __aicore__ inline void SignalGateReady(uint32_t localSlot)
    {
        SetFlag<HardEvent::V_MTE2>(gateReadyEvent_[localSlot]);
    }

    __aicore__ inline void WaitGateReady(uint32_t localSlot)
    {
        WaitFlag<HardEvent::V_MTE2>(gateReadyEvent_[localSlot]);
    }

    __aicore__ inline void LoadStage1G(const ChunkFwdOChunkLoc &loc, int64_t hv, uint32_t streamSlot)
    {
        GlobalTensor<GT> gGm;
        gGm.SetGlobalBuffer((__gm__ GT *)g_);
        const int64_t gOffset = ChunkFwdOGOffset(tiling_, loc, hv);
        const uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
        const uint32_t gScratchOff = ChunkFwdOGScratchOffset(streamSlot);

        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[streamSlot]);
        if constexpr (std::is_same<GT, float>::value) {
            LocalTensor<float> gFp32 = ubBuf_.GetWithOffset<float>(bt, gScratchOff);
            DataCopyPad(gFp32, gGm[gOffset],
                        {1, static_cast<uint32_t>(loc.chunkLen * sizeof(GT)), 0, 0, 0},
                        {false, 0, 0, 0});
        } else {
            LocalTensor<GT> gLocal = ubBuf_.GetWithOffset<GT>(bt, gScratchOff);
            DataCopyPad(gLocal, gGm[gOffset],
                        {1, static_cast<uint32_t>(loc.chunkLen * sizeof(GT)), 0, 0, 0},
                        {false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToV_[streamSlot]);
    }

    __aicore__ inline void ComputeStage1Gate(const ChunkFwdOChunkLoc &loc, uint32_t localSlot, uint32_t streamSlot)
    {
        const uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
        const uint32_t gScratchOff = ChunkFwdOGScratchOffset(streamSlot);

        WaitFlag<HardEvent::MTE2_V>(mte2ToV_[streamSlot]);

        LocalTensor<float> gFp32 = ubBuf_.GetWithOffset<float>(bt, gScratchOff);
        if constexpr (!std::is_same<GT, float>::value) {
            LocalTensor<GT> gLocal = ubBuf_.GetWithOffset<GT>(bt, gScratchOff);
            Cast(gFp32, gLocal, RoundMode::CAST_NONE, loc.chunkLen);
        }
        if (loc.chunkLen < static_cast<uint32_t>(CHUNK_FWD_O_A5_BT)) {
            AscendC::VF_CALL<PadGateInput64VF>(
                reinterpret_cast<__ubuf__ float *>(gFp32.GetPhyAddr()),
                static_cast<uint16_t>(loc.chunkLen));
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<float> gateO = ubBuf_.GetWithOffset<float>(bt, ChunkFwdOGateOOffset(localSlot));
        LocalTensor<float> gateA =
            ubBuf_.GetWithOffset<float>(bt * bt, ChunkFwdOGateAOffset(localSlot));
        if constexpr (UseExp2) {
            Muls(gateO, gFp32, CHUNK_FWD_O_LN2, bt);
            PipeBarrier<PIPE_V>();
            Exp(gateO, gateO, bt);
        } else {
            Exp(gateO, gFp32, bt);
        }
        PipeBarrier<PIPE_V>();
        AscendC::VF_CALL<Stage1Gate64VF<UseExp2>>(
            (__ubuf__ float *)gateA.GetPhyAddr(), (__ubuf__ float *)gFp32.GetPhyAddr(),
            static_cast<uint16_t>(loc.chunkLen));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BeginStage1Group()
    {
        streamSlot_ = 0U;
    }

    __aicore__ inline void ProcessStage1Head(const ChunkFwdOChunkLoc &loc, int64_t hvBase, int64_t headOffset)
    {
        const uint32_t localSlot = static_cast<uint32_t>(headOffset / 2);
        LoadStage1G(loc, hvBase + headOffset, streamSlot_);
        ComputeStage1Gate(loc, localSlot, streamSlot_);
        SignalGateReady(localSlot);
        streamSlot_ ^= 1U;
    }

    __aicore__ inline void ProcessStage1Group(const ChunkFwdOChunkLoc &loc, int64_t hvBase, int64_t taskCount,
                                              uint32_t subBlockIdx)
    {
        BeginStage1Group();
        for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
            if (static_cast<uint32_t>(headOffset % 2) == subBlockIdx) {
                ProcessStage1Head(loc, hvBase, headOffset);
            }
        }
    }

    __aicore__ inline void BeginStage3Group()
    {
        stage3StreamSlot_ = 0U;
        stage3ActiveMask_ = 0U;
    }

    __aicore__ inline void FinishStage3Group()
    {
        for (uint32_t streamSlot = 0; streamSlot < kStreamBankCount; ++streamSlot) {
            if ((stage3ActiveMask_ & (1U << streamSlot)) == 0U) {
                continue;
            }
            WaitFlag<HardEvent::MTE3_V>(mte3ToVStream_[streamSlot]);
            // Rearm the first-use permission consumed by the next task group.
            SetFlag<HardEvent::MTE3_V>(mte3ToVStream_[streamSlot]);
        }
        stage3ActiveMask_ = 0U;
    }

    __aicore__ inline void ProcessStage3(const ChunkFwdOChunkLoc &loc, uint32_t localSlot, uint32_t headOffset)
    {
        if constexpr (!kEnableStage3Compute) {
            return;
        }

        constexpr uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
        constexpr uint32_t vDim = static_cast<uint32_t>(CHUNK_FWD_O_A5_V);
        const uint32_t matrixElems = bt * bt;
        const uint32_t streamSlot = stage3StreamSlot_;

        WaitGateReady(localSlot);
        // The previous MTE3 reader of this ping/pong slot must finish before V
        // overwrites it. The other slot remains available to the next owner HV.
        WaitFlag<HardEvent::MTE3_V>(mte3ToVStream_[streamSlot]);

        LocalTensor<float> gateO = ubBuf_.GetWithOffset<float>(bt, ChunkFwdOGateOOffset(localSlot));
        LocalTensor<float> gateA = ubBuf_.GetWithOffset<float>(matrixElems, ChunkFwdOGateAOffset(localSlot));
        LocalTensor<float> aRaw = ubBuf_.GetWithOffset<float>(matrixElems, ChunkFwdOARawOffset(localSlot));
        LocalTensor<float> oSRaw = ubBuf_.GetWithOffset<float>(bt * vDim, ChunkFwdOOSRawOffset(localSlot));
        LocalTensor<float> oSPrime = ubBuf_.GetWithOffset<float>(bt * vDim, ChunkFwdOOsPrimeOffset(localSlot));
        LocalTensor<bfloat16_t> aPrimeBf16 =
            ubBuf_.GetWithOffset<bfloat16_t>(matrixElems, ChunkFwdOAPrimeBf16Offset(streamSlot));

        AscendC::VF_CALL<Stage3Gate64VF>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(aPrimeBf16.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(oSPrime.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(aRaw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(oSRaw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gateA.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gateO.GetPhyAddr()),
            static_cast<uint16_t>(loc.chunkLen));
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(vToMte3Stream_[streamSlot]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Stream_[streamSlot]);

        const uint32_t aivCoreIdx = AscendC::GetBlockIdx() / 2U;
        GM_ADDR aPrimeAddr = ChunkFwdOAPrimeGmOffset(workspace_, tiling_, aivCoreIdx, headOffset);
        GlobalTensor<bfloat16_t> aPrimeGm;
        aPrimeGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(aPrimeAddr));
        DataCopyExtParams aPrimeCopyParams{1, CHUNK_FWD_O_APRIME_SLOT_BYTES, 0, 0, 0};
        DataCopyPad(aPrimeGm, aPrimeBf16, aPrimeCopyParams);

        // Close the complete slot lifecycle only after all current MTE3 reads
        // have been issued. The next-group MTE2 and next V reuse wait on their
        // respective reverse dependencies while the other slot keeps running.
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[streamSlot]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVStream_[streamSlot]);
        stage3ActiveMask_ |= 1U << streamSlot;
        stage3StreamSlot_ ^= 1U;
    }

    __aicore__ inline void ProcessStage5(const ChunkFwdOChunkLoc &loc, int64_t hv, uint32_t localSlot)
    {
        constexpr uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
        constexpr uint32_t vDim = static_cast<uint32_t>(CHUNK_FWD_O_A5_V);
        LocalTensor<float> oSPrime =
            ubBuf_.GetWithOffset<float>(bt * vDim, ChunkFwdOOsPrimeOffset(localSlot));
        LocalTensor<float> oL = ubBuf_.GetWithOffset<float>(bt * vDim, ChunkFwdOOlOffset(localSlot));
        LocalTensor<bfloat16_t> oOut =
            ubBuf_.GetWithOffset<bfloat16_t>(bt * vDim, ChunkFwdOOlOffset(localSlot));

        AscendC::VF_CALL<Stage5Fuse64VF>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(oOut.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(oSPrime.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(oL.GetPhyAddr()), tiling_.scale,
            static_cast<uint16_t>(loc.chunkLen));
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        const int64_t oOffset = ChunkFwdOOOffset(tiling_, loc, hv);
        DataCopyExtParams outputCopyParams{
            static_cast<uint16_t>(loc.chunkLen),
            static_cast<uint32_t>(vDim * sizeof(bfloat16_t)),
            0,
            static_cast<uint32_t>((tiling_.vNumHead - 1) * vDim * sizeof(bfloat16_t)),
            0};
        DataCopyPad(oGm_[oOffset], oOut, outputCopyParams);
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
    TPipe *pipe_ = nullptr;

    TBuf<TPosition::VECCALC> ubBuf_;
    GlobalTensor<bfloat16_t> oGm_;
    uint32_t streamSlot_ = 0;
    uint32_t stage3StreamSlot_ = 0;
    uint32_t stage3ActiveMask_ = 0;
    TEventID mte2ToV_[kStreamBankCount];
    TEventID vToMte3Stream_[kStreamBankCount];
    TEventID mte3ToVStream_[kStreamBankCount];
    TEventID mte3ToMte2_[kStreamBankCount];
    TEventID gateReadyEvent_[kLocalSlotCount];
    TEventID vToMte3Event_ = 0;
    Catlass::Arch::CrossCoreFlag vecToCubeFlag_{CHUNK_FWD_O_VEC_TO_CUBE_READY_FLAG};
    Catlass::Arch::CrossCoreFlag cubeToVecFlag_{CHUNK_FWD_O_CUBE_TO_VEC_READY_FLAG};
};

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_VECTOR_H
