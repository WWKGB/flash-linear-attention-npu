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

__simd_callee__ inline void LoadGateFloatPair(RegTensor<float> &zero, RegTensor<float> &one, __ubuf__ float *src)
{
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zero, one, src);
}

__simd_callee__ inline void StoreGateFloatPair(__ubuf__ float *dst, RegTensor<float> &zero, RegTensor<float> &one,
                                              MaskReg &maskF32)
{
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst, zero, one, maskF32);
}

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

template <typename GT, bool UseExp2>
class ChunkFwdOA5VectorProcess {
public:
    using ArchTag = Catlass::Arch::Ascend950;

    static constexpr bool kEnableStage3Compute = false;
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

        pipe_->InitBuffer(ubBuf_, CHUNK_FWD_O_UB_TOTAL_BYTES);
        for (uint32_t bankIdx = 0; bankIdx < kStreamBankCount; ++bankIdx) {
            mte2ToV_[bankIdx] = pipe_->AllocEventID<HardEvent::MTE2_V>();
            vToMte3Stream_[bankIdx] = pipe_->AllocEventID<HardEvent::V_MTE3>();
            mte3ToMte2_[bankIdx] = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[bankIdx]);
        }
        for (uint32_t slotIdx = 0; slotIdx < kLocalSlotCount; ++slotIdx) {
            gateReadyEvent_[slotIdx] = pipe_->AllocEventID<HardEvent::V_MTE2>();
        }
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
    }

    __aicore__ inline void ProcessInit()
    {
        if (AscendC::GetSubBlockIdx() == 0) {
            ChunkFwdOWriteDebugHeader(workspace_, tiling_);
        }
    }

    __aicore__ inline void WaitStage2Ready(uint32_t headOffset)
    {
        Catlass::Arch::CrossCoreFlag flag{
            static_cast<Catlass::Arch::FlagID>(CHUNK_FWD_O_CC_CUBE_READY_BASE + headOffset)};
        Catlass::Arch::CrossCoreWaitFlag(flag);
    }

    __aicore__ inline void SignalStage1Ready(uint32_t headOffset)
    {
        Catlass::Arch::CrossCoreFlag flag{
            static_cast<Catlass::Arch::FlagID>(CHUNK_FWD_O_S1_READY_BASE + headOffset)};
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag);
    }

    __aicore__ inline void ReleaseStage2Group()
    {
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(groupReleaseFlag_);
    }

    __aicore__ inline void SignalStage2Consumed(uint32_t headOffset)
    {
        Catlass::Arch::CrossCoreFlag flag{
            static_cast<Catlass::Arch::FlagID>(CHUNK_FWD_O_CC_SLOT_RELEASE_BASE + headOffset)};
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag);
    }

    __aicore__ inline void SignalStage1GroupDone()
    {
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(stage1GroupDoneFlag_);
    }

    __aicore__ inline void SignalGateReady(uint32_t localSlot)
    {
        SetFlag<HardEvent::V_MTE2>(gateReadyEvent_[localSlot]);
    }

    __aicore__ inline void WaitGateReady(uint32_t localSlot)
    {
        WaitFlag<HardEvent::V_MTE2>(gateReadyEvent_[localSlot]);
    }

    __aicore__ inline int64_t FindFirstOwnerHeadOffset(int64_t taskCount, uint32_t subBlockIdx) const
    {
        for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
            if (static_cast<uint32_t>(headOffset % 2) == subBlockIdx) {
                return headOffset;
            }
        }
        return -1;
    }

    __aicore__ inline int64_t FindNextOwnerHeadOffset(int64_t headOffset, int64_t taskCount,
                                                      uint32_t subBlockIdx) const
    {
        for (int64_t nextHead = headOffset + 1; nextHead < taskCount; ++nextHead) {
            if (static_cast<uint32_t>(nextHead % 2) == subBlockIdx) {
                return nextHead;
            }
        }
        return -1;
    }

    __aicore__ inline void PrefetchStage1G(const ChunkFwdOChunkLoc &loc, int64_t hv, uint32_t streamSlot)
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
            Duplicate(gFp32[loc.chunkLen], static_cast<float>(0),
                      static_cast<int32_t>(CHUNK_FWD_O_A5_BT - loc.chunkLen));
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<float> gateO = ubBuf_.GetWithOffset<float>(bt, ChunkFwdOGateOOffset(localSlot));
        LocalTensor<float> gateA =
            ubBuf_.GetWithOffset<float>(bt * bt, ChunkFwdOGateAOffset(localSlot));
        AscendC::VF_CALL<Stage1Gate64VF<UseExp2>>(
            (__ubuf__ float *)gateO.GetPhyAddr(), (__ubuf__ float *)gateA.GetPhyAddr(),
            (__ubuf__ float *)gFp32.GetPhyAddr(), static_cast<uint16_t>(loc.chunkLen));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[streamSlot]);
    }

    __aicore__ inline void ProcessStage1Group(uint32_t loopIdx, const ChunkFwdOChunkLoc &loc, int64_t hvBase,
                                              int64_t taskCount, uint32_t subBlockIdx)
    {
        streamSlot_ = 0U;
        const int64_t firstOwner = FindFirstOwnerHeadOffset(taskCount, subBlockIdx);
        if (firstOwner >= 0) {
            PrefetchStage1G(loc, hvBase + firstOwner, streamSlot_);
        }

        // Both subblocks advance the same headOffset in lock-step (59e83dc /
        // PR404 §Stage0): only the owner computes; mode=0x2 SignalStage1Ready
        // runs at iteration end so the two AIV subblocks stay paired per head.
        for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
            if (static_cast<uint32_t>(headOffset % 2) == subBlockIdx) {
                const uint32_t localSlot = static_cast<uint32_t>(headOffset / 2);
                const int64_t nextOwner = FindNextOwnerHeadOffset(headOffset, taskCount, subBlockIdx);
                if (nextOwner >= 0) {
                    PrefetchStage1G(loc, hvBase + nextOwner, streamSlot_ ^ 1U);
                }
                ComputeStage1Gate(loc, localSlot, streamSlot_);
                SignalGateReady(localSlot);
                streamSlot_ ^= 1U;
            }
            SignalStage1Ready(static_cast<uint32_t>(headOffset));
        }
    }

    __aicore__ inline void DumpStage2ARaw(uint32_t loopIdx, int64_t hv, uint32_t localSlot)
    {
        if (!ChunkFwdODumpEnabled(tiling_)) {
            return;
        }
        const int64_t slotIdx = ChunkFwdODumpSlotIndex(tiling_, loopIdx, hv);
        GM_ADDR slotBase = ChunkFwdODumpSlotPtr(workspace_, tiling_, slotIdx);
        LocalTensor<float> aRaw =
            resource_.ubBuf.template GetBufferByByte<float>(ChunkFwdOARawOffset(localSlot));
        ChunkFwdODumpUbToGm(
            slotBase, CHUNK_FWD_O_DBG_ARAW_OFF, aRaw,
            CHUNK_FWD_O_A5_BT * CHUNK_FWD_O_A5_BT);
    }

    __aicore__ inline void DumpStage2OSRaw(uint32_t loopIdx, int64_t hv, uint32_t localSlot)
    {
        if (!ChunkFwdODumpEnabled(tiling_)) {
            return;
        }
        const int64_t slotIdx = ChunkFwdODumpSlotIndex(tiling_, loopIdx, hv);
        GM_ADDR slotBase = ChunkFwdODumpSlotPtr(workspace_, tiling_, slotIdx);
        const uint32_t vDim = static_cast<uint32_t>(CHUNK_FWD_O_A5_V);
        LocalTensor<float> oSRaw =
            resource_.ubBuf.template GetBufferByByte<float>(ChunkFwdOOSRawOffset(localSlot));
        ChunkFwdODumpUbToGm(
            slotBase, CHUNK_FWD_O_DBG_OSRAW_OFF, oSRaw,
            CHUNK_FWD_O_A5_BT * CHUNK_FWD_O_A5_V);
    }

    __aicore__ inline void DumpStage1Result(uint32_t loopIdx, const ChunkFwdOChunkLoc &loc, int64_t hv,
                                            uint32_t localSlot)
    {
        constexpr uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
        LocalTensor<float> gateO =
            ubBuf_.GetWithOffset<float>(bt, ChunkFwdOGateOOffset(localSlot));
        LocalTensor<float> gateA =
            ubBuf_.GetWithOffset<float>(bt * bt, ChunkFwdOGateAOffset(localSlot));
        DumpStage1(loc, loopIdx, hv, gateO, gateA);
    }

private:
    __aicore__ inline void DumpStage1(const ChunkFwdOChunkLoc &loc, uint32_t loopIdx, int64_t hv,
                                      LocalTensor<float> &gateO, LocalTensor<float> &gateA)
    {
        (void)loc;
        if ASCEND_IS_AIV {
            if (ChunkFwdODumpEnabled(tiling_)) {
                SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
                WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
                const int64_t slotIdx = ChunkFwdODumpSlotIndex(tiling_, loopIdx, hv);
                GM_ADDR slotBase = ChunkFwdODumpSlotPtr(workspace_, tiling_, slotIdx);
                const uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
                ChunkFwdODumpUbToGm(slotBase, CHUNK_FWD_O_DBG_GATE_O_OFF, gateO, bt);
                ChunkFwdODumpUbToGm(slotBase, CHUNK_FWD_O_DBG_GATE_A_OFF, gateA, bt * bt);
            }
        }
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
    Catlass::Arch::Resource<ArchTag> resource_;
    uint32_t streamSlot_ = 0;
    TEventID mte2ToV_[kStreamBankCount];
    TEventID vToMte3Stream_[kStreamBankCount];
    TEventID mte3ToMte2_[kStreamBankCount];
    TEventID gateReadyEvent_[kLocalSlotCount];
    TEventID vToMte3Event_ = 0;
    Catlass::Arch::CrossCoreFlag groupReleaseFlag_{CHUNK_FWD_O_VEC_TO_CUBE_RELEASE_FLAG};
    Catlass::Arch::CrossCoreFlag stage1GroupDoneFlag_{CHUNK_FWD_O_S1_GROUP_DONE_FLAG};
};

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_VECTOR_H
