/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * AIC Stage2: L0/L1 ping-pong + Fixpipe→UB. CrossCore per-head handshake
 * (WaitStage1Ready / WaitStage2Consumed) follows 59e83dc / PR404 cube side.
 */

#ifndef CHUNK_FWD_O_ARCH35_CUBE_H
#define CHUNK_FWD_O_ARCH35_CUBE_H

#define CATLASS_ARCH 3510

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "../chunk_fwd_o_struct.h"
#include "chunk_fwd_o_common.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace GDN {

using namespace AscendC;

class ChunkFwdOA5CubeProcess {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using Element = bfloat16_t;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;

    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutRM, Element, LayoutCM, Element,
                                                              LayoutRM>;
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutRM, Element, LayoutCM, Element,
                                                              LayoutRM>;
    using DirectTileCopyCC = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, Element, LayoutRM, Element, LayoutCM, float, LayoutRM, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using DirectTileCopyRM = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, Element, LayoutRM, Element, LayoutCM, float, LayoutRM>;

    static constexpr uint32_t kBt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
    static constexpr uint32_t kK = static_cast<uint32_t>(CHUNK_FWD_O_A5_K);
    static constexpr uint32_t kV = static_cast<uint32_t>(CHUNK_FWD_O_A5_V);
    static constexpr uint32_t kL0BufferCount = CHUNK_FWD_O_L0_BUFFER_COUNT;
    static constexpr uint32_t kL1StreamBankCount = CHUNK_FWD_O_STREAM_BANK_COUNT;

    static_assert(CHUNK_FWD_O_L0_A_BYTES * kL0BufferCount <= ArchTag::L0A_SIZE,
                  "Stage2 L0A ping/pong exceeds architecture limit.");
    static_assert(CHUNK_FWD_O_L0_B_BYTES * kL0BufferCount <= ArchTag::L0B_SIZE,
                  "Stage2 L0B ping/pong exceeds architecture limit.");
    static_assert(CHUNK_FWD_O_L0_C_BYTES * kL0BufferCount <= ArchTag::L0C_SIZE,
                  "Stage2 L0C ping/pong exceeds architecture limit.");
    static_assert(CHUNK_FWD_O_L1_STREAM_BANK_BYTES * kL1StreamBankCount <= ArchTag::L1_SIZE,
                  "Stage2 L1 stream banks exceed architecture limit.");

    // L1 resident events: slot0 uses 5/6, slot1 uses 7/8.
    static constexpr TEventID kL1Mte1Mte2Base = 5;
    static constexpr TEventID kL1Mte2Mte1Base = 6;

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
        qGm_.SetGlobalBuffer((__gm__ Element *)q_);
        kGm_.SetGlobalBuffer((__gm__ Element *)k_);
        hGm_.SetGlobalBuffer((__gm__ Element *)h_);
        if ASCEND_IS_AIC {
            SetLoadDataPaddingValue<Element>(static_cast<Element>(0));
            for (uint32_t slotIdx = 0; slotIdx < kL1StreamBankCount; ++slotIdx) {
                SetFlag<HardEvent::MTE1_MTE2>(L1Mte1Mte2Event(slotIdx));
            }
            for (uint32_t slotIdx = 0; slotIdx < kL0BufferCount; ++slotIdx) {
                SetFlag<HardEvent::M_MTE1>(L0AEvent(slotIdx));
                SetFlag<HardEvent::M_MTE1>(L0BEvent(slotIdx));
                SetFlag<HardEvent::FIX_M>(slotIdx);
            }
            l0ASlot_ = 0U;
            l0BSlot_ = 0U;
            l0CSlot_ = 0U;
            l1StreamSlot_ = 0U;
        }
    }

    __aicore__ inline void ProcessStage2Group(uint32_t loopIdx, const ChunkFwdOChunkLoc &loc, int64_t hvBase,
                                              int64_t taskCount)
    {
        if ASCEND_IS_AIV {
            return;
        }
        (void)loopIdx;
        if (taskCount <= 0) {
            return;
        }

        l1StreamSlot_ = 0U;
        const int64_t hv0 = hvBase;
        const int64_t hk0 = hv0 / tiling_.hvPerHk;
        PrefetchQKH(loc, hk0, hv0, l1StreamSlot_);

        for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
            if (headOffset >= 2) {
                WaitStage2Consumed(static_cast<uint32_t>(headOffset - 2));
            }
            WaitStage1Ready(static_cast<uint32_t>(headOffset));

            if (headOffset + 1 < taskCount) {
                const int64_t hvNext = hvBase + headOffset + 1;
                const int64_t hkNext = hvNext / tiling_.hvPerHk;
                PrefetchQKH(loc, hkNext, hvNext, l1StreamSlot_ ^ 1U);
            }

            const int64_t hv = hvBase + headOffset;
            const int64_t hk = hv / tiling_.hvPerHk;
            const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % 2);
            const uint32_t localSlot = static_cast<uint32_t>(headOffset / 2);
            ProcessStage2Head(loc, hk, hv, ownerSubBlock, localSlot, static_cast<uint32_t>(headOffset),
                              l1StreamSlot_);
            l1StreamSlot_ ^= 1U;
        }
    }

    __aicore__ inline void WaitStage2GroupRelease()
    {
        Catlass::Arch::CrossCoreWaitFlag(groupReleaseFlag_);
    }

    __aicore__ inline void WaitStage1Ready(uint32_t headOffset)
    {
        Catlass::Arch::CrossCoreFlag flag{
            static_cast<Catlass::Arch::FlagID>(CHUNK_FWD_O_S1_READY_BASE + headOffset)};
        Catlass::Arch::CrossCoreWaitFlag(flag);
    }

    __aicore__ inline void WaitStage2Consumed(uint32_t headOffset)
    {
        Catlass::Arch::CrossCoreFlag flag{
            static_cast<Catlass::Arch::FlagID>(CHUNK_FWD_O_CC_SLOT_RELEASE_BASE + headOffset)};
        Catlass::Arch::CrossCoreWaitFlag(flag);
    }

    __aicore__ inline void ProcessStage4(uint32_t loopIdx, const ChunkFwdOChunkLoc &loc, int64_t hk, int64_t hv)
    {
        (void)loopIdx;
        (void)loc;
        (void)hk;
        (void)hv;
    }

private:
    static constexpr TEventID L0AEvent(uint32_t slot)
    {
        return static_cast<TEventID>(2U * slot);
    }

    static constexpr TEventID L0BEvent(uint32_t slot)
    {
        return static_cast<TEventID>(2U * slot + 1U);
    }

    static constexpr TEventID L1Mte1Mte2Event(uint32_t slot)
    {
        return static_cast<TEventID>(kL1Mte1Mte2Base + slot * 2U);
    }

    static constexpr TEventID L1Mte2Mte1Event(uint32_t slot)
    {
        return static_cast<TEventID>(kL1Mte2Mte1Base + slot * 2U);
    }

    __aicore__ inline void PrefetchQKH(const ChunkFwdOChunkLoc &loc, int64_t hk, int64_t hv, uint32_t l1StreamSlot)
    {
        const uint32_t m = kBt;
        const int64_t qOffset = ChunkFwdOQKOffset(tiling_, loc, hk);
        const int64_t kOffset = ChunkFwdOQKOffset(tiling_, loc, hk);
        const int64_t hOffset = ChunkFwdOHOffset(tiling_, loc, hv);

        WaitFlag<HardEvent::MTE1_MTE2>(L1Mte1Mte2Event(l1StreamSlot));

        using LayoutTagL1Q = typename TileCopyQK::LayoutTagL1A;
        using LayoutTagL1K = typename TileCopyQK::LayoutTagL1B;
        using LayoutTagL1H = typename TileCopyQH::LayoutTagL1B;

        auto layoutQGm = tla::MakeLayout<Element, LayoutRM>(m, kK);
        auto layoutKGm = tla::MakeLayout<Element, LayoutCM>(kK, m);
        auto tensorQGm = tla::MakeTensor(qGm_[qOffset], layoutQGm, Catlass::Arch::PositionGM{});
        auto tensorKGm = tla::MakeTensor(kGm_[kOffset], layoutKGm, Catlass::Arch::PositionGM{});
        auto blockQ = GetTile(tensorQGm, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto blockK = GetTile(tensorKGm, tla::MakeCoord(0, 0), tla::MakeShape(kK, m));

        using CopyGmToL1Q = typename TileCopyQK::template CopyGmToL1A<decltype(blockQ)>;
        using CopyGmToL1K = typename TileCopyQK::template CopyGmToL1B<decltype(blockK)>;

        LocalTensor<Element> l1Q = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1QOffset(l1StreamSlot));
        LocalTensor<Element> l1K = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1KOffset(l1StreamSlot));
        auto layoutL1Q = tla::MakeLayout<Element, LayoutTagL1Q>(m, kK);
        auto layoutL1K = tla::MakeLayout<Element, LayoutTagL1K>(kK, m);
        auto tensorL1Q = tla::MakeTensor(l1Q, layoutL1Q, Catlass::Arch::PositionL1{});
        auto tensorL1K = tla::MakeTensor(l1K, layoutL1K, Catlass::Arch::PositionL1{});

        CopyGmToL1Q{}(tensorL1Q, blockQ);
        CopyGmToL1K{}(tensorL1K, blockK);

        auto layoutHGm = tla::MakeLayout<Element, LayoutCM>(kK, kV);
        auto tensorHGm = tla::MakeTensor(hGm_[hOffset], layoutHGm, Catlass::Arch::PositionGM{});
        auto blockH = GetTile(tensorHGm, tla::MakeCoord(0, 0), tla::MakeShape(kK, kV));
        using CopyGmToL1H = typename TileCopyQH::template CopyGmToL1B<decltype(blockH)>;

        LocalTensor<Element> l1H = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1HOffset(l1StreamSlot));
        auto layoutL1H = tla::MakeLayout<Element, LayoutTagL1H>(kK, kV);
        auto tensorL1H = tla::MakeTensor(l1H, layoutL1H, Catlass::Arch::PositionL1{});
        CopyGmToL1H{}(tensorL1H, blockH);

        SetFlag<HardEvent::MTE2_MTE1>(L1Mte2Mte1Event(l1StreamSlot));
    }

    __aicore__ inline void ProcessStage2Head(const ChunkFwdOChunkLoc &loc, int64_t hk, int64_t hv,
                                             uint32_t ownerSubBlock, uint32_t localSlot, uint32_t headOffset,
                                             uint32_t l1StreamSlot)
    {
        (void)loc;
        (void)hk;
        (void)hv;
        const uint32_t m = kBt;

        WaitFlag<HardEvent::MTE2_MTE1>(L1Mte2Mte1Event(l1StreamSlot));

        const uint32_t qktASlot = l0ASlot_;
        const uint32_t qktBSlot = l0BSlot_;
        const uint32_t qktCSlot = l0CSlot_;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        RunGemmQKT(m, l1StreamSlot, qktASlot, qktBSlot, qktCSlot);
        PublishQKT(m, ownerSubBlock, localSlot, qktCSlot);

        const uint32_t qhASlot = l0ASlot_;
        const uint32_t qhBSlot = l0BSlot_;
        const uint32_t qhCSlot = l0CSlot_;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        RunGemmQH(m, l1StreamSlot, qhASlot, qhBSlot, qhCSlot);
        PublishQH(m, ownerSubBlock, localSlot, qhCSlot);

        SetFlag<HardEvent::MTE1_MTE2>(L1Mte1Mte2Event(l1StreamSlot));

        Catlass::Arch::CrossCoreFlag flag{
            static_cast<Catlass::Arch::FlagID>(CHUNK_FWD_O_CC_CUBE_READY_BASE + headOffset)};
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flag);
    }

    template <typename DirectTileCopy, typename TensorL0C>
    __aicore__ inline void PublishDirectTile(TensorL0C &tensorL0C, uint32_t m, uint32_t n, uint32_t ubOffset,
                                             uint32_t ownerSubBlock, uint32_t l0CSlot)
    {
        auto layoutUb = tla::MakeLayout<float, LayoutRM>(m, n);
        auto tensorUb = tla::MakeTensor(resource_.ubBuf.template GetBufferByByte<float>(ubOffset), layoutUb,
                                        Catlass::Arch::PositionUB{});
        using CopyL0CToDst = typename DirectTileCopy::template CopyL0CToDst<decltype(tensorUb)>;
        CopyL0CToDst copyL0CToDst;

        SetFlag<HardEvent::M_FIX>(l0CSlot);
        WaitFlag<HardEvent::M_FIX>(l0CSlot);
        copyL0CToDst(tensorUb, tensorL0C, static_cast<uint8_t>(ownerSubBlock), 0);
        SetFlag<HardEvent::FIX_M>(l0CSlot);
    }

    __aicore__ inline void PublishQKT(uint32_t m, uint32_t ownerSubBlock, uint32_t localSlot, uint32_t l0CSlot)
    {
        LocalTensor<float> qktL0C =
            resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(l0CSlot));
        auto qktLayout = tla::MakeLayoutL0C(m, m);
        auto qktTensor = tla::MakeTensor(qktL0C, qktLayout, Catlass::Arch::PositionL0C{});
        auto qktTile = GetTile(qktTensor, tla::MakeCoord(0, 0), tla::MakeShape(m, m));
        PublishDirectTile<DirectTileCopyCC>(qktTile, m, m, ChunkFwdOARawOffset(localSlot), ownerSubBlock, l0CSlot);
    }

    __aicore__ inline void PublishQH(uint32_t m, uint32_t ownerSubBlock, uint32_t localSlot, uint32_t l0CSlot)
    {
        LocalTensor<float> qhL0C = resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(l0CSlot));
        auto qhLayout = tla::MakeLayoutL0C(m, kV);
        auto qhTensor = tla::MakeTensor(qhL0C, qhLayout, Catlass::Arch::PositionL0C{});
        auto qhTile = GetTile(qhTensor, tla::MakeCoord(0, 0), tla::MakeShape(m, kV));
        PublishDirectTile<DirectTileCopyRM>(qhTile, m, kV, ChunkFwdOOSRawOffset(localSlot), ownerSubBlock, l0CSlot);
    }

    __aicore__ inline void RunGemmQKT(uint32_t m, uint32_t l1StreamSlot, uint32_t l0ASlot, uint32_t l0BSlot,
                                      uint32_t l0CSlot)
    {
        using LayoutTagL1A = typename TileCopyQK::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyQK::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopyQK::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyQK::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopyQK::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyQK::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, LayoutTagL1A>;

        LocalTensor<Element> l1Q = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1QOffset(l1StreamSlot));
        LocalTensor<Element> l1K = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1KOffset(l1StreamSlot));
        LocalTensor<Element> l0A =
            resource_.l0ABuf.template GetBufferByByte<Element>(ChunkFwdOL0AOffset(l0ASlot));
        LocalTensor<Element> l0B =
            resource_.l0BBuf.template GetBufferByByte<Element>(ChunkFwdOL0BOffset(l0BSlot));
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(l0CSlot));

        auto layoutL1Q = tla::MakeLayout<Element, LayoutTagL1A>(m, kK);
        auto layoutL1K = tla::MakeLayout<Element, LayoutTagL1B>(kK, m);
        auto layoutL0A = tla::MakeLayout<Element, LayoutTagL0A>(m, kK);
        auto layoutL0B = tla::MakeLayout<Element, LayoutTagL0B>(kK, m);
        auto layoutL0C = tla::MakeLayoutL0C(m, m);
        auto tensorL1Q = tla::MakeTensor(l1Q, layoutL1Q, Catlass::Arch::PositionL1{});
        auto tensorL1K = tla::MakeTensor(l1K, layoutL1K, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1Q = GetTile(tensorL1Q, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto tileL1K = GetTile(tensorL1K, tla::MakeCoord(0, 0), tla::MakeShape(kK, m));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(kK, m));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, m));

        WaitFlag<HardEvent::M_MTE1>(L0AEvent(l0ASlot));
        WaitFlag<HardEvent::M_MTE1>(L0BEvent(l0BSlot));
        WaitFlag<HardEvent::FIX_M>(l0CSlot);
        CopyL1ToL0A{}(tileL0A, tileL1Q);
        CopyL1ToL0B{}(tileL0B, tileL1K);
        SetFlag<HardEvent::MTE1_M>(l0CSlot);
        WaitFlag<HardEvent::MTE1_M>(l0CSlot);
        TileMmad{}(tileL0C, tileL0A, tileL0B, m, m, kK, true, 0);
        SetFlag<HardEvent::M_MTE1>(L0AEvent(l0ASlot));
        SetFlag<HardEvent::M_MTE1>(L0BEvent(l0BSlot));
    }

    __aicore__ inline void RunGemmQH(uint32_t m, uint32_t l1StreamSlot, uint32_t l0ASlot, uint32_t l0BSlot,
                                     uint32_t l0CSlot)
    {
        using LayoutTagL1A = typename TileCopyQH::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyQH::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopyQH::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyQH::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopyQH::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyQH::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, LayoutTagL1A>;

        LocalTensor<Element> l1Q = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1QOffset(l1StreamSlot));
        LocalTensor<Element> l1H = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1HOffset(l1StreamSlot));
        LocalTensor<Element> l0A =
            resource_.l0ABuf.template GetBufferByByte<Element>(ChunkFwdOL0AOffset(l0ASlot));
        LocalTensor<Element> l0B =
            resource_.l0BBuf.template GetBufferByByte<Element>(ChunkFwdOL0BOffset(l0BSlot));
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(l0CSlot));

        auto layoutL1Q = tla::MakeLayout<Element, LayoutTagL1A>(m, kK);
        auto layoutL1H = tla::MakeLayout<Element, LayoutTagL1B>(kK, kV);
        auto layoutL0A = tla::MakeLayout<Element, LayoutTagL0A>(m, kK);
        auto layoutL0B = tla::MakeLayout<Element, LayoutTagL0B>(kK, kV);
        auto layoutL0C = tla::MakeLayoutL0C(m, kV);
        auto tensorL1Q = tla::MakeTensor(l1Q, layoutL1Q, Catlass::Arch::PositionL1{});
        auto tensorL1H = tla::MakeTensor(l1H, layoutL1H, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1Q = GetTile(tensorL1Q, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto tileL1H = GetTile(tensorL1H, tla::MakeCoord(0, 0), tla::MakeShape(kK, kV));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(kK, kV));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, kV));

        WaitFlag<HardEvent::M_MTE1>(L0AEvent(l0ASlot));
        WaitFlag<HardEvent::M_MTE1>(L0BEvent(l0BSlot));
        WaitFlag<HardEvent::FIX_M>(l0CSlot);
        CopyL1ToL0A{}(tileL0A, tileL1Q);
        CopyL1ToL0B{}(tileL0B, tileL1H);
        SetFlag<HardEvent::MTE1_M>(l0CSlot);
        WaitFlag<HardEvent::MTE1_M>(l0CSlot);
        TileMmad{}(tileL0C, tileL0A, tileL0B, m, kV, kK, true, 0);
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(L0AEvent(l0ASlot));
        SetFlag<HardEvent::M_MTE1>(L0BEvent(l0BSlot));
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
    Catlass::Arch::Resource<ArchTag> resource_;
    GlobalTensor<Element> qGm_;
    GlobalTensor<Element> kGm_;
    GlobalTensor<Element> hGm_;
    Catlass::Arch::CrossCoreFlag groupReleaseFlag_{CHUNK_FWD_O_VEC_TO_CUBE_RELEASE_FLAG};
    Catlass::Arch::CrossCoreFlag stage1GroupDoneFlag_{CHUNK_FWD_O_S1_GROUP_DONE_FLAG};
    uint32_t l0ASlot_ = 0U;
    uint32_t l0BSlot_ = 0U;
    uint32_t l0CSlot_ = 0U;
    uint32_t l1StreamSlot_ = 0U;
};

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_CUBE_H
