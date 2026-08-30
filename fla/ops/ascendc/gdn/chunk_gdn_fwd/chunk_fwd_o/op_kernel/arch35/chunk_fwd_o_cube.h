/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
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

    static constexpr TEventID kEventMte2Mte1 = 0;
    static constexpr TEventID kEventMte1M = 4;
    static constexpr TEventID kEventQktL0C = 0;
    static constexpr TEventID kEventQhL0C = 1;
    static constexpr TEventID kEventQktL0A = 0;
    static constexpr TEventID kEventQktL0B = 1;
    static constexpr TEventID kEventQhL0A = 2;
    static constexpr TEventID kEventQhL0B = 3;

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
            SetFlag<HardEvent::M_MTE1>(kEventQktL0A);
            SetFlag<HardEvent::M_MTE1>(kEventQktL0B);
            SetFlag<HardEvent::M_MTE1>(kEventQhL0A);
            SetFlag<HardEvent::M_MTE1>(kEventQhL0B);
            SetFlag<HardEvent::FIX_M>(kEventQktL0C);
            SetFlag<HardEvent::FIX_M>(kEventQhL0C);
        }
    }

    __aicore__ inline void ProcessStage2(uint32_t loopIdx, const ChunkFwdOChunkLoc &loc, int64_t hk, int64_t hv,
                                         uint32_t ownerSubBlock, uint32_t localSlot, uint32_t headOffset)
    {
        if ASCEND_IS_AIV {
            return;
        }

        (void)loopIdx;
        const uint32_t m = kBt;
        const int64_t qOffset = ChunkFwdOQKOffset(tiling_, loc, hk);
        const int64_t kOffset = ChunkFwdOQKOffset(tiling_, loc, hk);
        const int64_t hOffset = ChunkFwdOHOffset(tiling_, loc, hv);

        LoadQKToL1(qOffset, kOffset, m);
        ComputeQKTToUb(m);
        PublishQKT(m, ownerSubBlock, localSlot, kEventQktL0C);
        LoadHToL1(hOffset, m);
        ComputeQHToUb(m);
        if (headOffset == 0) {
            Catlass::Arch::CrossCoreWaitFlag(stage1GroupDoneFlag_);
        }
        PublishQH(m, ownerSubBlock, localSlot, kEventQhL0C);
        Catlass::Arch::CrossCoreFlag flag{
            static_cast<Catlass::Arch::FlagID>(CHUNK_FWD_O_CC_CUBE_READY_BASE + headOffset)};
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flag);
    }

    __aicore__ inline void WaitStage1Ready(uint32_t headOffset)
    {
        Catlass::Arch::CrossCoreFlag flag{
            static_cast<Catlass::Arch::FlagID>(CHUNK_FWD_O_S1_READY_BASE + headOffset)};
        Catlass::Arch::CrossCoreWaitFlag(flag);
    }

    __aicore__ inline void WaitStage2GroupRelease()
    {
        Catlass::Arch::CrossCoreWaitFlag(groupReleaseFlag_);
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
    template <typename DstElement, typename DirectTileCopy, typename TensorL0C>
    __aicore__ inline void PublishDirectTile(TensorL0C &tensorL0C, uint32_t m, uint32_t n,
                                             uint32_t ubOffset, uint32_t ownerSubBlock, TEventID l0CEvent)
    {
        auto layoutUb = tla::MakeLayout<DstElement, LayoutRM>(m, n);
        auto tensorUb = tla::MakeTensor(resource_.ubBuf.template GetBufferByByte<DstElement>(ubOffset),
                                        layoutUb, Catlass::Arch::PositionUB{});
        using CopyL0CToDst = typename DirectTileCopy::template CopyL0CToDst<decltype(tensorUb)>;
        CopyL0CToDst copyL0CToDst;

        SetFlag<HardEvent::M_FIX>(l0CEvent);
        WaitFlag<HardEvent::M_FIX>(l0CEvent);
        copyL0CToDst(tensorUb, tensorL0C, static_cast<uint8_t>(ownerSubBlock), 0);
        SetFlag<HardEvent::FIX_M>(l0CEvent);
    }

    __aicore__ inline void PublishQKT(uint32_t m, uint32_t ownerSubBlock, uint32_t localSlot,
                                      TEventID l0CEvent)
    {
        LocalTensor<float> qktL0C = resource_.l0CBuf.template GetBufferByByte<float>(0);
        auto qktLayout = tla::MakeLayoutL0C(m, m);
        auto qktTensor = tla::MakeTensor(qktL0C, qktLayout, Catlass::Arch::PositionL0C{});
        auto qktTile = GetTile(qktTensor, tla::MakeCoord(0, 0), tla::MakeShape(m, m));
        PublishDirectTile<float, DirectTileCopyCC>(
            qktTile, m, m, ChunkFwdOARawOffset(localSlot), ownerSubBlock, l0CEvent);
    }

    __aicore__ inline void PublishQH(uint32_t m, uint32_t ownerSubBlock, uint32_t localSlot,
                                     TEventID l0CEvent)
    {
        LocalTensor<float> qhL0C =
            resource_.l0CBuf.template GetBufferByByte<float>(CHUNK_FWD_O_L0C_QH_OFFSET);
        auto qhLayout = tla::MakeLayoutL0C(m, kV);
        auto qhTensor = tla::MakeTensor(qhL0C, qhLayout, Catlass::Arch::PositionL0C{});
        auto qhTile = GetTile(qhTensor, tla::MakeCoord(0, 0), tla::MakeShape(m, kV));
        PublishDirectTile<float, DirectTileCopyRM>(
            qhTile, m, kV, ChunkFwdOOSRawOffset(localSlot), ownerSubBlock, l0CEvent);
    }

    __aicore__ inline void LoadQKToL1(int64_t qOffset, int64_t kOffset, uint32_t m)
    {
        using LayoutTagL1Q = typename TileCopyQK::LayoutTagL1A;
        using LayoutTagL1K = typename TileCopyQK::LayoutTagL1B;

        auto layoutQGm = tla::MakeLayout<Element, LayoutRM>(m, kK);
        auto layoutKGm = tla::MakeLayout<Element, LayoutCM>(kK, m);
        auto tensorQGm = tla::MakeTensor(qGm_[qOffset], layoutQGm, Catlass::Arch::PositionGM{});
        auto tensorKGm = tla::MakeTensor(kGm_[kOffset], layoutKGm, Catlass::Arch::PositionGM{});
        auto blockQ = GetTile(tensorQGm, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto blockK = GetTile(tensorKGm, tla::MakeCoord(0, 0), tla::MakeShape(kK, m));

        using CopyGmToL1Q = typename TileCopyQK::template CopyGmToL1A<decltype(blockQ)>;
        using CopyGmToL1K = typename TileCopyQK::template CopyGmToL1B<decltype(blockK)>;

        LocalTensor<Element> l1Q = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1QOffset(0));
        LocalTensor<Element> l1K = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1KOffset(0));
        auto layoutL1Q = tla::MakeLayout<Element, LayoutTagL1Q>(m, kK);
        auto layoutL1K = tla::MakeLayout<Element, LayoutTagL1K>(kK, m);
        auto tensorL1Q = tla::MakeTensor(l1Q, layoutL1Q, Catlass::Arch::PositionL1{});
        auto tensorL1K = tla::MakeTensor(l1K, layoutL1K, Catlass::Arch::PositionL1{});

        CopyGmToL1Q copyGmToL1Q;
        CopyGmToL1K copyGmToL1K;
        copyGmToL1Q(tensorL1Q, blockQ);
        copyGmToL1K(tensorL1K, blockK);
        SetFlag<HardEvent::MTE2_MTE1>(kEventMte2Mte1);
    }

    __aicore__ inline void LoadHToL1(int64_t hOffset, uint32_t m)
    {
        using LayoutTagL1B = typename TileCopyQH::LayoutTagL1B;

        // Physical H is [V,K] row-major; view it as [K,V] column-major.
        auto layoutHGm = tla::MakeLayout<Element, LayoutCM>(kK, kV);
        auto tensorHGm = tla::MakeTensor(hGm_[hOffset], layoutHGm, Catlass::Arch::PositionGM{});
        auto blockH = GetTile(tensorHGm, tla::MakeCoord(0, 0), tla::MakeShape(kK, kV));

        using CopyGmToL1H = typename TileCopyQH::template CopyGmToL1B<decltype(blockH)>;

        LocalTensor<Element> l1H = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1HOffset(0));
        auto layoutL1H = tla::MakeLayout<Element, LayoutTagL1B>(kK, kV);
        auto tensorL1H = tla::MakeTensor(l1H, layoutL1H, Catlass::Arch::PositionL1{});

        CopyGmToL1H copyGmToL1H;
        copyGmToL1H(tensorL1H, blockH);
        SetFlag<HardEvent::MTE2_MTE1>(kEventMte2Mte1);
        (void)m;
    }

    __aicore__ inline void ComputeQKTToUb(uint32_t m)
    {
        using LayoutTagL1A = typename TileCopyQK::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyQK::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopyQK::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyQK::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopyQK::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyQK::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, LayoutTagL1A>;

        LocalTensor<Element> l1Q = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1QOffset(0));
        LocalTensor<Element> l1K = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1KOffset(0));
        LocalTensor<Element> l0A = resource_.l0ABuf.template GetBufferByByte<Element>(0);
        LocalTensor<Element> l0B = resource_.l0BBuf.template GetBufferByByte<Element>(0);
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(0);

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

        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;

        WaitFlag<HardEvent::MTE2_MTE1>(kEventMte2Mte1);
        WaitFlag<HardEvent::M_MTE1>(kEventQktL0A);
        WaitFlag<HardEvent::M_MTE1>(kEventQktL0B);
        WaitFlag<HardEvent::FIX_M>(kEventQktL0C);
        copyL1ToL0A(tileL0A, tileL1Q);
        copyL1ToL0B(tileL0B, tileL1K);
        SetFlag<HardEvent::MTE1_M>(kEventMte1M);
        WaitFlag<HardEvent::MTE1_M>(kEventMte1M);
        tileMmad(tileL0C, tileL0A, tileL0B, m, m, kK, true, 0);
        SetFlag<HardEvent::M_MTE1>(kEventQktL0A);
        SetFlag<HardEvent::M_MTE1>(kEventQktL0B);

    }

    __aicore__ inline void ComputeQHToUb(uint32_t m)
    {
        using LayoutTagL1A = typename TileCopyQH::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyQH::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopyQH::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyQH::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopyQH::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyQH::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, LayoutTagL1A>;

        LocalTensor<Element> l1Q = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1QOffset(0));
        LocalTensor<Element> l1H = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1HOffset(0));
        LocalTensor<Element> l0A =
            resource_.l0ABuf.template GetBufferByByte<Element>(CHUNK_FWD_O_L0AB_QH_OFFSET);
        LocalTensor<Element> l0B =
            resource_.l0BBuf.template GetBufferByByte<Element>(CHUNK_FWD_O_L0AB_QH_OFFSET);
        LocalTensor<float> l0C =
            resource_.l0CBuf.template GetBufferByByte<float>(CHUNK_FWD_O_L0C_QH_OFFSET);

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

        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;

        WaitFlag<HardEvent::MTE2_MTE1>(kEventMte2Mte1);
        WaitFlag<HardEvent::M_MTE1>(kEventQhL0A);
        WaitFlag<HardEvent::M_MTE1>(kEventQhL0B);
        WaitFlag<HardEvent::FIX_M>(kEventQhL0C);
        copyL1ToL0A(tileL0A, tileL1Q);
        copyL1ToL0B(tileL0B, tileL1H);
        SetFlag<HardEvent::MTE1_M>(kEventMte1M);
        WaitFlag<HardEvent::MTE1_M>(kEventMte1M);
        tileMmad(tileL0C, tileL0A, tileL0B, m, kV, kK, true, 0);
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(kEventQhL0A);
        SetFlag<HardEvent::M_MTE1>(kEventQhL0B);

    }

    __aicore__ inline void DumpStage2(uint32_t loopIdx, int64_t hv, uint32_t m)
    {
        if ASCEND_IS_AIC {
            if (ChunkFwdODumpEnabled(tiling_)) {
                const int64_t slotIdx = ChunkFwdODumpSlotIndex(tiling_, loopIdx, hv);
                GM_ADDR slotBase = ChunkFwdODumpSlotPtr(workspace_, tiling_, slotIdx);
                LocalTensor<float> aRaw =
                    resource_.ubBuf.template GetBufferByByte<float>(ChunkFwdOARawOffset(0));
                LocalTensor<float> oSRaw =
                    resource_.ubBuf.template GetBufferByByte<float>(ChunkFwdOOSRawOffset(0));
                ChunkFwdODumpUbToGm(slotBase, CHUNK_FWD_O_DBG_ARAW_OFF, aRaw, m * m);
                ChunkFwdODumpUbToGm(slotBase, CHUNK_FWD_O_DBG_OSRAW_OFF, oSRaw, m * kV);
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
    Catlass::Arch::Resource<ArchTag> resource_;
    GlobalTensor<Element> qGm_;
    GlobalTensor<Element> kGm_;
    GlobalTensor<Element> hGm_;
    Catlass::Arch::CrossCoreFlag groupReleaseFlag_{CHUNK_FWD_O_VEC_TO_CUBE_RELEASE_FLAG};
    Catlass::Arch::CrossCoreFlag stage1GroupDoneFlag_{CHUNK_FWD_O_S1_GROUP_DONE_FLAG};
};

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_CUBE_H
