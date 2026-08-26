/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_o.cpp
 * \brief
 */

#include "chunk_fwd_o_struct.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_fwd_o_a5.h"
#else
#include "gemm/kernel/gdn_fwd_o_kernel.hpp"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif
#endif

namespace GDN {

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310

template <typename InputT, typename GT, typename WorkspaceT>
__aicore__ inline void ChunkFwdOKernelImpl(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                           GM_ADDR cuSeqlens, GM_ADDR chunkOffsets, GM_ADDR o,
                                           GM_ADDR userWorkspace, const ChunkFwdOTilingData *tilingData)
{
    using GDNFwdOKernel = Catlass::Gemm::Kernel::GDNFwdOKernel<InputT, GT, WorkspaceT>;
    GDNFwdOKernel gdnFwdO;
    gdnFwdO.Init(q, k, v, h, g, cuSeqlens, chunkOffsets, o, tilingData, userWorkspace);
    gdnFwdO.Process();
}

#endif

__aicore__ inline void ChunkFwdODispatch(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                         GM_ADDR cuSeqlens, GM_ADDR chunkOffsets, GM_ADDR o,
                                         GM_ADDR userWorkspace, const ChunkFwdOTilingData *tilingData)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    if (tilingData->dataType == 1) {
        if (tilingData->gDataType == 2) {
            ChunkFwdOA5Dispatch<bfloat16_t, float>(q, k, v, h, g, cuSeqlens, chunkOffsets, o, userWorkspace,
                                                   tilingData);
        } else {
            ChunkFwdOA5Dispatch<bfloat16_t, bfloat16_t>(q, k, v, h, g, cuSeqlens, chunkOffsets, o, userWorkspace,
                                                        tilingData);
        }
    } else {
        if (tilingData->gDataType == 2) {
            ChunkFwdOA5Dispatch<half, float>(q, k, v, h, g, cuSeqlens, chunkOffsets, o, userWorkspace, tilingData);
        } else {
            ChunkFwdOA5Dispatch<half, half>(q, k, v, h, g, cuSeqlens, chunkOffsets, o, userWorkspace, tilingData);
        }
    }
#else
    using WorkspaceT = float;
    if (tilingData->dataType == 1) {
        if (tilingData->gDataType == 2) {
            ChunkFwdOKernelImpl<bfloat16_t, float, WorkspaceT>(q, k, v, h, g, cuSeqlens, chunkOffsets, o,
                                                               userWorkspace, tilingData);
        } else {
            ChunkFwdOKernelImpl<bfloat16_t, bfloat16_t, WorkspaceT>(q, k, v, h, g, cuSeqlens, chunkOffsets, o,
                                                                    userWorkspace, tilingData);
        }
    } else {
        if (tilingData->gDataType == 2) {
            ChunkFwdOKernelImpl<half, float, WorkspaceT>(q, k, v, h, g, cuSeqlens, chunkOffsets, o, userWorkspace,
                                                         tilingData);
        } else {
            ChunkFwdOKernelImpl<half, half, WorkspaceT>(q, k, v, h, g, cuSeqlens, chunkOffsets, o, userWorkspace,
                                                        tilingData);
        }
    }
#endif
}

} // namespace GDN

#ifndef TORCH_MODE
extern "C" __global__ __aicore__ void chunk_fwd_o(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h,
                                                   GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_offsets,
                                                   GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    REGISTER_TILING_DEFAULT(GDN::ChunkFwdOTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::ChunkFwdOTilingData, tilingData, tiling);

    GDN::ChunkFwdODispatch(q, k, v, h, g, cu_seqlens, chunk_offsets, o, user, &tilingData);
}
#endif
