/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 tile constants shared by host tiling and device kernel.
 */

#ifndef CHUNK_FWD_O_A5_CONSTANTS_H
#define CHUNK_FWD_O_A5_CONSTANTS_H

#include <cstdint>

namespace GDN {

constexpr int64_t CHUNK_FWD_O_A5_BT = 64;
constexpr int64_t CHUNK_FWD_O_A5_K = 128;
constexpr int64_t CHUNK_FWD_O_A5_V = 128;

// Debug dump layout (development-only GM workspace after sys workspace).
constexpr uint32_t CHUNK_FWD_O_DBG_MAGIC = 0xCF0DA5U;
constexpr uint32_t CHUNK_FWD_O_DBG_VERSION = 1U;
constexpr uint32_t CHUNK_FWD_O_DBG_HEADER_BYTES = 64U;
constexpr uint32_t CHUNK_FWD_O_DBG_MASK_BYTES = 4U * 1024U;
constexpr uint32_t CHUNK_FWD_O_DBG_GATE_O_BYTES = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT * sizeof(float));
constexpr uint32_t CHUNK_FWD_O_DBG_GATE_A_BYTES =
    static_cast<uint32_t>(CHUNK_FWD_O_A5_BT * CHUNK_FWD_O_A5_BT * sizeof(float));
constexpr uint32_t CHUNK_FWD_O_DBG_ARAW_BYTES =
    static_cast<uint32_t>(CHUNK_FWD_O_A5_BT * CHUNK_FWD_O_A5_BT * sizeof(float));
constexpr uint32_t CHUNK_FWD_O_DBG_OSRAW_BYTES =
    static_cast<uint32_t>(CHUNK_FWD_O_A5_BT * CHUNK_FWD_O_A5_V * sizeof(float));
constexpr uint32_t CHUNK_FWD_O_DBG_APRIME_BYTES =
    static_cast<uint32_t>(CHUNK_FWD_O_A5_BT * CHUNK_FWD_O_A5_BT * sizeof(uint16_t)); // bf16
constexpr uint32_t CHUNK_FWD_O_DBG_OSPRIME_BYTES =
    static_cast<uint32_t>(CHUNK_FWD_O_A5_BT * CHUNK_FWD_O_A5_V * sizeof(float)); // fp32
constexpr uint32_t CHUNK_FWD_O_DBG_OL_BYTES =
    static_cast<uint32_t>(CHUNK_FWD_O_A5_BT * CHUNK_FWD_O_A5_V * sizeof(float)); // fp32
constexpr uint32_t CHUNK_FWD_O_DBG_MASK_OFF = 0U;
constexpr uint32_t CHUNK_FWD_O_DBG_GATE_O_OFF = CHUNK_FWD_O_DBG_MASK_BYTES;
constexpr uint32_t CHUNK_FWD_O_DBG_GATE_A_OFF = CHUNK_FWD_O_DBG_GATE_O_OFF + CHUNK_FWD_O_DBG_GATE_O_BYTES;
constexpr uint32_t CHUNK_FWD_O_DBG_ARAW_OFF = CHUNK_FWD_O_DBG_GATE_A_OFF + CHUNK_FWD_O_DBG_GATE_A_BYTES;
constexpr uint32_t CHUNK_FWD_O_DBG_OSRAW_OFF = CHUNK_FWD_O_DBG_ARAW_OFF + CHUNK_FWD_O_DBG_ARAW_BYTES;
constexpr uint32_t CHUNK_FWD_O_DBG_APRIME_OFF = CHUNK_FWD_O_DBG_OSRAW_OFF + CHUNK_FWD_O_DBG_OSRAW_BYTES;
constexpr uint32_t CHUNK_FWD_O_DBG_OSPRIME_OFF = CHUNK_FWD_O_DBG_APRIME_OFF + CHUNK_FWD_O_DBG_APRIME_BYTES;
constexpr uint32_t CHUNK_FWD_O_DBG_OL_OFF = CHUNK_FWD_O_DBG_OSPRIME_OFF + CHUNK_FWD_O_DBG_OSPRIME_BYTES;
constexpr uint32_t CHUNK_FWD_O_DBG_SLOT_BYTES =
    (CHUNK_FWD_O_DBG_OL_OFF + CHUNK_FWD_O_DBG_OL_BYTES + 511U) / 512U * 512U;

constexpr uint32_t CHUNK_FWD_O_APRIME_SLOT_BYTES =
    static_cast<uint32_t>(CHUNK_FWD_O_A5_BT * CHUNK_FWD_O_A5_BT * sizeof(uint16_t));
constexpr uint32_t CHUNK_FWD_O_APRIME_SLOT_COUNT = 4U;
constexpr uint32_t CHUNK_FWD_O_APRIME_WORKSPACE_BYTES =
    CHUNK_FWD_O_APRIME_SLOT_BYTES * CHUNK_FWD_O_APRIME_SLOT_COUNT;

} // namespace GDN

#endif // CHUNK_FWD_O_A5_CONSTANTS_H
