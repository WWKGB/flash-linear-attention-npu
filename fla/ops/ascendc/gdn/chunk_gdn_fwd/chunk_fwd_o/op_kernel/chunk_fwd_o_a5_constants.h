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

} // namespace GDN

#endif // CHUNK_FWD_O_A5_CONSTANTS_H
