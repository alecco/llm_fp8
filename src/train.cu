// train fp8 GEMM
// Copyright (C) 2024  Ologan Ltd
// SPDX-License-Identifier: AGPL-3.0
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//
// Train GPT-2 on FP8
//

#include <iostream>
#include <format>
#include <getopt.h>
//#include <limits>

#include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
// #include <cuda_fp8.h>
// #include <cute/tensor.hpp>

#include <util/cuda_check.h>
#include <util/format_helpers.h>

// using namespace llm;

bool verbose = false;

enum Precision {
    FP32,
    FP16,
    BF16,
    FP8
};

template <>
struct std::formatter<Precision> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(Precision p, format_context& ctx) const {
        switch (p) {
            case FP32: return std::format_to(ctx.out(), "fp32");
            case FP16: return std::format_to(ctx.out(), "fp16");
            case BF16: return std::format_to(ctx.out(), "bf16");
            case FP8:  return std::format_to(ctx.out(), "fp8");
        }
        return ctx.out();
    }
};

void
device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << std::format("GPU {:2}: {}\n", device, deviceProp.name);
        std::cout << std::format("        Compute: {}.{}  MB: {} MPs: {} thr/blk: {} regs/blk: {}k\n",
            deviceProp.totalGlobalMem / (1024 * 1024), deviceProp.major, deviceProp.minor,
            deviceProp.multiProcessorCount,
            deviceProp.maxThreadsPerBlock, deviceProp.regsPerBlock / 1024
            );

        std::cout << std::format("        max shared m/blk: {} KB shared m/MP: {} KB, max warps/MP: {}\n",
            deviceProp.sharedMemPerBlock / 1024, deviceProp.sharedMemPerMultiprocessor / 1024,
            deviceProp.maxThreadsPerMultiProcessor / 32);
    }
}

int
main(int argc, char* argv[]) {

    // Define the long options
    static struct option long_options[] = {
        {"v",         required_argument, nullptr, 'v'},
        {nullptr, 0, nullptr, 0}
    };

    // Parse the command-line arguments
    int option_index = 0;
    int opt;
    while ((opt = getopt_long(argc, argv, "v", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'v':
                verbose = true;
                break;
            default:
                std::cout << std::format("Usage: {} [-v]\n", argv[0]);
                return 1;
        }
    }

    if (verbose) {
        device_info();
    }

    return 0;
}
