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
// #include <memory>
#include <string>

#include <cuda_runtime.h>
// #include <cuda_fp16.h>
#include <cuda_bf16.h>
// #include <cuda_fp8.h>
// #include <cute/tensor.hpp>

#include <runner.h>
#include <model_gpt2.h>
#include <tokenizer_gpt2.h>
#include <util/io_check.h>
#include <util/format_helpers.h>

// using namespace llm;

using namespace util;

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
    const char* input_dataset_prefix = "../llm.c/dev/data/tinyshakespeare/tiny_shakespeare";
    const char* token_table_file     = "../llm.c/gpt2_tokenizer.bin";
    std::string load_filename        = "../llm.c/gpt2_124M_bf16.bin"; // bf16 weights of the model
    bool overfit_single_batch        = false; // true = only load a single data batch once (debugging)
    int batch_size                   = 4;
    int seq_len_max                  = 1024;   // maximum sequence lenght
    std::string train_tokens_filename;
    std::string val_tokens_filename;

    // Define the long options
    static struct option long_options[] = {
        {"verbose",        no_argument,       nullptr, 'v'},
        {"overfit-single", no_argument,       nullptr, 'o'},
        {"input",          required_argument, nullptr, 'i'},
        {"batch-size",     required_argument, nullptr, 'B'},
        {"seq-len-max",    required_argument, nullptr, 'T'},
        {nullptr,          0,                 nullptr,  0}
    };

    // Parse the command-line arguments
    int option_index = 0;
    int opt;
    while ((opt = getopt_long(argc, argv, "vi:o:", long_options, &option_index)) != -1) {
        std::cerr << std::format("XXX opt {}\n", opt);
        switch (opt) {
            case 'v':
                verbose = true;
                break;
            case 'o':
                overfit_single_batch = true;
                break;
            case 'i':
                std::cerr << std::format("XXX opt {:c} optarg {}\n", opt, optarg);
                input_dataset_prefix = optarg;
                break;
            case 'B':
                batch_size = std::stoi(optarg);
                break;
            case 'T':
                seq_len_max = std::stoi(optarg);
                break;
            default:
                std::cout << std::format("Usage: {0} [-v] [-o]\n"
                                         "           --input <dataset prefix>\n"
                                         "           --batch-size <num>\n"
                                         "           --seq-len-max <num>\n"
                                         "e.g.   {0} -i data/TinyStories\n", argv[0]);
                return 1;
        }
    }

    if (verbose) {
        device_info();
        std::cout << std::format(
                "dataset: {}\n"
                "overfit_single_batch: {}\n",
                input_dataset_prefix, overfit_single_batch);
    }

    train_tokens_filename = std::format("{}_{}.bin", input_dataset_prefix,
                                        overfit_single_batch == 1? "val" : "train");
    // val_tokens_filename = std::format("{}_val.bin", input_dataset_prefix);
std::cout << std::format("XXX train_tokens_filename {}\n", train_tokens_filename); // XXX

    using ModelParameterType = __nv_bfloat16;  // type for model parameters
    using InternalTokenType  =  int32_t;       // type for internal tokens
    using FileTokenType      = uint16_t;       // type for internal tokens

    gpt2::Tokenizer<InternalTokenType> tokenizer(token_table_file);
    gpt2::TokenLoader<FileTokenType, InternalTokenType> train_loader(train_tokens_filename, batch_size, seq_len_max
            , tokenizer
            );
    // gpt2::TokenLoader<uint16_t, int> val_loader(val_tokens_filename, batch_size, seq_len_max, tokenizer);
    auto model = gpt2::TrainableModel<ModelParameterType, InternalTokenType>(load_filename);
    int num_epochs = 2;
    llm::train(model, train_loader, num_epochs);

    return 0;
}
