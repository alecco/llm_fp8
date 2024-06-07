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
// GPT-2 model
// With compile-time config.
//

#pragma once

#include <algorithm>
#include <cassert>
#include <format>
#include <fstream>
#include <iostream>
#include <string>

#include <model.h>

// #include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <util/cuda_check.h>
#include <util/cuda_memory.h>
#include <util/assert.h>

namespace gpt2 {

using util::assertf;
using util::cuda_check;
using util::CudaMemory;

template<typename P, size_t max_seq_len, size_t num_layers, size_t channels, size_t vocab_size_p>
class Parameters {
    static constexpr size_t wte_size      = vocab_size_p * channels;
    static constexpr size_t wpe_size      = max_seq_len  * channels;
    static constexpr size_t ln1w_size     = num_layers   * channels;
    static constexpr size_t ln1b_size     = num_layers   * channels;
    static constexpr size_t qkvw_size     = num_layers   * (3 * channels) * channels;
    static constexpr size_t qkvb_size     = num_layers   * (3 * channels);
    static constexpr size_t attprojw_size = num_layers   * channels * channels;
    static constexpr size_t attprojb_size = num_layers   * channels;
    static constexpr size_t ln2w_size     = num_layers   * channels;
    static constexpr size_t ln2b_size     = num_layers   * channels;
    static constexpr size_t fcw_size      = num_layers   * (4 * channels) * channels;
    static constexpr size_t fcb_size      = num_layers   * (4 * channels);
    static constexpr size_t fcprojw_size  = num_layers   * channels * (4 * channels);
    static constexpr size_t fcprojb_size  = num_layers   * channels;
    static constexpr size_t lnfw_size     = channels;
    static constexpr size_t lnfb_size     = channels;
public:
    static constexpr size_t num_params    = wte_size + wpe_size + ln1w_size + ln1b_size + \
                                            qkvw_size + qkvb_size + attprojw_size + \
                                            attprojb_size + ln2w_size + ln2b_size + \
                                            fcw_size + fcb_size + fcprojw_size + \
                                            fcprojb_size + lnfw_size + lnfb_size;
    P* wte;       // weight matrix for token embeds     (vocab padded, embedding dim/chan)
    P* wpe;       // weight matrix for position embeds  (max seq len, embedding dim)
    P* ln1w;      // layer normalization weights        (layers, hidden dim)
    P* ln1b;      // layer normalization biases         (layers, hidden dim)
    P* qkvw;      // Query, Key, Value proj weights     (layers, 3 hidden dim, hidden dim)
    P* qkvb;      // Query, Key, Value proj biases      (layers, 3 hidden dim)
    P* attprojw;  // Attention projection weights       (layers, hidden dim, hidden dim)
    P* attprojb;  // Attention projection biases        (layers, hidden dim)
    P* ln2w;      // Layer normalization weights        (layers, hidden dim)
    P* ln2b;      // Layer normalization biases         (layers, hidden dim)
    P* fcw;       // Fully connected layer weights      (layers, 4 layers, hidden dim)
    P* fcb;       // Fully connected layer biases       (layers, 4 layers)
    P* fcprojw;   // Fully connected proj weights       (layers, hidden dim, 4 hidden dim)
    P* fcprojb;   // Fully connected proj biases        (layers, hidden dim)
    P* lnfw;      // Layer normalization weights        (hidden dim)
    P* lnfb;      // Layer normalization biases         (hidden dim)
    Parameters(P* base_ptr) : wte(base_ptr),
                              wpe(wte           + wte_size),
                              ln1w(wpe          + wpe_size),
                              ln1b(ln1w         + ln1w_size),
                              qkvw(ln1b         + ln1b_size),
                              qkvb(qkvw         + qkvw_size),
                              attprojw(qkvb     + qkvb_size),
                              attprojb(attprojw + attprojw_size),
                              ln2w(attprojb     + attprojb_size),
                              ln2b(ln2w         + ln2w_size),
                              fcw(ln2b          + ln2b_size),
                              fcb(fcw           + fcw_size),
                              fcprojw(fcb       + fcb_size),
                              fcprojb(fcprojw   + fcprojw_size),
                              lnfw(fcprojb      + fcprojb_size),
                              lnfb(lnfw         + lnfw_size)      {}
};

template<typename A, size_t seq_len, size_t num_layers, size_t num_heads, size_t channels,
         size_t vocab_size_p, size_t batch_size>
class Activations {
    static constexpr size_t encoded_size   = batch_size * seq_len * channels;
    static constexpr size_t ln1_size       = num_layers * batch_size * seq_len * channels;
    static constexpr size_t ln1_mean_size  = num_layers * batch_size * seq_len;
    static constexpr size_t ln1_rstd_size  = num_layers * batch_size * seq_len;
    static constexpr size_t atty_size      = num_layers * batch_size * seq_len * channels;
    static constexpr size_t att_size       = num_layers * batch_size * num_heads * seq_len * seq_len;
    static constexpr size_t attproj_size   = num_layers * batch_size * seq_len * channels;
    static constexpr size_t residual2_size = num_layers * batch_size * seq_len * channels;
    static constexpr size_t ln2_size       = num_layers * batch_size * seq_len * channels;
    static constexpr size_t ln2_mean_size  = num_layers * batch_size * seq_len;
    static constexpr size_t ln2_rstd_size  = num_layers * batch_size * seq_len;
    static constexpr size_t fch_size       = num_layers * batch_size * seq_len * 4 * channels;
    // XXX if constexpr no_recompute then * numlayers
    static constexpr size_t fch_gelu_size  = batch_size * seq_len * 4 * channels;
    static constexpr size_t fcproj_size    = num_layers * batch_size * seq_len * channels;
    static constexpr size_t residual3_size = num_layers * batch_size * seq_len * channels;
    static constexpr size_t lnf_size       = batch_size * seq_len * channels;
    static constexpr size_t lnf_mean_size  = batch_size * seq_len;
    static constexpr size_t lnf_rstd_size  = batch_size * seq_len;
    static constexpr size_t losses_size    = batch_size * seq_len;
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    static constexpr size_t qkvr_size      = num_layers * batch_size * seq_len * 3 * channels;
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (batch_size, seq_len, 3C),
    // (batch_size, num_heads, seq_len, seq_len), and (batch_size, seq_len, V) shaped tensors.
    static constexpr size_t output_size    = batch_size * seq_len * std::max(3 * channels, std::max(num_heads * seq_len, vocab_size_p));

public:
    static constexpr size_t num_activations = encoded_size + ln1_size + ln1_mean_size + \
                                              ln1_rstd_size + atty_size + att_size + \
                                              attproj_size + residual2_size + ln2_size + \
                                              ln2_mean_size + ln2_rstd_size + fch_size + \
                                              fch_gelu_size + fcproj_size + residual3_size + \
                                              lnf_size + lnf_mean_size + lnf_rstd_size + \
                                              losses_size + qkvr_size + output_size;
    A* encoded;     // (batch_size, seq_len,    channels)
    A* ln1;         // (num_layers, batch_size, seq_len,   channels)
    A* ln1_mean;    // (num_layers, batch_size, seq_len)
    A* ln1_rstd;    // (num_layers, batch_size, seq_len)
    A* atty;        // (num_layers, batch_size, seq_len,   channels)
    A* att;         // (num_layers, batch_size, num_heads, seq_len, seq_len)
    A* attproj;     // (num_layers, batch_size, seq_len,   channels)
    A* residual2;   // (num_layers, batch_size, seq_len,   channels)
    A* ln2;         // (num_layers, batch_size, seq_len,   channels)
    A* ln2_mean;    // (num_layers, batch_size, seq_len)
    A* ln2_rstd;    // (num_layers, batch_size, seq_len)
    A* fch;         // (num_layers, batch_size, seq_len,   4*channels)
    A* fch_gelu;    // (num_layers, batch_size, seq_len,   4*channels)
    A* fcproj;      // (num_layers, batch_size, seq_len,   channels)
    A* residual3;   // (num_layers, batch_size, seq_len,   channels)
    A* lnf;         // (batch_size, seq_len,    channels)
    A* lnf_mean;    // (batch_size, seq_len)
    A* lnf_rstd;    // (batch_size, seq_len)
    A* losses;      // (batch_size, seq_len)
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    A* qkvr;        // (num_layers, batch_size, seq_len, 3*channels)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (batch_size, seq_len, 3C),
    // (batch_size, num_heads, seq_len, seq_len), and (batch_size, seq_len, V) shaped tensors.
    A* output;

    Activations(A* base_ptr) : encoded(base_ptr),
                               ln1(encoded       + encoded_size),
                               ln1_mean(ln1      + ln1_size),
                               ln1_rstd(ln1_mean + ln1_mean_size),
                               atty(ln1_rstd     + ln1_rstd_size),
                               att(atty          + atty_size),
                               attproj(att       + att_size),
                               residual2(attproj + attproj_size),
                               ln2(residual2     + residual2_size),
                               ln2_mean(ln2      + ln2_size),
                               ln2_rstd(ln2_mean + ln2_mean_size),
                               fch(ln2_rstd      + ln2_rstd_size),
                               fch_gelu(fch      + fch_size),
                               fcproj(fch_gelu   + fch_gelu_size),
                               residual3(fcproj  + fcproj_size),
                               lnf(residual3     + residual3_size),
                               lnf_mean(lnf      + lnf_size),
                               lnf_rstd(lnf_mean + lnf_mean_size),
                               losses(lnf_rstd   + lnf_rstd_size),
                               qkvr(losses       + losses_size),
                               output(qkvr       + qkvr_size)      { }
};

template<typename PType,
         typename TType,                    // Input type  (e.g. int32_t)
         typename AType         = PType,    // Activations type
         size_t   max_seq_len   = 1024,     // max sequence length
         size_t   vocab_size    = 50257,    // vocab size
         size_t   num_layers    = 12,       // number of layers
         size_t   num_heads     = 12,       // number of heads in attention
         size_t   channels      = 768,      // embedding size (768/1024/1280/1600)
         size_t   vocab_size_p  = 50304,    // vocab size padded to e.g. %128==0, 50304
         size_t   batch_size    = 4,        // batch size
         size_t   heads         = 12        // number of heads
         >
class Model : public llm::Model<TType> {
    // Hyperparameter positions in file header
    static constexpr size_t MAX_SEQ_LEN_H        = 2;
    static constexpr size_t VOCAB_SIZE_H         = 3;
    static constexpr size_t NUM_LAYERS_H         = 4;
    static constexpr size_t NUM_HEADS_H          = 5;
    static constexpr size_t CHANNELS_H           = 6;
    static constexpr size_t PADDED_VOCAB_SIZE_H  = 7;
    static constexpr size_t HEADER_SIZE_B        = 256 * sizeof(uint32_t); // total header size (?)
    static constexpr size_t HEADER_ELEMS         = 8;                      // elements we want
    static constexpr size_t MAGIC_NUMBER         = 20240326;

protected:
    using Params = Parameters<PType, max_seq_len, num_layers, channels, vocab_size_p>;
    CudaMemory<PType> params_device;
    Params params;
    static constexpr size_t num_params = Params::num_params;

    CudaMemory<AType> activations_device;
    using Activs = Activations<AType, max_seq_len, num_layers, num_heads, channels, vocab_size_p, batch_size>;
    Activs activations;
    static constexpr size_t num_activations = Activs::num_activations;

    CudaMemory<TType> inputs_device;
    CudaMemory<TType> targets_device;

public:
    // Load model from checkpoint file
    Model(const std::string filename) : params_device(num_params, "params_device"),
                                        params(params_device.get()),
                                        activations_device(num_activations, "activations_device"),
                                        activations(activations_device.get()),
                                        inputs_device(batch_size),
                                        targets_device(batch_size)
    {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            std::cerr << std::format("Model: Failed to open model file: {}\n", filename);
            std::exit(EXIT_FAILURE);
        }

        // Read the header (hyperparameters)
        std::array<int, HEADER_ELEMS> header;
        if (!file.read(reinterpret_cast<char *>(header.data()), header.size() * sizeof(header[0]))) {
            std::cerr << std::format("Model: failed to read header: {}\n", filename);
            std::exit(EXIT_FAILURE);
        }

        if (header[0] != MAGIC_NUMBER) {
            std::cerr << std::format("Model: file {} not valid magic {}\n", filename, header[0]);
        }
        uint32_t version    = header[1];
        assert(version >= 2 && "Model: file version < 2");

        if (!(version == 3 || version == 5)) {
            // 3 = fp32, padded vocab
            // 5 = bf16, padded vocab, layernorms also in bf16
            std::cerr << std::format("Model: file {} version {} not supported (3, 5)\n"
                                     "    HINT: try to re-run llm.c `python train_gpt2.py`\n",
                                     filename, version);
            exit(EXIT_FAILURE);
        }
        if (std::is_same<PType, __half>::value && version != 5) {
            std::cerr << std::format("Model: precision BF16 but file {} version {} not 5\n"
                                     "    HINT: are you sure you're loading a _bf16.bin file?\n",
                                     filename, version);
            exit(EXIT_FAILURE);
        }
        if (std::is_same<PType, float>::value && version != 3) {
            std::cerr << std::format("Model: precision FP32 but file {} version {} not 3\n"
                    "    HINT: to turn on FP32 you have to compile like: `make train_gpt2cu PRECISION=FP32`\n"
                    "    HINT: are you sure you're loading a .bin file without any _bf16 in the name?\n",
                    filename, version);
            exit(EXIT_FAILURE);
        }

        assertf((max_seq_len == header[MAX_SEQ_LEN_H]), "File does not match max_seq_len {} != {}",
                max_seq_len, header[MAX_SEQ_LEN_H]);
        assertf(vocab_size   == header[VOCAB_SIZE_H],   "File does not match vocab_size {} == {}",
                vocab_size, header[VOCAB_SIZE_H]);
        assertf(num_layers   == header[NUM_LAYERS_H],   "File does not match num_layers {} == {}",
                num_layers,header[NUM_LAYERS_H]);
        assertf(num_heads    == header[NUM_HEADS_H],    "File does not match num_heads {} != {}",
                num_heads, header[NUM_HEADS_H]);
        assertf(channels     == header[CHANNELS_H],     "File does not match channels {} != {}",
                channels, header[CHANNELS_H]);
        assertf(vocab_size_p == header[PADDED_VOCAB_SIZE_H], "File does not match vocab_size_p {} != {}",
                vocab_size_p, header[PADDED_VOCAB_SIZE_H]);

        file.seekg(HEADER_SIZE_B, std::ios::beg);   // skip header
        std::unique_ptr<PType[]> params_memory_cpu;
        try {
            params_memory_cpu = std::make_unique<PType[]>(Params::num_params);  // CPU params
        } catch (const std::bad_alloc& e) {
            std::cerr << std::format("Model: failed to allocate parameters (CPU)\n");
            exit(EXIT_FAILURE);
        }

        errno = 0;
        file.read(reinterpret_cast<char *>(params_memory_cpu.get()), params_device.size_bytes());
        if (!file.good()) {
            std::cerr << std::format("Model: failed to load parameters from file {}: {}\n",
                    filename, std::strerror(errno));
        }
        // copy params to device
        cuda_check(cudaMemcpy(params_device.get(), params_memory_cpu.get(), params_device.size_bytes(),
                   cudaMemcpyHostToDevice), "Model: copying params to device");
    }

    // Forward pass
    void run(const llm::TokenLoader<typename llm::Model<TType>::TokenType>::Batch batch) override {
        cuda_check(cudaMemcpy(inputs_device.get(), batch.inputs.data(), inputs_device.size_bytes(),
                   cudaMemcpyHostToDevice), "Model: copying inputs to device");
        cuda_check(cudaMemcpy(targets_device.get(), batch.targets.data(), targets_device.size_bytes(),
                   cudaMemcpyHostToDevice), "Model: copying targets to device");
        // XXX continue
    }

    Model(const Model&) = delete;             // Delete the copy constructor
    Model& operator=(const Model&) = delete;  // Delete the copy assignment operator
    Model(Model&&) = default;                 // Allow move constructor
    Model& operator=(Model&&) = default;      // Allow move assignment operator
};

// A model for training. With gradients. Forward all template arguments.
template<typename PType,           // type for params
         typename TType,           // Input type  (e.g. int32_t)
         typename GType = PType,   // type for gradients
         typename... Args>
class TrainableModel final : public Model<PType, TType, Args...>, public llm::TrainableModel<TType> {
protected:
    using BaseModel = Model<PType, TType, Args...>;

    CudaMemory<GType> grads_device;
    BaseModel::Params grads;
public:
    TrainableModel(const std::string filename) : BaseModel(filename),
                                                 grads_device(BaseModel::num_params, "grads_device"),
                                                 grads(grads_device.get()) {}

// static_assert(std::is_same<TType, int8_t>::value); // XXX
// static_assert(std::is_same<TType, typename llm::Model<TType>::TokenType>::value); // XXX
    void run(const llm::TokenLoader<TType>::Batch batch) override {
        return BaseModel::run(std::move(batch));
    }

    virtual void train(const llm::TokenLoader<TType>::Batch) override {
        // XXX continue
    }
    // Backward pass
    // void backward() override;
};

} // namespace gpt2
