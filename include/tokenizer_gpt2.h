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
// GPT-2 tokenizer
//

#pragma once

#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <tokenizer.h>

#include <util/assert.h>
#include <util/cuda_check.h>

using util::assertf;
using util::cuda_check;

// TODO: this should run in GPU (concatenating variable sized strings)

namespace gpt2 {

template <std::integral TType>        // internal token type  (e.g. uint32_t)
class Tokenizer final : public llm::Tokenizer<TType> {
private:
    std::unique_ptr<char[]>  buffer;                   // raw strings
    std::vector<std::string_view>    token_table;      // views of the strings in the file
    static constexpr int  HEADER_SIZE_B      = 256 * sizeof(uint32_t); // total header size (?)
    static constexpr int  MAGIC_NUMBER       = 20240328;
    static constexpr int  MAGIC_H            = 0;
    static constexpr int  VERSION_H          = 1;
    static constexpr int  VOCAB_SIZE_H       = 2;
    static constexpr int  EOT_TOKEN_H        = 3;
    static constexpr int  TOKEN_HEADER_ELEMS = 4;                      // elements we want
    uint32_t              eot_token;               // <|endoftext|> token id
public:
    Tokenizer(std::string filename) {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            std::cerr << std::format("Tokenizer: Failed to open token map: {}\n", filename);
            std::exit(EXIT_FAILURE);
        }

        // Read the header
        std::array<int, TOKEN_HEADER_ELEMS> header;
        if (!file.read(reinterpret_cast<char *>(header.data()), header.size() * sizeof(header[0]))) {
            std::cerr << std::format("Failed to read header: {}\n", filename);
            std::exit(EXIT_FAILURE);
        }

        auto magic      = header[MAGIC_H];
        auto version    = header[VERSION_H];
        auto vocab_size = header[VOCAB_SIZE_H];
        eot_token       = header[EOT_TOKEN_H];
        assertf(magic == MAGIC_NUMBER, "Tokenizer: bad magic {} ({})\n", magic, MAGIC_NUMBER);
        assertf(version >= 2, "Tokenizer: file version {} < 2", version);

        // Read the strings into buffer
        file.seekg(0, std::ios::end);
        std::streamsize bsize = static_cast<std::streamoff>(file.tellg()) - HEADER_SIZE_B;

        file.seekg(HEADER_SIZE_B, std::ios::beg);      // skip header
        buffer = std::make_unique<char[]>(bsize);      // alloc string buffer
        if (!file.read(buffer.get(), bsize)) {
            std::cerr << std::format("Tokenizer: failed to read {} bytes: {}\n", bsize, filename);
            exit(EXIT_FAILURE);
        }

        token_table.reserve(vocab_size);
        const char* ptr = buffer.get();
        while (ptr < buffer.get() + bsize) {
            uint8_t len = static_cast<uint8_t>(*ptr++);
            token_table.emplace_back(ptr, len);
            ptr += len;
        }
        if (token_table.size() != vocab_size) {
            std::cerr << std::format("Tokenizer: warning: token table {} != vocabulary size {}\n",
                                     token_table.size(), vocab_size);
        }

        file.close();
        assertf(!token_str(eot_token).empty(), std::format("Tokenizer: eot token {} not valid '{}'",
                                                          eot_token, token_str(eot_token)));
    }

    std::string_view token_str(const TType tok) override {
        assertf(tok < token_table.size(), "Tokenizer: bad token");
        return token_table[tok];
    }

    Tokenizer(const Tokenizer&)            = delete;  // Delete the copy constructor
    Tokenizer& operator=(const Tokenizer&) = delete;  // Delete the copy assignment operator
    Tokenizer(Tokenizer&&)                 = default; // Allow move constructor
    Tokenizer& operator=(Tokenizer&&)      = default; // Allow move assignment operator

    uint32_t vocab_size() {
        return token_table.size();
    }
    uint32_t eot_token_id() {
        return eot_token;
    }
};

template <std::integral FType,        // file token type  (e.g. uint16_t)
          std::integral TType,        // internal token type  (e.g. uint32_t)
          std::integral HType = int>
class TokenLoader final : public llm::TokenLoader<TType> {
    static_assert(std::numeric_limits<FType>::min() >= std::numeric_limits<TType>::min() &&
                  std::numeric_limits<FType>::max() <= std::numeric_limits<TType>::max(),
                  "TokenLoader: values of type FType must fit into type TType");
    static constexpr int  HEADER_SIZE        = 256;
    static constexpr int  HEADER_SIZE_B      = HEADER_SIZE * sizeof(HType);
    static constexpr int  MAGIC_NUMBER       = 20240520;
    static constexpr int  MAGIC_H            = 0;
    static constexpr int  VERSION_H          = 1;
    static constexpr int  NUM_TOKENS_H       = 2;
    static constexpr int  TOKEN_HEADER_ELEMS = 3;                      // elements we want
    std::string         tokens_file_path;
    std::ifstream       file;
    std::vector<FType>  buffer;         // tokens read from file
    std::vector<TType>  aux;            // only used if FType and TType are different
    size_t              batch_size;
    size_t              num_batches;
    size_t              tok_max;
    size_t              seq_len_max;
    size_t              file_size;
Tokenizer<TType>& tokenizer;  // XXX  debug
public:
    size_t              num_tokens;

    TokenLoader(const std::string_view& tokens_file_path_, int batch_size, int seq_len_max,
                Tokenizer<TType>& tokenizer,   // XXX debug
                size_t tok_max = 50258)
            : tokens_file_path(tokens_file_path_), batch_size(batch_size), tok_max(tok_max),
              seq_len_max(seq_len_max)
              , tokenizer(tokenizer)   // XXX  debug
              {

        assertf(tok_max <= std::numeric_limits<FType>::max(), "TokenLoader: max token value {} > {}",
                tok_max, std::numeric_limits<FType>::max());

        file.open(tokens_file_path, std::ifstream::in | std::ifstream::binary);
        assertf(file.is_open(), "Failed to open tokens file: {}\n", tokens_file_path);
        // Read the header and check
        std::array<int, TOKEN_HEADER_ELEMS> header;
        if (!file.read(reinterpret_cast<char *>(header.data()), header.size() * sizeof(header[0]))) {
            std::cerr << std::format("Failed to read header: {}\n", tokens_file_path);
            std::exit(EXIT_FAILURE);
        }
        auto magic      = header[MAGIC_H];
        auto version    = header[VERSION_H];
        num_tokens      = header[NUM_TOKENS_H];
        assertf(magic == MAGIC_NUMBER, "Tokenizer: bad magic {} ({})\n", magic, MAGIC_NUMBER);
        assertf(version == 1, "Tokenizer: file version {} != 1", version);
        assertf(num_tokens > 0, "Tokenizer: empty {}", num_tokens);

        file.seekg(0, std::ios::end);
        std::streamsize actual_file_size = file.tellg();
        file_size = HEADER_SIZE_B + num_tokens * sizeof(FType);   // expected
        assertf(actual_file_size == file_size,
                "TokenLoader: Error: file size is too small for the batch size and sequence length");
        num_batches = actual_file_size / (batch_size * seq_len_max * sizeof(FType));
#if 0
std::cerr << std::format("XXX size FType {} size TType {} size HType {}, HEADER_SIZE_B {} (expected {})\n", sizeof(FType), sizeof(TType), sizeof(HType), HEADER_SIZE_B, 256 * sizeof(int)); // XXX
uint16_t x;
file.seekg(256 * sizeof(int), std::ios::beg);
file.read(reinterpret_cast<char *>(&x), sizeof(x));
std::cerr << tokens_file_path << std::endl;
std::cerr << std::format("XXX A read {}\n", x); // XXX
file.seekg(HEADER_SIZE_B, std::ios::beg);
file.read(reinterpret_cast<char *>(&x), sizeof(x));
std::cerr << std::format("XXX B read {}\n", x); // XXX
#endif

        file.seekg(HEADER_SIZE_B, std::ios::beg);

        buffer.resize(buffer_len());
        if constexpr (!std::is_same_v<FType, TType>) {
            aux.resize(buffer_len());
        }
    }

    TokenLoader(const TokenLoader&) = delete;             // Delete the copy constructor
    TokenLoader& operator=(const TokenLoader&) = delete;  // Delete the copy assignment operator
    TokenLoader(TokenLoader&&) = default;                 // Allow move constructor
    TokenLoader& operator=(TokenLoader&&) = default;      // Allow move assignment operator
    ~TokenLoader() {
        file.close();
    }

    inline size_t buffer_len() const {
        return batch_size * seq_len_max + 1;
    }
    // buffer length in bytes
    inline size_t buffer_len_b() const {
        return buffer_len() * sizeof(FType);
    }

    void reset() {
        file.clear();
        file.seekg(HEADER_SIZE_B, std::ios::beg);
    }

    // If we are at the end of the file, loop back to the beginning
    void check_remaining_and_loop() {
        size_t remaining_size = file_size - file.tellg();
        if (remaining_size < buffer_len_b()) {
            reset();
        }
    }

    // read batch_size * seq_len_max + 1 FType tokens, create input/target spans for inputs and targets
    llm::TokenLoader<TType>::Batch next_batch() {
        assertf(file.is_open(), "Failed tokens file not readable: {}\n", tokens_file_path);

        check_remaining_and_loop();
        file.read(reinterpret_cast<char*>(buffer.data()), buffer_len_b());
        // check tokens are valid before returning
        for (size_t i = 0; i < batch_size; i++) {
            FType t = buffer[i];
            if (std::is_unsigned<FType>::value ? (t > tok_max) : (t < 0 && t > tok_max)) {
                std::cerr << std::format("Bad token {} at {}:{}  eot {}\n",
                        t, 
                        // tokenizer.token_str(t),
                        tokens_file_path, i
                        , tokenizer.eot_token_id()
                        );
                std::exit(EXIT_FAILURE);
            }
            if constexpr (std::is_same_v<FType, TType>) {
                aux[i]  = static_cast<TType>(t);    // convert  (e.g. uint16_t to int32_t)
            }
        }
        if constexpr (std::is_same_v<FType, TType>) {
            return {
                std::span<TType>(buffer.data(),      buffer_len() - 2),  // skip last
                std::span<TType>(buffer.data(), + 1, buffer_len() - 2)   // skip first and last
            };
        } else {
            return {
                std::span<TType>(aux.data(),         buffer_len() - 2),  // skip last
                std::span<TType>(aux.data()     + 1, buffer_len() - 2)   // skip first and last
            };
        }
    }

};

} // namespace gpt2
