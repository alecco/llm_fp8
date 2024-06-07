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
// LLM base tokenizer
//
#pragma once

#include <span>
#include <string_view>

namespace llm {

template<std::integral TType>
class Tokenizer {
    virtual std::string_view token_str(const TType tok) = 0;
};

template<std::integral TType>
class TokenLoader {
public:
    using TokenType = TType;
    struct Batch {
        std::span<TType>    inputs;         // first batch_size * seq_len_max items
        std::span<TType>    targets;        // Target in GPT is next token, so inputs + 1
    };
};

}  // namespace llm
