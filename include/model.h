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
// LLM base model
//

#pragma once

#include <tokenizer.h>

namespace llm {

template<typename TType>
class Model {
public:
    using TokenType = TType;
    virtual ~Model() = default;
    virtual void run(const TokenLoader<TType>::Batch batch) = 0;
};

template<typename TType>
class TrainableModel : public Model<TType> {
public:
    virtual ~TrainableModel() = default;
    virtual void train(const TokenLoader<TType>::Batch batch) = 0;
};

}  // namespace llm
