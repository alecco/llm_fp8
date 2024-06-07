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
// LLM base model trainer
//
#pragma once

#include <model.h>
#include <tokenizer.h>

namespace llm {

template<class ModelType>
concept DerivedFromBaseModel = std::is_base_of_v<Model<typename ModelType::TokenType>, ModelType>;

template<class TrainableModelType>
concept DerivedFromTrainableModel = std::is_base_of_v<TrainableModel<typename TrainableModelType::TokenType>, TrainableModelType>;

template<class LoaderType>
concept DerivedFromBaseLoader = std::is_base_of_v<TokenLoader<typename LoaderType::TokenType>, LoaderType>;

template<DerivedFromBaseModel ModelType, DerivedFromBaseLoader LoaderType>
void run(ModelType& model, LoaderType& loader, size_t num_epochs) {
    for (size_t i = 0; i < num_epochs; i++) {
        model.run(loader.next_batch());
    }
}

template<DerivedFromTrainableModel TrainableModelType, DerivedFromBaseLoader LoaderType>
void train(TrainableModelType& model, LoaderType& loader, size_t num_epochs) {
    for (size_t i = 0; i < num_epochs; i++) {
        model.train(loader.next_batch());
    }
}

}  // namespace llm
