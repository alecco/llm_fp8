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

#include <concepts>
#include <model.h>
#include <tokenizer.h>

namespace llm {

// base model
template<class ModelType>
concept DerivedFromModel  = std::is_base_of_v<Model<typename ModelType::TokenType>, ModelType>;

// trainable model
template<class TrainableModelType>
concept DerivedFromTrainableModel = std::is_base_of_v<TrainableModel<typename TrainableModelType::TokenType>, TrainableModelType>;

// data loader
template<class LoaderType>
concept DerivedFromLoader = std::is_base_of_v<TokenLoader<typename LoaderType::TokenType>, LoaderType>;

template<DerivedFromTrainableModel TrainableModelType, DerivedFromLoader LoaderType>
void train(TrainableModelType& model,
        // LoaderType& loader,
        size_t num_epochs) {
    for (size_t i = 0; i < num_epochs; i++) {
        model.train(); // loader.next_batch());
    }
}

}  // namespace llm
