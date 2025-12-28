#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

using InType = std::vector<std::vector<int>>;  // Вход: матрица
using OutType = std::vector<int>;              // Выход: вектор максимальных значений по столбцам
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace luchnikov_e_max_val_in_col_of_mat
