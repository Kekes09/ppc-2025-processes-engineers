#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnilkovEMaxValInColOfMatSEQ::LuchnilkovEMaxValInColOfMatSEQ(const InType &in) : matrix_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  result_.clear();
}

bool LuchnilkovEMaxValInColOfMatSEQ::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  std::size_t cols = matrix[0].size();
  for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool LuchnilkovEMaxValInColOfMatSEQ::PreProcessingImpl() {
  const auto &matrix = GetInput();

  if (!matrix.empty()) {
    std::size_t cols = matrix[0].size();
    result_.assign(cols, std::numeric_limits<int>::min());
  }

  return true;
}

bool LuchnilkovEMaxValInColOfMatSEQ::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  std::size_t rows = matrix.size();
  std::size_t cols = matrix[0].size();

  for (std::size_t j = 0; j < cols; ++j) {
    int max_val = std::numeric_limits<int>::min();
    for (std::size_t i = 0; i < rows; ++i) {
      max_val = std::max(matrix[i][j], max_val);
    }
    result_[j] = max_val;
  }

  return true;
}

bool LuchnilkovEMaxValInColOfMatSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return !result_.empty();
}

}  // namespace luchnikov_e_max_val_in_col_of_mat
