#include "urin_o_max_val_in_col_of_mat/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace urin_o_max_val_in_col_of_mat {

UrinOMaxValInColOfMatSeq::UrinOMaxValInColOfMatSeq(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  if (!in.empty()) {
    GetInput() = in;
  } else {
    GetInput() = InType();
    GetOutput() = OutType{};
  }

  bool UrinOMaxValInColOfMatSeq::ValidationImpl() {
    const auto &matrix = GetInput();

    if (matrix.empty() || matrix[0].empty()) {
      return false;
    }

    size_t cols = matrix[0].size();
    return std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; });
  }

  bool UrinOMaxValInColOfMatSeq::PreProcessingImpl() {
    return true;
  }

  bool UrinOMaxValInColOfMatSeq::RunImpl() {
    const auto &matrix = GetInput();

    if (matrix.empty() || matrix[0].empty()) {
      GetOutput() = OutType();
      return false;
    }

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    OutType column_maxima(cols);

    for (size_t col = 0; col < cols; ++col) {
      int max_val = matrix[0][col];

      for (size_t row = 1; row < rows; ++row) {
        max_val = std::max(matrix[row][col], max_val);
      }

      column_maxima[col] = max_val;
    }

    GetOutput() = column_maxima;
    return true;
  }

  bool UrinOMaxValInColOfMatSeq::PostProcessingImpl() {
    return !GetOutput().empty();
  }

}  // namespace urin_o_max_val_in_col_of_mat
