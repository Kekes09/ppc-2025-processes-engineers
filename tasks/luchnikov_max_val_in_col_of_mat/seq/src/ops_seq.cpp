#include "luchnikov_max_val_in_col_of_mat/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "luchnikov_max_val_in_col_of_mat/common/include/common.hpp"
#include "util/include/util.hpp"

namespace luchnikov_max_val_in_col_of_mat {

LuchnikovEMaxValInColOfMatSEQ::LuchnikovEMaxValInColOfMatSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LuchnikovEMaxValInColOfMatSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool LuchnikovEMaxValInColOfMatSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool LuchnikovEMaxValInColOfMatSEQ::RunImpl() {
  int n = GetInput();  // Размер матрицы n x n

  if (n == 0) {
    return false;
  }

  // Генерация матрицы и вычисление максимумов по столбцам
  std::vector<int> max_col(n, std::numeric_limits<int>::min());

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      // Генерация значения элемента матрицы
      int value = (i * n + j) % (n + 1);  // Такая же генерация как в MPI версии
      if (value > max_col[j]) {
        max_col[j] = value;
      }
    }
  }

  // Вычисление суммы максимумов по столбцам
  int sum_max = 0;
  for (int j = 0; j < n; j++) {
    sum_max += max_col[j];
  }

  GetOutput() = sum_max;
  return GetOutput() >= 0;
}

bool LuchnikovEMaxValInColOfMatSEQ::PostProcessingImpl() {
  // Никакой постобработки не требуется
  return GetOutput() > 0;
}

}  // namespace luchnikov_max_val_in_col_of_mat
