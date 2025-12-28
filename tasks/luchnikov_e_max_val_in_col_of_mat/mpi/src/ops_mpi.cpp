#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "util/include/util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnilkovEMaxValInColOfMatMPI::LuchnilkovEMaxValInColOfMatMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  matrix_ = in;
  result_.clear();
}

bool LuchnilkovEMaxValInColOfMatMPI::ValidationImpl() {
  const auto &matrix = GetInput();

  // Проверяем, что матрица не пустая
  if (matrix.empty()) {
    return false;
  }

  // Проверяем, что все строки имеют одинаковую длину
  size_t cols = matrix[0].size();
  for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;
    }
  }

  // Проверяем, что выход еще не вычислен
  return GetOutput().empty();
}

bool LuchnilkovEMaxValInColOfMatMPI::PreProcessingImpl() {
  const auto &matrix = GetInput();

  // Инициализируем результат минимальными значениями
  if (!matrix.empty()) {
    size_t cols = matrix[0].size();
    result_.assign(cols, std::numeric_limits<int>::min());
  }

  return true;
}

bool LuchnilkovEMaxValInColOfMatMPI::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = static_cast<int>(matrix.size());
  int cols = static_cast<int>(matrix[0].size());

  // Вычисляем количество строк для каждого процесса
  int rows_per_process = rows / size;
  int remainder = rows % size;

  int start_row = rank * rows_per_process + (rank < remainder ? rank : remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);

  // Локальные максимумы для каждого столбца
  std::vector<int> local_max(cols, std::numeric_limits<int>::min());

  // Каждый процесс обрабатывает свои строки
  for (int i = start_row; i < end_row && i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (matrix[i][j] > local_max[j]) {
        local_max[j] = matrix[i][j];
      }
    }
  }

  // Редукция для нахождения глобальных максимумов по столбцам
  if (rank == 0) {
    result_ = local_max;

    // Получаем результаты от других процессов
    for (int proc = 1; proc < size; ++proc) {
      std::vector<int> other_max(cols);
      MPI_Recv(other_max.data(), cols, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Объединяем результаты
      for (int j = 0; j < cols; ++j) {
        if (other_max[j] > result_[j]) {
          result_[j] = other_max[j];
        }
      }
    }
  } else {
    // Отправляем свои результаты процессу 0
    MPI_Send(local_max.data(), cols, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool LuchnilkovEMaxValInColOfMatMPI::PostProcessingImpl() {
  GetOutput() = result_;
  return !result_.empty();
}

}  // namespace luchnikov_e_max_val_in_col_of_mat
