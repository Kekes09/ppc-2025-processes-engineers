#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnilkovEMaxValInColOfMatMPI::LuchnilkovEMaxValInColOfMatMPI(const InType &in) : matrix_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  result_.clear();
}

bool LuchnilkovEMaxValInColOfMatMPI::ValidationImpl() {
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

bool LuchnilkovEMaxValInColOfMatMPI::PreProcessingImpl() {
  const auto &matrix = GetInput();

  if (!matrix.empty()) {
    std::size_t cols = matrix[0].size();
    result_.assign(cols, std::numeric_limits<int>::min());
  }

  return true;
}

bool LuchnilkovEMaxValInColOfMatMPI::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = static_cast<int>(matrix.size());
  int cols = static_cast<int>(matrix[0].size());

  int rows_per_process = rows / size;
  int remainder = rows % size;

  int start_row = rank * rows_per_process + (rank < remainder ? rank : remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);

  if (end_row > rows) {
    end_row = rows;
  }

  std::vector<int> local_max(cols, std::numeric_limits<int>::min());

  for (int i = start_row; i < end_row && i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      local_max[j] = std::max(matrix[i][j], local_max[j]);
    }
  }

  if (rank == 0) {
    result_ = local_max;

    for (int proc = 1; proc < size; ++proc) {
      std::vector<int> other_max(cols);
      MPI_Recv(other_max.data(), cols, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int j = 0; j < cols; ++j) {
        result_[j] = std::max(other_max[j], result_[j]);
      }
    }

    for (int proc = 1; proc < size; ++proc) {
      MPI_Send(result_.data(), cols, MPI_INT, proc, 1, MPI_COMM_WORLD);
    }
  } else {
    MPI_Send(local_max.data(), cols, MPI_INT, 0, 0, MPI_COMM_WORLD);

    result_.resize(cols);
    MPI_Recv(result_.data(), cols, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool LuchnilkovEMaxValInColOfMatMPI::PostProcessingImpl() {
  GetOutput() = result_;
  return !result_.empty();
}

}  // namespace luchnikov_e_max_val_in_col_of_mat
