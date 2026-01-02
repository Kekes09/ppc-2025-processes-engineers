#include "luchnikov_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "luchnikov_max_val_in_col_of_mat/common/include/common.hpp"
#include "util/include/util.hpp"

namespace luchnikov_max_val_in_col_of_mat {

LuchnikovEMaxValInColOfMatMPI::LuchnikovEMaxValInColOfMatMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LuchnikovEMaxValInColOfMatMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool LuchnikovEMaxValInColOfMatMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool LuchnikovEMaxValInColOfMatMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = GetInput();  // Размер матрицы n x n

  if (n == 0) {
    return false;
  }

  // Каждый процесс вычисляет свою часть матрицы
  int rows_per_process = n / size;
  int remainder = n % size;
  int start_row = rank * rows_per_process + std::min(rank, remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
  if (rank == size - 1) {
    end_row = n;
  }

  // Генерация матрицы и вычисление локальных максимумов по столбцам
  std::vector<int> local_max_col(n, std::numeric_limits<int>::min());

  for (int i = start_row; i < end_row; i++) {
    for (int j = 0; j < n; j++) {
      // Генерация значения элемента матрицы
      int value = (i * n + j) % (n + 1);  // Простая генерация значений
      if (value > local_max_col[j]) {
        local_max_col[j] = value;
      }
    }
  }

  // Сбор глобальных максимумов по столбцам на процессе 0
  std::vector<int> global_max_col(n, std::numeric_limits<int>::min());

  if (rank == 0) {
    // Процесс 0 получает данные от всех процессов
    global_max_col = local_max_col;

    for (int proc = 1; proc < size; proc++) {
      std::vector<int> recv_buffer(n);
      MPI_Recv(recv_buffer.data(), n, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int j = 0; j < n; j++) {
        if (recv_buffer[j] > global_max_col[j]) {
          global_max_col[j] = recv_buffer[j];
        }
      }
    }

    // Вычисление суммы максимумов по столбцам
    int sum_max = 0;
    for (int j = 0; j < n; j++) {
      sum_max += global_max_col[j];
    }
    GetOutput() = sum_max;

  } else {
    // Остальные процессы отправляют свои локальные максимумы процессу 0
    MPI_Send(local_max_col.data(), n, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  // Распространение результата на все процессы
  int result = 0;
  if (rank == 0) {
    result = GetOutput();
  }
  MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput() = result;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return GetOutput() >= 0;
}

bool LuchnikovEMaxValInColOfMatMPI::PostProcessingImpl() {
  // Никакой постобработки не требуется
  return GetOutput() > 0;
}

}  // namespace luchnikov_max_val_in_col_of_mat
