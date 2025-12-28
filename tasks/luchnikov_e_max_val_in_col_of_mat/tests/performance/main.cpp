#include <gtest/gtest.h>

#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnilkovEMaxValInColOfMatRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 100;
  InType input_data_{};
  OutType expected_output_{};

  void SetUp() override {
    // Генерируем большую матрицу для теста производительности
    input_data_ = GenerateLargeMatrix(kMatrixSize_);
    expected_output_ = CalculateExpectedResult(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType GenerateLargeMatrix(int size) {
    InType matrix(size, std::vector<int>(size));

    // Детерминированная генерация без случайных чисел
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        // Используем детерминированную формулу на основе индексов
        matrix[i][j] = (i * 19 + j * 23) % 10000 + 1;  // От 1 до 10000
      }
    }

    return matrix;
  }

  OutType CalculateExpectedResult(const InType &matrix) {
    if (matrix.empty()) {
      return {};
    }

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    OutType result(cols, std::numeric_limits<int>::min());

    for (size_t j = 0; j < cols; ++j) {
      for (size_t i = 0; i < rows; ++i) {
        if (matrix[i][j] > result[j]) {
          result[j] = matrix[i][j];
        }
      }
    }

    return result;
  }
};

TEST_P(LuchnilkovEMaxValInColOfMatRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuchnilkovEMaxValInColOfMatMPI, LuchnilkovEMaxValInColOfMatSEQ>(
        PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuchnilkovEMaxValInColOfMatRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnilkovEMaxValInColOfMatRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace luchnikov_e_max_val_in_col_of_mat
