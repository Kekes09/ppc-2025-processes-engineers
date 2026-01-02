#include <gtest/gtest.h>

#include "luchnikov_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_max_val_in_col_of_mat {

class LuchnikovEMaxValInColOfMatPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Вычисляем ожидаемое значение для размера матрицы 100x100
    int n = kCount_;
    int expected = 0;

    for (int j = 0; j < n; j++) {
      int max_val = std::numeric_limits<int>::min();
      for (int i = 0; i < n; i++) {
        int value = (i * n + j) % (n + 1);
        if (value > max_val) {
          max_val = value;
        }
      }
      expected += max_val;
    }

    return expected == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LuchnikovEMaxValInColOfMatPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuchnikovEMaxValInColOfMatMPI, LuchnikovEMaxValInColOfMatSEQ>(
        PPC_SETTINGS_luchnikov_max_val_in_col_of_mat);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuchnikovEMaxValInColOfMatPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnikovEMaxValInColOfMatPerfTest, kGtestValues, kPerfTestName);

}  // namespace luchnikov_max_val_in_col_of_mat
