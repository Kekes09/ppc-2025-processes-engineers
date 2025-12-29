#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnilkovEMaxValInColOfMatPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int matrix_size = 50;
    test_matrix_.resize(matrix_size, std::vector<int>(matrix_size));

    for (int i = 0; i < matrix_size; ++i) {
      for (int j = 0; j < matrix_size; ++j) {
        test_matrix_[i][j] = ((i + j) % 100) + 1;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty() && output_data.size() == test_matrix_[0].size();
  }

  InType GetTestInputData() final {
    return test_matrix_;
  }

 public:
  static std::string CustomPerfTestName(const testing::TestParamInfo<BaseRunPerfTests::ParamType> &info) {
    return "PerfTest_" + std::to_string(info.index);
  }

 private:
  std::vector<std::vector<int>> test_matrix_;
};

TEST_P(LuchnilkovEMaxValInColOfMatPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuchnilkovEMaxValInColOfMatMPI, LuchnilkovEMaxValInColOfMatSEQ>(
        PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuchnilkovEMaxValInColOfMatPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnilkovEMaxValInColOfMatPerfTests, kGtestValues, kPerfTestName);

}  // namespace luchnikov_e_max_val_in_col_of_mat
