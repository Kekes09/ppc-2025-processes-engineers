#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "luchnikov_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace luchnikov_max_val_in_col_of_mat {

class LuchnikovEMaxValInColOfMatFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int n = input_data_;
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

    return (expected == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(LuchnikovEMaxValInColOfMatFuncTests, MaxValInColTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LuchnikovEMaxValInColOfMatMPI, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_max_val_in_col_of_mat),
                                           ppc::util::AddFuncTask<LuchnikovEMaxValInColOfMatSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_max_val_in_col_of_mat));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LuchnikovEMaxValInColOfMatFuncTests::PrintFuncTestName<LuchnikovEMaxValInColOfMatFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixTests, LuchnikovEMaxValInColOfMatFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luchnikov_max_val_in_col_of_mat
