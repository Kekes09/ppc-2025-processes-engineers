#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnilkovEMaxValInColOfMatRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int matrix_size = std::get<0>(params);
    std::string test_type = std::get<1>(params);

    // Генерируем тестовую матрицу
    input_data_ = GenerateTestMatrix(matrix_size, test_type);

    // Вычисляем ожидаемый результат
    expected_output_ = CalculateExpectedResult(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;

  InType GenerateTestMatrix(int size, const std::string &test_type) {
    InType matrix(size, std::vector<int>(size));

    if (test_type == "random1") {
      // Детерминированный "случайный" паттерн 1
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = (i * 17 + j * 13) % 100;
        }
      }
    } else if (test_type == "ascending") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = i * size + j + 1;
        }
      }
    } else if (test_type == "descending") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = size * size - (i * size + j);
        }
      }
    } else if (test_type == "constant") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = 42;
        }
      }
    } else if (test_type == "diagonal") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = (i == j) ? 1000 : 1;
        }
      }
    } else if (test_type == "negative") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = -((i * 17 + j * 13) % 100 + 1);
        }
      }
    } else if (test_type == "mixed") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          int val = (i * 17 + j * 13) % 201 - 100;  // От -100 до 100
          matrix[i][j] = val;
        }
      }
    } else if (test_type == "single_max") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = 1;
        }
      }
      // Помещаем максимальное значение в фиксированную позицию
      int max_row = size / 2;
      int max_col = size / 2;
      if (size > 0) {
        matrix[max_row][max_col] = 10000;
      }
    } else if (test_type == "first_col_max") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = j + 1;  // Максимум в последнем столбце
        }
      }
    } else if (test_type == "last_col_max") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = size - j;  // Максимум в первом столбце
        }
      }
    } else if (test_type == "random2") {
      // Другой детерминированный "случайный" паттерн
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = ((i + 1) * (j + 1) * 7) % 150;
        }
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

namespace {

TEST_P(LuchnilkovEMaxValInColOfMatRunFuncTestsProcesses, MaxValInColTest) {
  ExecuteTest(GetParam());
}

// 10 различных тестовых случаев
const std::array<TestType, 10> kTestParam = {std::make_tuple(3, "random1"),       std::make_tuple(5, "ascending"),
                                             std::make_tuple(7, "descending"),    std::make_tuple(4, "constant"),
                                             std::make_tuple(6, "diagonal"),      std::make_tuple(8, "negative"),
                                             std::make_tuple(10, "mixed"),        std::make_tuple(3, "single_max"),
                                             std::make_tuple(5, "first_col_max"), std::make_tuple(7, "last_col_max")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LuchnilkovEMaxValInColOfMatMPI, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat),
                                           ppc::util::AddFuncTask<LuchnilkovEMaxValInColOfMatSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LuchnilkovEMaxValInColOfMatRunFuncTestsProcesses::PrintFuncTestName<
    LuchnilkovEMaxValInColOfMatRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMaxColTests, LuchnilkovEMaxValInColOfMatRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_max_val_in_col_of_mat
