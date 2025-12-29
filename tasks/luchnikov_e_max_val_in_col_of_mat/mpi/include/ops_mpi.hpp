#pragma once

#include <utility>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnilkovEMaxValInColOfMatMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  using InType = std::vector<std::vector<int>>;
  using OutType = std::vector<int>;
  explicit LuchnilkovEMaxValInColOfMatMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::pair<int, int> GetMatrixDimensions(int rank);
  void DistributeMatrixData(int rank, int rows, int cols, std::vector<std::vector<int>> &local_matrix);
  static std::pair<int, int> CalculateColumnDistribution(int rank, int size, int cols);
  static std::vector<int> ComputeLocalMaxima(const std::vector<std::vector<int>> &matrix, int rows, int start_col,
                                             int col_count);
  static OutType GatherResults(const std::vector<int> &local_maxima, int size, int cols);
};

}  // namespace luchnikov_e_max_val_in_col_of_mat
