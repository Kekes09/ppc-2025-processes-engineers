#pragma once

#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnilkovEMaxValInColOfMatSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  using InType = std::vector<std::vector<int>>;
  using OutType = std::vector<int>;

  explicit LuchnilkovEMaxValInColOfMatSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace luchnikov_e_max_val_in_col_of_mat
