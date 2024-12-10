#ifndef DQN_TRAINER_H
#define DQN_TRAINER_H


#include <torch/torch.h> // 引進 PyTorch 的核心庫
#include "DQNModel.h"    // 引入 DQN 模型
#include <vector>        // 用於儲存經驗回放


// 定義經驗回放資料結構
// Experience 結構用於儲存訓練過程中的狀態轉移 (s, a, r, s')，每次訓練都從此結構中提取樣本
struct Experience
{
  torch::Tensor state;      // 目前狀態
  torch::Tensor next_state; // 下一個狀態
  int action;               // 動作
  float reward;             // 獎勵
};


// 定義 DQN 訓練器類
class DQNTrainer
{
public:
  DQNTrainer(int input_size, int output_size, float gamma, float lr, int target_update_freq, int replay_memory_size, int batch_size);


  // 訓練函數：用於根據經驗回放資料訓練模型
  // replay_memory：經驗重播緩衝區
  // train_step：目前訓練步數
  void train(std::vector<Experience> &replay_memory, int train_step);


  // 動作選擇函數：根據 epsilon-greedy 策略選擇動作
  // state：目前狀態張量
  int select_action(torch::Tensor state);


  // 前向傳播函數：傳遞狀態通過網絡，返回 Q 值
  torch::Tensor forward(torch::Tensor state);


  // 更新 epsilon 值，逐步減少探索率
  void update_epsilon();


  // save model and load model
  void save_model(const std::string &model_path);
  void load_model(const std::string &model_path);


  // 取得裝置資訊：返回模型所在的裝置（GPU）
  torch::Device get_device();


private:
  DQN model;                                       // 主網路模型，用於實際訓練
  DQN target_model;                                // 目標網路模型，用於 Double DQN 中的目標 Q 值計算
  torch::optim::Adam optimizer;                    // Adam 優化器，用於更新主網路的權重
  std::shared_ptr<torch::optim::StepLR> scheduler; // 學習率調度器，用於動態調整學習率


  float gamma;                 // 折扣因子，決定未來獎勵的影響程度
  float epsilon;               // 目前探索率，影響 epsilon-greedy 策略的選擇
  float epsilon_decay;         // 探索率衰減係數，每次訓練後減少 epsilon
  float epsilon_min;           // 最小探索率，防止探索率過低
  int target_update_frequency; // 目標網路的更新頻率，每隔多少步驟更新一次目標網絡
  int replay_memory_size;
  int batch_size;


  // 更新目標網路的權重，使其與主網路保持同步
  void update_target_model();
};


#endif // DQN_TRAINER_H




