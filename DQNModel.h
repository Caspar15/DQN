#ifndef DQN_MODEL_H
#define DQN_MODEL_H


#include <torch/torch.h>


// DQN 模型結構：使用多層全連結神經網路進行 Q 值預測
// 這個結構體定義了 DQN 模型，它繼承自 torch::nn::Module，並包含了必要的網絡層和方法
struct DQN : torch::nn::Module
{
  // 定義網路中的全連接層（Linear Layer）
  // 每一層的輸出將作為下一層的輸入
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
  torch::nn::BatchNorm1d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
  torch::nn::Dropout dropout{nullptr}; // 定義一個 Dropout 層，用於防止過度擬合。


  // Dueling DQN 部分
  torch::nn::Linear state_value{nullptr};      // 這部分的狀態價值 (state-value) 會估計每個狀態的價值
  torch::nn::Linear action_advantage{nullptr}; // 這部分的動作優勢 (action-advantage) 會估計不同動作相對於平均狀態的優勢


  // 建構子：初始化每一層神經網絡
  DQN(int input_size, int output_size);


  // 前向傳播函數：定義資料如何在網路中流動
  torch::Tensor forward(torch::Tensor x);


  // 保存模型權重
  void save_weights(const std::string &file_path);


  // 載入模型權重
  void load_weights(const std::string &file_path);
};


#endif // DQN_MODEL_H




