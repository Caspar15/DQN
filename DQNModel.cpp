#include "DQNModel.h"

// DQN 類別的構造函數，用來初始化神經網絡的結構。
// input_size: 代表神經網絡輸入的大小，通常與輸入特徵的數量一致（例如，在 5G 模擬中，可能是與狀態特徵相關的數量）。
// output_size: 代表神經網絡輸出的大小，通常與動作空間的大小一致，意味著有多少個可能的動作可以選擇。
DQN::DQN(int input_size, int output_size)
{
   // 定義第一層全連接層 (input_size -> 512)，並註冊此層
   fc1 = register_module("fc1", torch::nn::Linear(input_size, 512));
   // 定義第一層的批次正規化，保持數值穩定，並註冊此層
   bn1 = register_module("bn1", torch::nn::BatchNorm1d(512));

   // 定義第二層全連接層 (512 -> 256) 與正規化
   fc2 = register_module("fc2", torch::nn::Linear(512, 256));
   bn2 = register_module("bn2", torch::nn::BatchNorm1d(256));

   // 定義第三層全連接層 (256 -> 128) 與正規化
   fc3 = register_module("fc3", torch::nn::Linear(256, 128));
   bn3 = register_module("bn3", torch::nn::BatchNorm1d(128));

   // 定義第四層全連接層 (128 -> 64)
   fc4 = register_module("fc4", torch::nn::Linear(128, 64));

   // 定義 Dropout 層，用於防止過度擬合
   dropout = register_module("dropout", torch::nn::Dropout(0.1)); // 丟棄 10% 的神經元輸出

   // 定義 Dueling DQN 的狀態價值輸出層
   state_value = register_module("state_value", torch::nn::Linear(64, 1));                     // 狀態價值
   // 定義 Dueling DQN 的動作優勢輸出層
   action_advantage = register_module("action_advantage", torch::nn::Linear(64, output_size)); // 動作優勢
}

// 前向傳播函數
// x: 輸入的特徵張量，形狀應為 [batch_size, input_size]
torch::Tensor DQN::forward(torch::Tensor x)
{
   // 第一層前向傳播，使用 Leaky ReLU 激活函數
   x = torch::leaky_relu(bn1->forward(fc1->forward(x)), 0.01);
   // 在第一層之後套用 Dropout，防止過度擬合
   x = dropout->forward(x);

   // 第二層前向傳播，使用 Leaky ReLU
   x = torch::leaky_relu(bn2->forward(fc2->forward(x)), 0.01);

   // 第三層前向傳播，使用 Leaky ReLU
   x = torch::leaky_relu(bn3->forward(fc3->forward(x)), 0.01);

   // 第四層前向傳播
   x = torch::leaky_relu(fc4->forward(x), 0.01);

   // Dueling DQN 部分
   // 計算狀態價值 (value)，輸出形狀為 [batch_size, 1]
   torch::Tensor value = state_value->forward(x);
   // 計算動作優勢 (advantage)，輸出形狀為 [batch_size, output_size]
   torch::Tensor advantage = action_advantage->forward(x);

   // 結合狀態價值與動作優勢，計算 Q 值
   // 注意：advantage - advantage.mean() 的目的是將優勢值去中心化，避免重複計算動作偏差
   return value + (advantage - advantage.mean());
}

// 保存模型權重到指定文件
// file_path: 欲保存權重的文件路徑
void DQN::save_weights(const std::string &file_path)
{
   // 使用 Torch 提供的序列化工具保存模型
   torch::serialize::OutputArchive archive;
   this->save(archive);
   archive.save_to(file_path); // 保存到指定路徑
}

// 從指定文件載入模型權重
// file_path: 欲載入權重的文件路徑
void DQN::load_weights(const std::string &file_path)
{
   // 使用 Torch 提供的序列化工具載入模型
   torch::serialize::InputArchive archive;
   archive.load_from(file_path); // 從指定路徑載入
   this->load(archive);
}
