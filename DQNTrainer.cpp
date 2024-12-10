#include "DQNTrainer.h"

// DQNTrainer 構造函數
// 初始化 DQN 模型、目標網路、最佳化器以及相關的超參數
DQNTrainer::DQNTrainer(int input_size, int output_size, float gamma, float lr, int target_update_freq, int replay_memory_size, int batch_size)
    : model(input_size, output_size), // 初始化主網路
      optimizer(model.parameters(), torch::optim::AdamOptions(lr).weight_decay(1e-5)), // 使用 Adam 優化器並設定權重衰減
      gamma(gamma),                             // 折扣因子
      target_model(input_size, output_size),    // 初始化目標網路
      target_update_frequency(target_update_freq), // 目標網路更新頻率
      replay_memory_size(replay_memory_size),   // 記憶回放緩衝區大小
      batch_size(batch_size)                    // 每次訓練的批量大小
{
    epsilon = 1.0;         // 初始化探索率為 1.0，完全隨機探索
    epsilon_decay = 0.995; // 設定探索率的衰減因子，每次訓練後減少
    epsilon_min = 0.01;    // 設定探索率的最小值，避免過度利用

    // 將模型移動到 GPU
    if (torch::cuda::is_available())
    {
        model.to(torch::kCUDA);        // 將主網路移至 CUDA 設備
        target_model.to(torch::kCUDA); // 將目標網路移至 CUDA 設備
        std::cout << "Using CUDA" << std::endl;
        std::cout << "CUDA Device Count: " << torch::cuda::device_count() << std::endl;
    }
    else
    {
        std::cout << "Using CPU" << std::endl;
    }

    // 初始化目標網路的參數
    update_target_model();

    // 設定學習率調度器：每訓練 100 個步驟時學習率乘以 0.1
    scheduler = std::make_shared<torch::optim::StepLR>(optimizer, /*step_size=*/100, /*gamma=*/0.1);
}

// 訓練函數，基於 Double DQN 算法進行
void DQNTrainer::train(std::vector<Experience> &replay_memory, int train_step)
{
    // 如果回放緩衝區小於批量大小，則跳過訓練
    if (replay_memory.empty() || replay_memory.size() < batch_size)
    {
        return;
    }

    model.train(); // 設置模型為訓練模式

    // 隨機抽樣一批數據
    int memory_size = std::min(static_cast<int>(replay_memory.size()), replay_memory_size);
    std::random_shuffle(replay_memory.begin(), replay_memory.end());
    std::vector<Experience> batch(replay_memory.begin(), replay_memory.begin() + batch_size);

    for (auto &experience : batch)
    {
        // 確保所有張量位於同一設備（CPU/GPU）
        auto device = model.parameters().begin()->device();
        auto state = experience.state.to(device);                                                 // 當前狀態
        auto next_state = experience.next_state.to(device);                                       // 下一狀態
        auto action = torch::tensor({experience.action}, torch::dtype(torch::kLong)).to(device); // 動作
        auto reward = torch::tensor({experience.reward}).to(device);                             // 獎勵

        // 計算當前狀態的 Q 值
        auto q_values = model.forward(state);
        auto q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1); // 根據動作選取對應的 Q 值

        // 計算下一狀態的最大 Q 值 (Double DQN)
        auto next_q_values_online = model.forward(next_state);
        auto next_action_online = std::get<1>(torch::max(next_q_values_online, 1)); // 主網路選擇動作
        auto next_q_values_target = target_model.forward(next_state);
        auto max_next_q_value = next_q_values_target.gather(1, next_action_online.unsqueeze(1)).squeeze(1);

        // 計算目標 Q 值
        auto gamma_tensor = torch::tensor({gamma}).to(device);
        auto target = reward + gamma_tensor * max_next_q_value;

        // 計算損失
        auto loss = torch::mse_loss(q_value, target.detach());

        // 反向傳播與參數更新
        optimizer.zero_grad(); // 清除梯度
        loss.backward();       // 計算梯度
        optimizer.step();      // 更新模型參數
    }

    // 更新 epsilon（探索率）
    update_epsilon();

    // 如果達到目標網路更新頻率，則更新目標網路
    if (train_step % target_update_frequency == 0)
    {
        update_target_model();
    }

    // 更新學習率
    scheduler->step();
}

// 保存模型
void DQNTrainer::save_model(const std::string &model_path)
{
    torch::serialize::OutputArchive archive;
    model.save(archive);
    archive.save_to(model_path);
}

// 載入模型
void DQNTrainer::load_model(const std::string &model_path)
{
    torch::serialize::InputArchive archive;
    archive.load_from(model_path);
    model.load(archive);
    target_model.load(archive);
}

// 選擇動作，使用 epsilon-greedy 策略
int DQNTrainer::select_action(torch::Tensor state)
{
    model.eval(); // 設置模型為評估模式
    auto device = model.parameters().begin()->device();
    state = state.to(device);

    if (torch::rand(1).item<float>() < epsilon)
    {
        // 探索：隨機選擇動作
        return torch::randint(0, 5, {1}, torch::TensorOptions().device(device)).item<int>();
    }
    else
    {
        // 利用：選擇預測的最佳動作
        auto q_values = model.forward(state);
        return std::get<1>(torch::max(q_values, 1)).item<int>();
    }
}

// 更新 epsilon 值
void DQNTrainer::update_epsilon()
{
    if (epsilon > epsilon_min)
    {
        epsilon *= epsilon_decay;
    }
}

// 更新目標網路
void DQNTrainer::update_target_model()
{
    torch::NoGradGuard no_grad; // 禁用梯度計算
    auto model_params = model.named_parameters();
    auto target_params = target_model.named_parameters();
    for (const auto &item : model_params)
    {
        auto name = item.key();
        auto *target_param = target_params.find(name);
        if (target_param != nullptr)
        {
            target_param->copy_(item.value());
        }
    }
}

// 前向傳播，用於評估
torch::Tensor DQNTrainer::forward(torch::Tensor state)
{
    auto device = model.parameters().begin()->device();
    return model.forward(state.to(device));
}

// 取得模型所在設備
torch::Device DQNTrainer::get_device()
{
    return model.parameters().begin()->device();
}
