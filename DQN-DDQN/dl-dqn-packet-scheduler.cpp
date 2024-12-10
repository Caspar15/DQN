#include "dl-dqn-packet-scheduler.h"
#include "DQNTrainer.h"
#include "../mac-entity.h"
#include "../../packet/Packet.h"
#include "../../packet/packet-burst.h"
#include "../../../device/NetworkNode.h"
#include "../../../flows/radio-bearer.h"
#include "../../../protocolStack/rrc/rrc-entity.h"
#include "../../../flows/application/Application.h"
#include "../../../device/GNodeB.h"
#include "../../../protocolStack/mac/AMCModule.h"
#include "../../../phy/phy.h"
#include "../../../core/spectrum/bandwidth-manager.h"

// 帶參數的建構子：初始化 DQN 訓練器
DL_DQN_PacketScheduler::DL_DQN_PacketScheduler(int input_size, int output_size)
    : train_step(0), target_update_frequency(100)
{ // 初始化訓練步數與目標網路更新頻率
    // 初始化 DQNTrainer，傳入輸入、輸出維度、折扣因子、學習率和目標網路更新頻率
    m_dqnTrainer = new DQNTrainer(input_size, output_size, 0.99f, 0.001f, target_update_frequency);
    SetMacEntity(nullptr);   // 初始化 MAC 實體
    CreateFlowsToSchedule(); // 建立調度的流
}

// 預設建構函數，預設輸入和輸出維度為 5
DL_DQN_PacketScheduler::DL_DQN_PacketScheduler()
    : DL_DQN_PacketScheduler(5, 5)
{
}

DL_DQN_PacketScheduler::~DL_DQN_PacketScheduler()
{
    delete m_dqnTrainer; // 釋放記憶體
    Destroy();
}

void DL_DQN_PacketScheduler::DoSchedule()
{
    DEBUG_LOG_START_1(SIM_ENV_SCHEDULER_DEBUG)
    std::cout << "Start DL DQN-based packet scheduler for node "
              << GetMacEntity()->GetDevice()->GetIDNetworkNode() << std::endl;
    DEBUG_LOG_END

    UpdateAverageTransmissionRate();
    CheckForDLDropPackets();
    SelectFlowsToSchedule();
    ComputeWeights(); // 更新權重

    if (GetFlowsToSchedule()->size() == 0)
    {
        // 沒有需要調度的流
    }
    else
    {
        RBsAllocation(); // 資源分配
    }

    StopSchedule();
    ClearFlowsToSchedule();
}

// 計算調度指標，傳回最大 Q 值
double DL_DQN_PacketScheduler::ComputeSchedulingMetric(RadioBearer *bearer, double spectralEfficiency, int subChannel)
{
    torch::Tensor state = GetStateTensor(bearer, spectralEfficiency, subChannel); // 取得目前狀態張量

    // 使用 DQN 的 forward 方法取得 Q 值
    torch::Tensor q_values = m_dqnTrainer->forward(state);

    // 印出 q_values 的形狀和內容以進行偵錯
    std::cout << "Q-values shape: " << q_values.sizes() << std::endl;
    std::cout << "Q-values: " << q_values << std::endl;

    // 取得 Q 值中的最大值和對應的動作索引
    auto max_result = q_values.max(1);
    torch::Tensor max_q_value_tensor = std::get<0>(max_result); // 最大 Q 值
    torch::Tensor max_action_tensor = std::get<1>(max_result);  // 對應的動作索引

    // 傳回最大 Q 值為 metric
    return max_q_value_tensor.cpu().item<double>();
}

// 計算每個流的權重，並更新 DQN 模型
void DL_DQN_PacketScheduler::ComputeWeights()
{
    DEBUG_LOG_START_1(SIM_ENV_SCHEDULER_DEBUG)
    std::cout << "Compute DQN Weights" << std::endl;
    DEBUG_LOG_END

    std::vector<Experience> replay_memory; // 經驗重播緩衝區
    for (auto flow : *GetFlowsToSchedule())
    {
        RadioBearer *bearer = flow->GetBearer();
        if (bearer->HasPackets())
        {
            torch::Tensor state = GetStateTensor(bearer, 0.0, 0);
            torch::Tensor next_state = GetNextStateTensor(bearer);
            int action = m_dqnTrainer->select_action(state); // 選動作
            float reward = ComputeReward(bearer);            // 計算獎勵

            // 將狀態轉移儲存到經驗重播中
            Experience exp = {state, next_state, action, reward};
            replay_memory.push_back(exp);
        }
    }

    // 使用經驗回放資料訓練 DQN，並增加訓練步數
    m_dqnTrainer->train(replay_memory, train_step++);
}

torch::Tensor DL_DQN_PacketScheduler::GetStateTensor(RadioBearer *bearer, double spectralEfficiency, int subChannel)
{
    // 取得狀態值
    float hol_delay = static_cast<float>(bearer->GetHeadOfLinePacketDelay());
    float avg_tx_rate = static_cast<float>(bearer->GetAverageTransmissionRate());
    float spectral_eff = static_cast<float>(spectralEfficiency);
    float sub_channel = static_cast<float>(subChannel);
    float queue_size = static_cast<float>(bearer->GetQueueSize());

    // 取得模型所在設備
    auto device = m_dqnTrainer->get_device();

    // 建立狀態值的向量
    std::vector<float> state_values = {hol_delay, avg_tx_rate, spectral_eff, sub_channel, queue_size};

    // 從向量建立一維張量
    torch::Tensor state_tensor = torch::tensor(state_values, torch::TensorOptions().dtype(torch::kFloat32)).to(device);

    // 調整張量形狀為 [1, 5]
    return state_tensor.view({1, 5});
}

torch::Tensor DL_DQN_PacketScheduler::GetNextStateTensor(RadioBearer *bearer)
{
    // 假設下一狀態中的一些參數會發生變化
    float next_HOL = static_cast<float>(bearer->GetHeadOfLinePacketDelay() - 0.001f);
    float next_rate = static_cast<float>(bearer->GetAverageTransmissionRate() * 1.01f);
    float next_queue_size = static_cast<float>(bearer->GetQueueSize() - 1);

    // 取得模型所在設備
    auto device = m_dqnTrainer->get_device();

    // 建立下一狀態值的向量
    std::vector<float> next_state_values = {next_HOL, next_rate, 0.0f, 0.0f, next_queue_size};

    // 從向量建立一維張量
    torch::Tensor next_state_tensor = torch::tensor(next_state_values, torch::TensorOptions().dtype(torch::kFloat32)).to(device);

    // 調整張量形狀為 [1, 5]
    return next_state_tensor.view({1, 5});
}

float DL_DQN_PacketScheduler::ComputeReward(RadioBearer *bearer)
{
    double HOL = bearer->GetHeadOfLinePacketDelay();
    double rate = bearer->GetAverageTransmissionRate();

    // 獎勵函數：減少延遲，提高吞吐量
    double max_rate = 100000;
    double min_delay = 0.001;

    double delay_reward = (min_delay / (HOL + min_delay));

    double rate_reward = (rate / max_rate);

    double total_reward = 0.5 * delay_reward + 0.5 * rate_reward;

    return total_reward;
}

// 保存模型
void DL_DQN_PacketScheduler::SaveModel(const std::string &model_path)
{
    m_dqnTrainer->save_model(model_path);
}

// 加載模型
void DL_DQN_PacketScheduler::LoadModel(const std::string &model_path)
{
    m_dqnTrainer->load_model(model_path);
}