import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import flappy_bird_gymnasium
from collections import deque
import matplotlib.pyplot as plt 

# SpikingJelly 核心组件
from spikingjelly.activation_based import neuron, layer, functional, surrogate, encoding

# ===============================
# 1. 修复的脉冲残差模块
# ===============================
class MS_ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = layer.Linear(dim, dim)
        self.bn = nn.LayerNorm(dim)
        self.lif = neuron.LIFNode(
            surrogate_function=surrogate.ATan(), 
            detach_reset=True, 
            step_mode='m'
        )

    def forward(self, x_seq):
        identity = x_seq
        out = self.conv(x_seq)
        out = self.bn(out)
        out = self.lif(out + identity)
        return out

# ===============================
# 2. 优化的 ReSNN-DQN 网络 (关键修复)
# ===============================
class SNNQNet(nn.Module):
    def __init__(self, state_dim, action_dim, T=8):
        super().__init__()
        self.T = T
        self.encoder = encoding.PoissonEncoder()
        self.input_fc = layer.Linear(state_dim, 128)
        self.input_lif = neuron.LIFNode(
            surrogate_function=surrogate.ATan(), 
            detach_reset=True, 
            step_mode='m'
        )
        self.backbone = nn.Sequential(
            MS_ResBlock(128),
            MS_ResBlock(128)
        )
        self.output_fc = layer.Linear(128, action_dim)
        self.output_lif = neuron.LIFNode(
            v_threshold=np.inf, 
            detach_reset=True, 
            step_mode='m'
        )

    def forward(self, x):
        # 关键修复：在每次前向传播前重置整个网络状态
        functional.reset_net(self)
        
        # 扩展时间维度 [B, D] -> [T, B, D]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1)
        
        # 直接泊松编码 (无需sigmoid归一化)
        x_seq = self.encoder(x_seq)
        
        x_seq = self.input_fc(x_seq)
        x_seq = self.input_lif(x_seq)
        x_seq = self.backbone(x_seq)
        
        out_voltage = self.output_lif(self.output_fc(x_seq))
        q_value = out_voltage.mean(0)
        
        return q_value

# ===============================
# 3. 改进的 Double DQN Agent
# ===============================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = SNNQNet(state_dim, action_dim).to(self.device)
        self.target_net = SNNQNet(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-5)
        self.memory = deque(maxlen=50000)

        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.warmup_steps = 5000
        self.total_steps = 0
        self.train_freq = 4
        self.target_update = 0.001

    def select_action(self, state):
        self.total_steps += 1
        if random.random() < self.epsilon:
            return random.randrange(2)

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(state)
        self.policy_net.train()
        return q.argmax(dim=1).item()

    def train(self):
        if self.total_steps < self.warmup_steps:
            return None
            
        if self.total_steps % self.train_freq != 0:
            return None

        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # 当前Q值
        q_all = self.policy_net(states)
        q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 目标Q值 (Double DQN)
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)
            next_actions = next_q_policy.argmax(dim=1)
            
            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 仅使用TD Loss
        loss = nn.MSELoss()(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.target_update * param.data + 
                                  (1.0 - self.target_update) * target_param.data)

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

# ===============================
# 4. 主程序
# ===============================
def main():
    env = gym.make("FlappyBird-v0", render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    episodes = 20000
    losses = []
    ep_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 简化奖励重塑
            shaped_reward = reward
            if done:
                shaped_reward = -10.0
            else:
                try:
                    bird_y = state[0]
                    pipe_center_y = (state[3] + state[4]) / 2
                    shaped_reward = 0.1 * (1.0 - abs(bird_y - pipe_center_y) / 0.5)  # 归一化距离
                except:
                    shaped_reward = 0.1

            agent.memory.append((state, action, shaped_reward, next_state, done))
            l_val = agent.train()
            if l_val is not None:
                losses.append(l_val)
                
            state = next_state
            total_reward += reward

        print(f"Ep {ep:4d} | Reward {total_reward:6.1f} | Epsilon {agent.epsilon:.3f} | Steps {agent.total_steps}")
        ep_rewards.append(total_reward)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if len(losses) > 100:
        smooth_loss = np.convolve(losses, np.ones(100)/100, mode='valid')
        plt.plot(smooth_loss)
    else:
        plt.plot(losses)
    plt.title("Training Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(ep_rewards)
    plt.title("Episode Rewards")
    
    plt.tight_layout()
    plt.savefig("training_results.png")
    print("Training plot saved as training_results.png")
    torch.save(agent.policy_net.state_dict(), "snn_double_dqn_flappy.pth")

if __name__ == "__main__":
    main()