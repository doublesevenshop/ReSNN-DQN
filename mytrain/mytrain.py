import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 引入 SpikingJelly 组件
from spikingjelly.activation_based import neuron, functional, surrogate, layer

# 假设这两个模块在你本地的 src 文件夹下
from src.flappy_bird import FlappyBird
from src.utils import pre_processing

class DeepSQN(nn.Module):
    def __init__(self, T=8): # 修改点1：增加 T 到 8
        """
        Deep Spiking Q-Network
        T: 模拟的时间步数 (Time Steps)
        """
        super(DeepSQN, self).__init__()
        self.T = T

        # 使用 ATan 作为代理梯度，alpha 设为 2.0 可以让梯度在初期更容易回传
        surrogate_function = surrogate.ATan(alpha=2.0)

        # 1. 卷积层部分
        self.conv_net = nn.Sequential(
            # Conv 1
            layer.Conv2d(4, 32, kernel_size=8, stride=4, step_mode='m'),
            layer.BatchNorm2d(32, step_mode='m'), 
            # 这里的 LIF 使用 detach_reset=True 是对的
            neuron.LIFNode(surrogate_function=surrogate_function, detach_reset=True, step_mode='m'),
            
            # Conv 2
            layer.Conv2d(32, 64, kernel_size=4, stride=2, step_mode='m'),
            layer.BatchNorm2d(64, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate_function, detach_reset=True, step_mode='m'),
            
            # Conv 3
            layer.Conv2d(64, 64, kernel_size=3, stride=1, step_mode='m'),
            layer.BatchNorm2d(64, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate_function, detach_reset=True, step_mode='m'),
            
            layer.Flatten(step_mode='m')
        )

        # 2. 全连接层部分
        self.fc_net = nn.Sequential(
            layer.Linear(3136, 512, step_mode='m'),
            # 最后一层隐藏层，确保它能发脉冲。
            neuron.LIFNode(surrogate_function=surrogate_function, detach_reset=True, step_mode='m')
        )

        # 3. 输出层 (Q-Value Head)
        # 我们直接取 Linear 层的输出作为电流/电压值
        self.fc_out = layer.Linear(512, 2, step_mode='m')

    def forward(self, x):
        # x shape: [Batch, 4, 84, 84]
        
        functional.reset_net(self)

        # [Batch, C, H, W] -> [T, Batch, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        features = self.conv_net(x_seq)
        hidden = self.fc_net(features)
        
        out_seq = self.fc_out(hidden)

        q_values = out_seq.mean(0) 
        
        return q_values

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Spiking Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=128, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4) # SNN 可能需要调整一点学习率，默认 1e-6 可能太小，建议尝试 1e-4
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=20000000)
    parser.add_argument("--replay_memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard_sqn")
    parser.add_argument("--saved_path", type=str, default="trained_models_sqn")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    
    # 使用修改后的 SNN 模型，T=4
    model = DeepSQN(T=8).cuda()
    target_q_net = DeepSQN(T=8).cuda()
    
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    writer = SummaryWriter(opt.log_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    
    # 假设 FlappyBird 类依然正常工作
    game_state = FlappyBird("dpn") 
    state, reward, terminal = game_state.step(0)

    replay_memory = []
    iter = 0
    max_reward = 0
    
    while iter < opt.num_iters:
        # 模型预测 (前向传播会自动处理 reset 和时间维度)
        prediction = model(state)
        
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (
                (opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters)
        u = random()
        random_action = u <= epsilon
        
        if random_action:
            action = randint(0, 1)
        else:
            action = prediction.argmax().item()

        next_state, reward, terminal = game_state.step(action)
        replay_memory.append([state, action, reward, next_state, terminal])
        state = next_state
        
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
            
        loss = 0
        if len(replay_memory) > 32:
            batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
            
            states = torch.cat(state_batch, dim=0).cuda()
            actions = torch.tensor(action_batch).view(-1, 1).cuda()
            rewards = torch.tensor(reward_batch).view(-1, 1).cuda()
            dones = torch.tensor(terminal_batch).view(-1, 1).int().cuda()
            next_states = torch.cat(next_state_batch, dim=0).cuda()

            # 计算 Q(s, a)
            q_values = model(states).gather(1, actions)
            
            # 计算 Max Q(s', a') (Target Net)
            # 同样，Target Net 也是 SNN，内部也会 reset 和 repeat
            max_next_q_values = target_q_net(next_states).gather(1, actions).max(1)[0].view(-1, 1) # 这里逻辑保持原版简化处理，Double DQN 可进一步优化
            # 假设你想用标准的 DQN:
            max_next_q_values = target_q_net(next_states).detach().max(1)[0].view(-1, 1)

            q_targets = rewards + opt.gamma * max_next_q_values * (1 - dones)
            
            optimizer.zero_grad()
            loss = torch.mean(criterion(q_values, q_targets))
            loss.backward()
            
            # SNN 梯度修剪 (可选，但推荐)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            if iter % 100 == 0:
                target_q_net.load_state_dict(model.state_dict())
                
        if reward > max_reward:
            max_reward = reward
            print("max_reward Iteration: {}/{}, Action: {}, Loss: {:.6f}, Epsilon {:.4f}, Reward: {}, Q-value: {:.4f}".format(
                iter + 1,
                opt.num_iters,
                action,
                loss.item() if isinstance(loss, torch.Tensor) else loss,
                epsilon, reward, torch.max(prediction)))

        if iter % 1000 == 0:
            print("Iteration: {}/{}, Action: {}, Loss: {:.6f}, Epsilon {:.4f}, Reward: {}, Q-value: {:.4f}".format(
                iter + 1,
                opt.num_iters,
                action,
                loss.item() if isinstance(loss, torch.Tensor) else loss,
                epsilon, reward, torch.max(prediction)))
                
        iter += 1
        writer.add_scalar('Train/Loss', loss, iter)
        writer.add_scalar('Train/Epsilon', epsilon, iter)
        writer.add_scalar('Train/Reward', reward, iter)
        writer.add_scalar('Train/Q-value', torch.max(prediction), iter)
        
        if (iter + 1) % 1000000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(opt.saved_path, iter + 1))
            
    torch.save(model, "{}/flappy_bird".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)