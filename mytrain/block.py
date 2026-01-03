import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron, functional, surrogate, layer

# ===============================
# 1. 修正后的脉冲残差模块 (保持不变)
# ===============================
class MS_ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 必须显式设置 step_mode='m'
        self.conv = layer.Linear(dim, dim, step_mode='m')
        self.bn = layer.BatchNorm1d(dim, step_mode='m') 
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
# 2. 修正后的 Spiking 特征提取器 (关键修复点)
# ===============================
class SpikingFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 关键修复：所有 layer.Conv2d 和 layer.BatchNorm2d 都必须加 step_mode='m'
        self.conv_net = nn.Sequential(
            layer.Conv2d(4, 32, kernel_size=8, stride=4, step_mode='m'),
            layer.BatchNorm2d(32, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m'),
            
            layer.Conv2d(32, 64, kernel_size=4, stride=2, step_mode='m'),
            layer.BatchNorm2d(64, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m'),
            
            layer.Conv2d(64, 64, kernel_size=3, stride=1, step_mode='m'),
            layer.BatchNorm2d(64, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m'),
            
            layer.Flatten(step_mode='m') # Flatten 也需要知道它在处理序列
        )
        
    def forward(self, x_seq):
        # x_seq: [T, B, 4, 84, 84]
        # 加上 step_mode='m' 后，layer.Conv2d 会自动合并 T 和 B 维度
        return self.conv_net(x_seq)

# ===============================
# 3. SNN Actor (策略网络)
# ===============================
class SNNActor(nn.Module):
    def __init__(self, T=4): # 建议 T 先设为 4，跑通后再加
        super().__init__()
        self.T = T
        self.feature_extractor = SpikingFeatureExtractor()
        
        # 84x84 经过 Conv 层后的维度计算:
        # 84 -> (84-8)/4 + 1 = 20 (Conv1)
        # 20 -> (20-4)/2 + 1 = 9  (Conv2)
        # 9  -> (9-3)/1 + 1 = 7   (Conv3)
        # output: 7*7*64 = 3136
        feature_dim = 3136 
        
        self.fc_reduce = layer.Linear(feature_dim, 128, step_mode='m')
        self.lif_reduce = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m')
        
        self.res_block = MS_ResBlock(128)
        
        self.fc_out = layer.Linear(128, 2, step_mode='m')
        # 最后一层不需要 LIF，直接输出电压
        self.output_lif = neuron.LIFNode(v_threshold=np.inf, detach_reset=True, step_mode='m') 

    def forward(self, x):
        # 如果 x 是 numpy 数组（例如在 step 过程中），先转 tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).cuda()
        
        # 如果输入是单张图片 [C, H, W]，先扩充 Batch 维度 -> [1, C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        functional.reset_net(self) 
        
        # 扩展时间维度 [B, C, H, W] -> [T, B, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        
        features = self.feature_extractor(x_seq)
        x_seq = self.fc_reduce(features)
        x_seq = self.lif_reduce(x_seq)
        
        x_seq = self.res_block(x_seq)
        
        out = self.fc_out(x_seq)
        out_voltage = self.output_lif(out) 
        
        # 取时间步平均作为 Logits
        logits = out_voltage.mean(0) 
        
        return nn.functional.softmax(logits, dim=1)

# ===============================
# 4. SNN Critic (价值网络)
# ===============================
class SNNCritic(nn.Module):
    def __init__(self, T=4):
        super().__init__()
        self.T = T
        self.feature_extractor = SpikingFeatureExtractor()
        
        feature_dim = 3136
        self.fc_reduce = layer.Linear(feature_dim, 128, step_mode='m')
        self.lif_reduce = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m')
        
        self.res_block = MS_ResBlock(128)
        
        self.fc_out = layer.Linear(128, 1, step_mode='m')
        self.output_lif = neuron.LIFNode(v_threshold=np.inf, detach_reset=True, step_mode='m')

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).cuda()
        if x.dim() == 3:
            x = x.unsqueeze(0)

        functional.reset_net(self)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        
        features = self.feature_extractor(x_seq)
        x_seq = self.fc_reduce(features)
        x_seq = self.lif_reduce(x_seq)
        x_seq = self.res_block(x_seq)
        
        out = self.fc_out(x_seq)
        out_voltage = self.output_lif(out)
        
        value = out_voltage.mean(0) 
        return value