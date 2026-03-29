#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Dueling Deep Q Network 核心代码
# 导入pytorch做为训练的基本核心
import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()

        # 共享特征提取层
        self.fc1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(inplace=True))

        # Value Stream: 评估当前状态本身的好坏
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # Advantage Stream: 评估每个动作相对于平均水平的好坏
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Q = V + (A - mean(A))
        # 减去均值保证 advantage 的可辨识性
        return value + advantage - advantage.mean()
