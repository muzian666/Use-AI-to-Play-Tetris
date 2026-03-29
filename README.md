# 用 Deep Q-Network 训练 AI 自主游玩俄罗斯方块

基于 PyTorch 实现的 Deep Q-Network (DQN) 强化学习算法，让 AI 学会自主游玩俄罗斯方块。

## 项目结构

```
├── core/
│   ├── game.py              # 俄罗斯方块游戏环境
│   └── deep_q_network.py    # Deep Q-Network 神经网络模型
├── training.py              # 训练入口脚本
├── LICENSE                  # MIT 许可证
└── README.md
```

## 算法原理

本项目使用 **Deep Q-Network (DQN)** 算法，核心要素如下：

### 状态表示

游戏状态由 4 个特征组成：

| 特征 | 说明 |
|------|------|
| `lines_cleared` | 已消除的行数 |
| `holes` | 棋盘上的空洞数（被方块覆盖的空格） |
| `bumpiness` | 相邻列的高度差总和 |
| `height` | 所有列的高度总和 |

### 奖励函数

- 每成功放置一个方块：`+1`
- 每消除 N 行：额外奖励 `N^2 × width`
- 游戏结束：`-2`

### 神经网络结构

```
Linear(4 → 64) → ReLU → Linear(64 → 64) → ReLU → Linear(64 → 1)
```

- 输入：4 维状态向量
- 输出：1 维 Q 值（对该状态的评估）
- 权重初始化：Xavier 均匀分布

### 训练策略

- **Epsilon-Greedy 探索**：epsilon 从 1.0 线性衰减至 0.001，衰减周期 2000 个 epoch
- **经验回放 (Experience Replay)**：使用 deque 存储历史经验，训练时随机采样 batch
- **优化器**：Adam (lr=1e-3)
- **损失函数**：MSE Loss

## 环境依赖

- Python 3.8+
- PyTorch
- NumPy
- OpenCV (`opencv-python`)
- Pillow
- Matplotlib
- TensorBoardX

安装依赖：

```bash
pip install torch numpy opencv-python pillow matplotlib tensorboardX
```

## 使用方法

### 开始训练

```bash
python training.py
```

### 自定义参数

```bash
python training.py --width 10 --height 20 --block_size 30 --num_epochs 3000 --batch_size 512 --lr 1e-3
```

### 全部参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--width` | 10 | 棋盘宽度 |
| `--height` | 20 | 棋盘高度 |
| `--block_size` | 30 | 渲染方块大小（像素） |
| `--batch_size` | 512 | 每批次样本数 |
| `--lr` | 1e-3 | 学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--initial_epsilon` | 1 | 初始探索率 |
| `--final_epsilon` | 1e-3 | 最终探索率 |
| `--num_decay_epochs` | 2000 | 探索率衰减 epoch 数 |
| `--num_epochs` | 3000 | 总训练 epoch 数 |
| `--replay_memory_size` | 5 | 经验回放池大小 |
| `--log_path` | tensorboard | TensorBoard 日志路径 |
| `--saved_path` | trained_models | 模型保存路径 |

### 查看 TensorBoard 训练曲线

```bash
tensorboard --logdir=tensorboard
```

## 游戏特性

- 7 种标准俄罗斯方块（I、O、T、S、Z、J、L）
- 7-bag 随机系统（保证方块分布均匀）
- 实时 OpenCV 渲染，显示得分、方块数、消除行数
- 支持 GPU 加速训练（自动检测 CUDA）

## License

[MIT](LICENSE)
