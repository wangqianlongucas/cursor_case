# TSP强化学习求解项目

本项目使用Q-learning和DQN（Deep Q-Network）算法求解旅行商问题(Traveling Salesman Problem, TSP)。

## 项目结构

```
├── src/
│   ├── tsp_env.py          # 基础TSP环境实现
│   ├── q_learning.py       # Q-learning智能体
│   ├── train.py            # Q-learning训练脚本
│   ├── dqn_env.py          # 增强的DQN TSP环境
│   ├── dqn_agent.py        # DQN智能体实现
│   ├── dqn_train.py        # DQN训练脚本
│   └── compare_methods.py  # Q-learning vs DQN比较
├── tsp_rl_experiment.ipynb # Jupyter实验notebook
├── requirements.txt        # 依赖包
└── README.md              # 项目说明
```

## 算法实现

### 1. Q-learning实现
- **环境**: `tsp_env.py` - 基础TSP环境
- **智能体**: `q_learning.py` - 表格式Q-learning
- **训练**: `train.py` - Q-learning训练脚本

### 2. DQN实现 🆕
- **环境**: `dqn_env.py` - 增强状态表示的TSP环境
- **智能体**: `dqn_agent.py` - 深度Q网络智能体
- **训练**: `dqn_train.py` - DQN训练脚本

## DQN主要改进

### 1. 丰富的状态表示 (`dqn_env.py`)
DQN环境提供了比Q-learning更丰富的状态特征：

1. **基础特征**:
   - 访问城市掩码（one-hot编码）
   - 当前城市位置（one-hot编码）

2. **距离特征**:
   - 当前城市到所有未访问城市的归一化距离
   - 相对位置向量（归一化坐标差）

3. **进度特征**:
   - 完成比例（已访问城市数/总城市数）
   - 步数比例
   - 累计距离比例

4. **启发式特征**:
   - 最近邻信息（前3个最近城市的权重）
   - 统计特征（到剩余城市的最小/平均/最大距离）

5. **增强奖励塑形**:
   - 归一化距离惩罚
   - 完成奖励
   - 进度奖励
   - 最近邻奖励
   - 回溯惩罚

### 2. 深度神经网络架构 (`dqn_agent.py`)
- **网络结构**: 3层全连接网络 (512→256→128→输出)
- **激活函数**: ReLU + Dropout(0.2)
- **权重初始化**: Xavier初始化
- **经验回放**: 100K容量的回放缓冲区
- **目标网络**: 每1000步更新一次
- **梯度裁剪**: 防止梯度爆炸

### 3. 训练优化
- **批量训练**: 64样本批次
- **双网络**: 主网络 + 目标网络
- **ε-贪婪策略**: 动态探索率衰减
- **有效动作掩码**: 只考虑合法动作

## 安装和使用

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行Q-learning训练
```bash
python src/train.py
```

### 3. 运行DQN训练
```bash
python src/dqn_train.py
```

### 4. 比较两种方法
```bash
python src/compare_methods.py
```

### 5. 使用Jupyter Notebook
```bash
jupyter notebook tsp_rl_experiment.ipynb
```

## 性能比较

### Q-learning vs DQN

| 特性 | Q-learning | DQN |
|------|------------|-----|
| 状态表示 | 简单二进制向量 | 丰富多维特征 |
| 学习方式 | 表格式Q值 | 神经网络近似 |
| 内存需求 | 低 | 高 |
| 训练时间 | 快 | 慢 |
| 扩展性 | 有限 | 好 |
| 收敛稳定性 | 高 | 中等 |

### 预期性能
- **小规模问题** (5-10城市): Q-learning通常表现更好
- **中等规模问题** (10-20城市): DQN开始显示优势
- **大规模问题** (20+城市): DQN明显优于Q-learning

## 算法原理

### Q-learning算法
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

### DQN算法
```
L(θ) = E[(r + γ max Q(s',a';θ⁻) - Q(s,a;θ))²]
```

其中：
- θ: 主网络参数
- θ⁻: 目标网络参数
- 经验回放打破数据相关性
- 目标网络提供稳定的学习目标

## 实验结果

### 收敛性改进
1. **Q-learning改进**:
   - 增强状态表示（当前城市 + 访问状态）
   - 优化超参数（学习率0.3，探索率0.9→0.01）
   - 改进奖励塑形

2. **DQN改进**:
   - 丰富的状态特征工程
   - 深度网络的表示学习能力
   - 经验回放提高样本效率
   - 目标网络稳定训练

### 性能指标
- **收敛速度**: DQN通常需要更多训练轮次
- **最终性能**: DQN在复杂问题上表现更好
- **稳定性**: Q-learning更稳定，DQN可能有波动
- **计算资源**: DQN需要更多GPU/CPU资源

## 使用建议

### 选择Q-learning的情况
- 城市数量 ≤ 10
- 计算资源有限
- 需要快速原型验证
- 要求训练稳定性

### 选择DQN的情况
- 城市数量 > 10
- 有GPU加速
- 需要最佳性能
- 可以接受较长训练时间

## 扩展方向

1. **算法改进**:
   - Double DQN
   - Dueling DQN
   - Prioritized Experience Replay
   - Rainbow DQN

2. **问题扩展**:
   - 动态TSP
   - 多目标TSP
   - 带时间窗的TSP
   - 车辆路径问题(VRP)

3. **混合方法**:
   - DQN + 启发式算法
   - 多智能体协作
   - 迁移学习

## 注意事项

1. **DQN训练**:
   - 需要足够的训练轮次（通常5000+）
   - 对超参数敏感
   - 可能出现训练不稳定

2. **状态空间**:
   - DQN状态维度较高，计算开销大
   - 特征工程对性能影响显著

3. **硬件要求**:
   - DQN建议使用GPU加速
   - 内存需求较高（经验回放缓冲区）

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！特别欢迎：
- 新的算法实现
- 性能优化
- 更好的特征工程
- 实验结果分析