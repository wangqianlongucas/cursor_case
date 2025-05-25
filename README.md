# TSP强化学习求解项目 🚀

本项目使用三种不同的强化学习算法求解旅行商问题(Traveling Salesman Problem, TSP)：Q-learning、DQN（Deep Q-Network）和PPO（Proximal Policy Optimization）。

## 🎯 项目特色

- **统一环境架构**：创建了支持多种算法的统一TSP环境
- **三种RL算法**：Q-learning、DQN、PPO的完整实现
- **丰富状态表示**：从简单到复杂的状态特征工程
- **全面性能对比**：详细的算法性能分析和可视化
- **模块化设计**：清晰的代码结构，易于扩展

## 📁 项目结构

```
├── src/
│   ├── unified_env.py          # 🆕 统一TSP环境（支持simple/enhanced模式）
│   ├── q_learning.py           # Q-learning智能体
│   ├── dqn_agent.py           # DQN智能体实现
│   ├── ppo_agent.py           # 🆕 PPO智能体实现
│   ├── train.py               # Q-learning训练脚本
│   ├── dqn_train.py           # DQN训练脚本
│   ├── ppo_train.py           # 🆕 PPO训练脚本
│   ├── compare_all_methods.py # 🆕 三种方法全面对比
│   ├── test_unified_env.py    # 🆕 环境测试脚本
│   ├── tsp_env.py             # 原始基础TSP环境
│   ├── dqn_env.py             # 原始DQN环境
│   └── compare_methods.py     # Q-learning vs DQN比较
├── models/                    # 保存训练好的模型
├── requirements.txt           # 项目依赖
├── SETUP_GUIDE.md            # 详细环境配置指南
└── README.md                 # 项目说明文档
```

## 🔧 环境配置

### 快速安装
```bash
# 克隆项目
git clone <repository-url>
cd cursor_case

# 安装依赖
pip install -r requirements.txt

# 测试环境
python src/test_unified_env.py
```

### 依赖包
- `numpy>=1.21.0` - 数值计算
- `matplotlib>=3.4.0` - 可视化
- `tqdm>=4.62.0` - 进度条
- `torch>=1.9.0` - 深度学习框架
- `torchvision>=0.10.0` - PyTorch视觉工具
- `pandas>=1.3.0` - 数据处理

## 🚀 快速开始

### 1. 运行单个算法

```bash
# Q-learning训练
python src/train.py

# DQN训练
python src/dqn_train.py

# PPO训练
python src/ppo_train.py
```

### 2. 全面算法对比

```bash
# 运行三种算法的完整对比
python src/compare_all_methods.py
```

### 3. 测试统一环境

```bash
# 验证环境功能
python src/test_unified_env.py
```

## 🧠 算法实现详解

### 1. 统一环境架构 (`unified_env.py`)

**核心创新**：
- **双模式支持**：`simple`模式用于Q-learning，`enhanced`模式用于DQN/PPO
- **状态表示可配置**：根据算法需求自动调整状态复杂度
- **奖励函数统一**：保证不同算法的公平比较

**状态表示对比**：

| 特性 | Simple模式 | Enhanced模式 |
|------|------------|--------------|
| 状态维度 | 城市数量 | 7×城市数量+7 |
| 访问掩码 | ✅ | ✅ |
| 当前位置 | 隐式 | 显式one-hot |
| 距离信息 | ❌ | 归一化距离 |
| 相对位置 | ❌ | 归一化坐标 |
| 进度特征 | ❌ | 完成度/步数比例 |
| 启发式信息 | ❌ | 最近邻/统计特征 |

### 2. Q-learning实现

**特点**：
- 表格式Q值存储
- ε-贪婪探索策略
- 简单状态表示
- 快速收敛，适合小规模问题

**核心参数**：
```python
learning_rate=0.3      # 学习率
discount_factor=0.9    # 折扣因子
epsilon=0.9→0.01      # 探索率衰减
```

### 3. DQN实现

**特点**：
- 深度神经网络逼近Q函数
- 经验回放机制
- 目标网络稳定训练
- 丰富状态特征

**网络架构**：
```
输入层 → 512 → 256 → 128 → 输出层
       ReLU   ReLU   ReLU
```

### 4. PPO实现 🆕

**特点**：
- Actor-Critic架构
- 策略梯度方法
- 重要性采样比率裁剪
- 广义优势估计(GAE)
- 动作掩码支持

**核心组件**：
- **Actor网络**：策略函数π(a|s)
- **Critic网络**：价值函数V(s)
- **GAE**：优势函数估计
- **PPO裁剪**：稳定策略更新

## 📊 性能对比

### 算法特性对比

| 算法 | 适用规模 | 训练速度 | 最终性能 | 稳定性 | 内存需求 |
|------|----------|----------|----------|--------|----------|
| Q-learning | 5-10城市 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| DQN | 10-20城市 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| PPO | 10+城市 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 预期性能指标

**10城市TSP问题**：
- **Random Baseline**: ~500-600距离
- **Q-learning**: ~350-400距离
- **DQN**: ~300-350距离  
- **PPO**: ~280-320距离

## 🔬 实验功能

### 1. 训练可视化
- 实时训练曲线
- 奖励/距离趋势
- 损失函数变化
- 最优解路径图

### 2. 性能分析
- 收敛速度对比
- 最终性能评估
- 成功率统计
- 训练时间分析

### 3. 随机基线对比
- 随机策略性能
- 改进倍数计算
- 统计显著性测试

## 🎨 可视化特性

### 训练过程可视化
- 📈 训练曲线（奖励、距离、损失）
- 📊 性能分布直方图
- 🎯 最优解路径图
- ⏱️ 训练时间对比

### 解决方案可视化
- 🗺️ 城市分布图
- 🛣️ 最优路径展示
- 🏷️ 城市标签和方向箭头
- 📏 距离信息显示

## 🔧 高级功能

### 1. 模型保存/加载
```python
# 保存模型
agent.save_model("models/ppo_tsp_10cities.pth")

# 加载模型
agent.load_model("models/ppo_tsp_10cities.pth")
```

### 2. 超参数配置
```python
# PPO配置示例
config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01
}
```

### 3. 批量实验
```python
# 运行多种配置的对比实验
python src/compare_all_methods.py
```

## 📈 扩展方向

### 1. 算法扩展
- [ ] A3C (Asynchronous Actor-Critic)
- [ ] SAC (Soft Actor-Critic)
- [ ] Rainbow DQN
- [ ] 图神经网络方法

### 2. 问题扩展
- [ ] 动态TSP（时变距离）
- [ ] 多目标TSP
- [ ] 带容量约束的VRP
- [ ] 更大规模城市（50+）

### 3. 技术改进
- [ ] 注意力机制
- [ ] 图卷积网络
- [ ] 元学习方法
- [ ] 分布式训练

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- OpenAI Gym启发的环境设计
- PyTorch深度学习框架
- 强化学习社区的算法贡献

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 📧 Email: [wangqianlong21@mails.ucas.ac.cn]
- 🐛 Issues: [GitHub Issues](https://github.com/wangqianlongucas/cursor_case/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/wangqianlongucas/cursor_case/discussions)

---

**Happy Coding! 🎉**