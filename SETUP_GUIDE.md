# TSP强化学习项目 - 环境配置指南

本文档提供了完整的开发环境配置指南，包括编辑器安装、扩展配置以及项目使用说明。

## 📋 目录
1. [VSCode安装](#1-vscode安装)
2. [Cursor安装](#2-cursor安装)
3. [Cursor扩展配置](#3-cursor扩展配置)
4. [Python和Jupyter内核配置](#4-python和jupyter内核配置)
5. [项目指令总结](#5-项目指令总结)

---

## 1. VSCode安装

### Windows系统
1. 访问 [Visual Studio Code官网](https://code.visualstudio.com/)
2. 点击 "Download for Windows" 下载安装包
3. 运行下载的 `.exe` 文件
4. 按照安装向导完成安装
5. 启动VSCode并进行基本配置

### macOS系统
1. 访问 [Visual Studio Code官网](https://code.visualstudio.com/)
2. 点击 "Download for Mac" 下载 `.zip` 文件
3. 解压下载的文件
4. 将 `Visual Studio Code.app` 拖拽到 `Applications` 文件夹
5. 从启动台或应用程序文件夹启动VSCode

### Linux系统
```bash
# Ubuntu/Debian
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code

# CentOS/RHEL/Fedora
sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
sudo sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" > /etc/yum.repos.d/vscode.repo'
sudo yum install code
```

---

## 2. Cursor安装

Cursor是一个基于VSCode的AI增强代码编辑器，提供更智能的代码补全和AI辅助功能。

### 下载安装
1. 访问 [Cursor官网](https://cursor.sh/)
2. 根据你的操作系统选择对应的下载链接：
   - **Windows**: 下载 `.exe` 安装包
   - **macOS**: 下载 `.dmg` 文件
   - **Linux**: 下载 `.AppImage` 或 `.deb` 文件

### Windows安装步骤
1. 运行下载的 `.exe` 文件
2. 选择安装路径（建议使用默认路径）
3. 勾选 "Add to PATH" 选项
4. 完成安装并启动Cursor

### macOS安装步骤
1. 打开下载的 `.dmg` 文件
2. 将Cursor拖拽到Applications文件夹
3. 首次启动时可能需要在系统偏好设置中允许运行

### Linux安装步骤
```bash
# 使用AppImage（推荐）
chmod +x cursor-*.AppImage
./cursor-*.AppImage

# 或者使用deb包（Ubuntu/Debian）
sudo dpkg -i cursor-*.deb
sudo apt-get install -f  # 解决依赖问题
```

---

## 3. Cursor扩展配置

启动Cursor后，需要安装以下关键扩展来支持我们的TSP强化学习项目：

### 3.1 SSH Remote扩展
用于远程开发和服务器连接。

**安装步骤：**
1. 打开Cursor
2. 点击左侧活动栏的扩展图标（或按 `Ctrl+Shift+X`）
3. 搜索 "Remote - SSH"
4. 点击 "Install" 安装
5. 重启Cursor以激活扩展

**配置SSH连接：**
```bash
# 1. 按 Ctrl+Shift+P 打开命令面板
# 2. 输入 "Remote-SSH: Connect to Host"
# 3. 选择 "Configure SSH Hosts"
# 4. 编辑SSH配置文件，添加服务器信息：

Host myserver
    HostName your-server-ip
    User your-username
    Port 22
    IdentityFile ~/.ssh/id_rsa
```

### 3.2 Python扩展
提供Python语言支持、调试、智能提示等功能。

**安装步骤：**
1. 在扩展市场搜索 "Python"
2. 安装由Microsoft发布的Python扩展
3. 安装 "Pylance" 扩展（Python语言服务器）

**推荐的Python相关扩展：**
- **Python** (Microsoft) - 核心Python支持
- **Pylance** (Microsoft) - 高级语言服务
- **Python Docstring Generator** - 自动生成文档字符串
- **autoDocstring** - 智能文档字符串生成

### 3.3 Jupyter扩展
支持Jupyter Notebook的编辑和运行。

**安装步骤：**
1. 搜索 "Jupyter"
2. 安装以下扩展：
   - **Jupyter** (Microsoft) - 核心Jupyter支持
   - **Jupyter Keymap** - Jupyter快捷键
   - **Jupyter Notebook Renderers** - 增强渲染支持

---

## 4. Python和Jupyter内核配置

### 4.1 Python解释器选择

**步骤：**
1. 打开Python文件（如 `src/train.py`）
2. 按 `Ctrl+Shift+P` 打开命令面板
3. 输入 "Python: Select Interpreter"
4. 选择合适的Python解释器：
   - 系统Python：`/usr/bin/python3`
   - Conda环境：`~/anaconda3/envs/your-env/bin/python`
   - 虚拟环境：`./venv/bin/python`

**验证Python环境：**
```bash
# 检查Python版本
python --version

# 检查已安装的包
pip list

# 安装项目依赖
pip install -r requirements.txt
```

### 4.2 Jupyter内核配置

**为项目创建专用内核：**
```bash
# 1. 创建虚拟环境
python -m venv tsp_rl_env

# 2. 激活虚拟环境
# Windows:
tsp_rl_env\Scripts\activate
# macOS/Linux:
source tsp_rl_env/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装Jupyter内核
pip install ipykernel
python -m ipykernel install --user --name=tsp_rl_env --display-name="TSP RL Environment"
```

**在Cursor中选择Jupyter内核：**
1. 打开 `.ipynb` 文件
2. 点击右上角的内核选择器
3. 选择 "TSP RL Environment" 内核
4. 或者按 `Ctrl+Shift+P` 输入 "Jupyter: Select Interpreter to Start Jupyter Server"

### 4.3 内核管理命令

```bash
# 查看可用内核
jupyter kernelspec list

# 删除内核
jupyter kernelspec remove tsp_rl_env

# 重新安装内核
python -m ipykernel install --user --name=tsp_rl_env --display-name="TSP RL Environment"
```

---

## 5. 项目指令总结

以下是在开发TSP强化学习项目过程中使用的主要指令和操作流程：

### 5.1 项目初始化指令

```bash
# 创建项目目录结构
mkdir -p src

# 创建依赖文件
cat > requirements.txt << EOF
numpy>=1.21.0
matplotlib>=3.4.0
tqdm>=4.62.0
torch>=1.9.0
torchvision>=0.10.0
EOF

# 安装依赖
pip install -r requirements.txt
```

### 5.2 核心开发指令序列

#### 第一阶段：基础Q-learning实现
```bash
# 1. 创建TSP环境
# 用户指令：帮我建一个src的文件夹，然后写一个TSP用强化学习求解的项目，其中注意文件拆分可按照下面的推荐
# tsp_env.py, q_learning.py, train.py

# 2. 优化收敛性
# 用户指令：检查修改一下代码吧，现在的收敛性不太好，然后创建一个ipy文件

# 3. 移除训练过程可视化
# 用户指令：训练过程中不要画图，最后输出
```

#### 第二阶段：DQN实现和增强
```bash
# 4. 实现DQN版本
# 用户指令：现在写一个DQN和对于的训练代码吧，可以丰富一下state，提高学习效率

# 主要改进内容：
# - 创建 dqn_env.py：增强状态表示的TSP环境
# - 创建 dqn_agent.py：深度Q网络智能体
# - 创建 dqn_train.py：DQN训练脚本
# - 创建 compare_methods.py：Q-learning vs DQN比较
```

### 5.3 运行项目的关键命令

```bash
# 运行Q-learning训练
python src/train.py

# 运行DQN训练
python src/dqn_train.py

# 比较两种方法
python src/compare_methods.py

# 启动Jupyter Notebook
jupyter notebook tsp_rl_experiment.ipynb
```

### 5.4 项目文件结构总结

```
TSP强化学习项目/
├── src/
│   ├── tsp_env.py          # 基础TSP环境
│   ├── q_learning.py       # Q-learning智能体
│   ├── train.py            # Q-learning训练脚本
│   ├── dqn_env.py          # 增强DQN环境（丰富状态表示）
│   ├── dqn_agent.py        # DQN智能体（神经网络+经验回放）
│   ├── dqn_train.py        # DQN训练脚本
│   └── compare_methods.py  # 方法对比脚本
├── models/                 # 保存训练好的模型
├── requirements.txt        # 项目依赖
├── README.md              # 项目说明文档
└── SETUP_GUIDE.md         # 环境配置指南（本文档）
```

### 5.5 关键技术改进总结

#### Q-learning优化：
- **状态表示增强**：当前城市 + 访问状态
- **超参数优化**：学习率0.3，探索率0.9→0.01
- **奖励塑形改进**：归一化距离奖励

#### DQN创新特性：
- **丰富状态特征**：7个维度的特征工程
  - 基础特征：访问掩码 + 当前位置
  - 距离特征：归一化距离 + 相对位置
  - 进度特征：完成度 + 步数 + 距离比例
  - 启发式特征：最近邻信息
  - 统计特征：最小/平均/最大距离

- **深度网络架构**：512→256→128→输出
- **高级训练技术**：经验回放 + 目标网络 + 梯度裁剪

### 5.6 性能对比预期

| 特性 | Q-learning | DQN |
|------|------------|-----|
| 适用规模 | 5-10城市 | 10+城市 |
| 训练速度 | 快 | 慢 |
| 最终性能 | 中等 | 优秀 |
| 资源需求 | 低 | 高 |
| 稳定性 | 高 | 中等 |

---

## 🚀 快速开始

1. **安装Cursor**：下载并安装Cursor编辑器
2. **配置扩展**：安装SSH Remote、Python、Jupyter扩展
3. **克隆项目**：获取TSP强化学习项目代码
4. **安装依赖**：`pip install -r requirements.txt`
5. **选择内核**：为Python和Jupyter选择正确的解释器
6. **开始训练**：运行 `python src/train.py` 或 `python src/dqn_train.py`

## 📞 技术支持

如果在配置过程中遇到问题，请检查：
- Python版本是否 ≥ 3.7
- 是否正确安装了PyTorch
- 内核是否正确配置
- 扩展是否正常加载

---

**祝您使用愉快！** 🎉 