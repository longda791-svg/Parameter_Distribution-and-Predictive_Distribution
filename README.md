# 贝叶斯线性回归中的参数分布与预测分布可视化

## 项目简介

本项目实现了贝叶斯线性回归的参数后验分布和预测分布的可视化。基于 Bishop 的《Pattern Recognition and Machine Learning》第3章（图3.7-3.9）的理论，展示了贝叶斯方法如何通过概率框架来处理不确定性。项目通过实验展示了在数据量从 N=1 增加到 N=100 过程中，模型如何逐步"学习"并减少不确定性。

## 理论背景

### 贝叶斯线性回归

在贝叶斯线性回归中，我们使用概率方法处理模型参数。通过多项式基函数 φ(x) 进行特征转换，使用概率框架处理不确定性：

1. **先验分布**:

   $$ p(w) = \mathcal{N}(w|0, \alpha^{-1}I) $$

   其中 w 是权重向量，α 是先验分布的精度参数。较大的 α 值会使模型偏向更简单的解释。

2. **似然函数**:

   $$ p(t|x,w,\beta) = \mathcal{N}(w^T\phi(x), \beta^{-1}) $$

   其中 φ(x) 是多项式基函数，β 是观测噪声的精度参数。较大的 β 值表示对数据的高度信任。

3. **后验分布**:

   $$ p(w|D) = \mathcal{N}(w|m_N, S_N) $$

   其中：
   - $ S_N^{-1} = \alpha I + \beta \Phi^T \Phi $ （后验协方差的逆）
   - $ m_N = \beta S_N \Phi^T t $ （后验均值）
   - Φ 是设计矩阵，其中每行是一个数据点的特征向量 φ(x)

4. **预测分布**:

   $$ p(t_*|x_*,D) = \mathcal{N}(t_*|m(x_*), s^2(x_*)) $$

   其中：
   - $ m(x_*) = m_N^T \phi(x_*) $ （预测均值）
   - $ s^2(x_*) = \frac{1}{\beta} + \phi(x_*)^T S_N \phi(x_*) $ （预测方差）
   
这个预测方差的两个组成部分反映了两种不确定性来源：
1. $ \frac{1}{\beta} $ 表示数据的固有噪声
2. $ \phi(x_*)^T S_N \phi(x_*) $ 表示参数的不确定性

## 项目结构

```plaintext
.
├── README.md              # 项目文档
├── bayesian_regression.py # 贝叶斯回归模型实现
├── main.py               # 主程序和可视化代码
├── requirements.txt      # 项目依赖
├── .gitignore           # Git 忽略文件配置
└── LICENSE              # MIT 许可证
```

## 代码结构说明

### bayesian_regression.py

核心模型实现，包含以下功能：

- `BayesianRegression` 类的完整实现
- 支持任意阶数的多项式基函数
- 完整的贝叶斯推断过程
- 参数后验分布可视化（注意：仅展示前两个权重的分布）
- 预测分布的计算

### main.py

可视化和实验代码，包含以下功能：

- 合成数据生成（基于正弦函数）
- 不同数据量下的学习效果展示
- 参数后验分布的等高线图
- 预测分布的可视化（含置信区间）

## 安装说明

1. 克隆仓库：

   ```bash
   git clone <repository-url>
   cd Parameter_Distribution\ and\ Predictive_Distribution
   ```

2. 创建虚拟环境（推荐）：

   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

### 基本使用

```python
from bayesian_regression import BayesianRegression
import numpy as np

# 创建模型实例（可配置超参数）
model = BayesianRegression(
    degree=3,    # 多项式阶数：决定模型复杂度
    alpha=2.0,   # 先验精度：控制正则化强度
    beta=25.0    # 噪声精度：反映数据可靠性
)

# 生成或准备数据
x = np.linspace(0, 1, 10)
t = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, 10)

# 拟合模型
model.fit(x, t)

# 预测新数据
x_test = np.linspace(0, 1, 100)
mean, var = model.predict(x_test)  # 返回预测均值和方差
```

### 运行示例程序

```bash
python main.py
```

运行后会生成三组图形，展示不同训练数据量下的贝叶斯学习过程：

1. N=1 的情况：
   - 参数后验分布较"扁平"（高度不确定）
   - 预测的置信区间很宽
   - 预测均值可能偏离真实函数

2. N=10 的情况：
   - 参数后验分布开始"收缩"
   - 置信区间变窄
   - 预测均值更接近真实函数

3. N=100 的情况：
   - 参数后验分布变得"尖锐"
   - 置信区间显著变窄
   - 预测均值很好地拟合了真实函数

## 主要特性

1. **完整的贝叶斯推断实现**

   - 多项式基函数特征转换
   - 精确的后验分布计算
   - 预测分布的解析解
   - 完整的不确定性量化

2. **灵活的模型配置**

   - 可配置的多项式阶数
   - 可调节的先验强度
   - 自适应的噪声处理
   - 透明的超参数控制

3. **直观的可视化展示**

   - 参数空间的后验分布等高线图
   - 数据空间的预测分布曲线
   - 置信区间的动态变化
   - 学习过程的渐进展示

## 超参数调优指南

模型包含三个关键超参数，每个都影响着不同的学习方面：

1. `degree`：多项式基函数的阶数

   - 控制模型的复杂度和拟合能力
   - 较小值（如3）适合简单的函数关系
   - 较大值（如9）可以捕捉更复杂的非线性关系
   - 需要注意：阶数越高，需要的数据越多

2. `alpha`：权重先验的精度参数

   - 控制模型的正则化强度
   - 较大值促使权重接近零，产生更简单的模型
   - 较小值允许权重取更大范围，可能过拟合
   - 建议值范围：[0.1, 10.0]

3. `beta`：观测噪声的精度参数

   - 反映对数据的信任程度
   - 较大值表示数据噪声小，模型会更严格地拟合数据点
   - 较小值表示数据噪声大，模型会产生更平滑的拟合
   - 建议根据数据的实际噪声水平调整

## 贡献指南

欢迎为项目做出贡献！以下是建议的步骤：

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/NewFeature`
3. 提交改动：`git commit -m 'Add NewFeature'`
4. 推送分支：`git push origin feature/NewFeature`
5. 提交 Pull Request

### 开发建议

- 添加新特性时请包含相应的文档
- 确保代码通过所有现有测试
- 遵循项目的代码风格
- 添加必要的注释和文档字符串

## 参考文献

1. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
   - 第3章：线性回归的贝叶斯处理
   - 图3.7-3.9：参数后验和预测分布的可视化

2. MacKay, D. J. C. (2003). Information Theory, Inference, and Learning Algorithms.
   - 第2章：概率推断
   - 第3章：线性模型中的贝叶斯方法

## 许可证

本项目采用 MIT 许可证 - 详情见 LICENSE 文件。