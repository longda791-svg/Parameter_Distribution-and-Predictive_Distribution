# 贝叶斯线性回归中的参数分布与预测分布可视化

## 项目简介

本项目实现了贝叶斯线性回归的参数后验分布和预测分布的可视化。基于 Bishop 的《Pattern Recognition and Machine Learning》第3章中的理论，展示了贝叶斯方法如何通过概率框架来处理不确定性。

## 理论背景

### 贝叶斯线性回归

在贝叶斯线性回归中，我们使用概率方法处理模型参数，主要包括：

1. **先验分布**:

   ```math
   p(w) = N(w|0, α⁻¹I)
   ```

   其中 w 是模型参数，α 是精度参数。

2. **似然函数**:

   ```math
   p(t|x,w,β) = N(w^Tφ(x), β⁻¹)
   ```

   其中 φ(x) 是基函数，β 是观测噪声的精度。

3. **后验分布**:

   ```math
   p(w|D) = N(w|m_N, S_N)
   ```

   其中：
   - S_N⁻¹ = αI + βΦ^TΦ
   - m_N = βS_NΦ^Tt

4. **预测分布**:

   ```math
   p(t_*|x_*,D) = N(t_*|m(x_*), s²(x_*))
   ```

   其中：
   - m(x_*) = m_N^T φ(x_*)
   - s²(x_*) = 1/β + φ(x_*)^T S_N φ(x_*)

## 项目结构

```plaintext
.
├── README.md              # 项目文档
├── bayesian_regression.py # 贝叶斯回归模型实现
├── main.py               # 主程序和可视化代码
└── requirements.txt      # 项目依赖
```

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

# 创建模型实例
model = BayesianRegression(degree=3, alpha=2.0, beta=25.0)

# 生成数据
x = np.linspace(0, 1, 10)
t = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, 10)

# 拟合模型
model.fit(x, t)

# 预测新数据
x_test = np.linspace(0, 1, 100)
mean, var = model.predict(x_test)
```

### 运行示例程序

```bash
python main.py
```

运行后会生成三个图形，展示不同训练数据量（N=1, 10, 100）下的：

- 参数后验分布（等高线图）
- 预测分布（包含预测均值和95%置信区间）

## 主要特性

1. **完整的贝叶斯推断**

   - 参数的完整后验分布
   - 预测的不确定性估计
   - 自动噪声水平处理

2. **灵活的模型配置**

   - 可调节的多项式基函数阶数
   - 可配置的先验精度（α）
   - 可设置的观测噪声精度（β）

3. **直观的可视化**

   - 参数空间的后验分布可视化
   - 数据空间的预测分布可视化
   - 不确定性的清晰展示

## 参数调优

模型有三个主要超参数：

1. `degree`：多项式基函数的阶数

   - 较小的值（如3）适合简单的函数
   - 较大的值（如9）可以拟合更复杂的函数

2. `alpha`：权重先验的精度参数

   - 较大的值会使模型偏向更简单的解释
   - 较小的值允许更复杂的拟合

3. `beta`：观测噪声的精度参数

   - 较大的值表示对数据的高度信任
   - 较小的值表示数据可能包含更多噪声

## 贡献指南

欢迎提交问题和改进建议！请：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建一个 Pull Request

## 参考文献

1. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer. Chapter 3.
2. MacKay, D. J. C. (2003). Information Theory, Inference, and Learning Algorithms. Cambridge University Press.

## 许可证

本项目采用 MIT 许可证 - 详情见 LICENSE 文件。