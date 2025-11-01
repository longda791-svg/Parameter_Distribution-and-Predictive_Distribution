import numpy as np
import matplotlib.pyplot as plt
from bayesian_regression import BayesianRegression

# 设置随机种子以确保结果可重复
np.random.seed(0)

def generate_data(N, noise_std=0.2):
    """生成合成数据"""
    x = np.linspace(0, 1, N)
    t = np.sin(2 * np.pi * x) + np.random.normal(0, noise_std, N)
    return x, t

def plot_distributions(N=10, degree=3, alpha=2.0, beta=25.0):
    """绘制参数分布和预测分布
    
    参数:
    N: int, 训练数据点的数量
    degree: int, 多项式基函数的阶数
    alpha: float, 权重先验分布的精度参数
    beta: float, 观测噪声的精度参数
    
    图形说明:
    1. 左图：参数后验分布
       - 展示前两个权重(w₀, w₁)的后验分布
       - 对于degree>1的情况，这是高维分布的2D投影
       - 等高线表示概率密度
    2. 右图：预测分布
       - 蓝点：训练数据
       - 绿线：真实的sin函数
       - 红线：预测均值
       - 粉色区域：95%置信区间(±2σ)
    
    注意：随着N增大，您应该观察到：
    1. 参数后验分布变得更"尖锐"（不确定性降低）
    2. 预测的置信区间变窄（模型更确定）
    3. 预测均值更接近真实函数
    """
    # 生成数据
    x_train, t_train = generate_data(N)
    x_test = np.linspace(0, 1, 100)
    
    # 创建并拟合模型
    model = BayesianRegression(degree=degree, alpha=alpha, beta=beta)
    model.fit(x_train, t_train)
    
    # 计算预测分布
    mean, var = model.predict(x_test)
    std = np.sqrt(var)
    
    # 获取参数后验分布（注意：这是高维分布在w₀-w₁平面上的投影）
    W, B, Z = model.get_parameter_distribution()
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 参数后验分布
    ax1 = plt.subplot(121)
    plt.contourf(W, B, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.xlabel('w₀')
    plt.ylabel('w₁')
    plt.title(f'Parameter Posterior Distribution (N={N})')
    
    # 2. 预测分布
    ax2 = plt.subplot(122)
    # 绘制训练数据
    plt.scatter(x_train, t_train, c='b', label='Training data')
    # 绘制真实函数
    plt.plot(x_test, np.sin(2 * np.pi * x_test), 'g', label='True function')
    # 绘制预测均值
    plt.plot(x_test, mean, 'r', label='Predictive mean')
    # 绘制置信区间（±2个标准差）
    plt.fill_between(x_test, mean - 2*std, mean + 2*std, color='r', alpha=0.2, label='95% confidence')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Predictive Distribution (N={N})')
    plt.legend()
    
    plt.tight_layout()
    return fig

def main():
    """
    展示不同训练数据量下的贝叶斯学习效果
    
    生成三组图，展示N=1、10、100时的参数后验分布和预测分布：
    - N=1: 极少数据点，高度不确定性
    - N=10: 中等数据量，不确定性开始降低
    - N=100: 充足数据，模型达到较好的确定性
    
    观察重点：
    1. 参数后验分布从"扁平"变得"尖锐"
    2. 预测置信区间逐渐变窄
    3. 预测均值逐渐逼近真实函数
    """
    # 为了便于对比，固定随机种子
    np.random.seed(0)
    
    # 展示不同数据量的学习效果
    for N in [1, 10, 100]:
        fig = plot_distributions(N=N)
        # 可选：保存图片
        # fig.savefig(f'bayesian_regression_N{N}.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()
