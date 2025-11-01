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
    """绘制参数分布和预测分布"""
    # 生成数据
    x_train, t_train = generate_data(N)
    x_test = np.linspace(0, 1, 100)
    
    # 创建并拟合模型
    model = BayesianRegression(degree=degree, alpha=alpha, beta=beta)
    model.fit(x_train, t_train)
    
    # 计算预测分布
    mean, var = model.predict(x_test)
    std = np.sqrt(var)
    
    # 获取参数后验分布
    W, B, Z = model.get_parameter_distribution()
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 参数后验分布
    ax1 = plt.subplot(121)
    plt.contourf(W, B, Z, levels=20)
    plt.colorbar()
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.title('Parameter Posterior Distribution')
    
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
    """展示不同数据量下的结果"""
    for N in [1, 10, 100]:
        fig = plot_distributions(N=N)
        plt.show()

if __name__ == "__main__":
    main()
