import numpy as np
from scipy.stats import multivariate_normal

class BayesianRegression:
    """贝叶斯线性回归模型
    
    实现了完整的贝叶斯线性回归，包括：
    1. 多项式基函数特征转换
    2. 贝叶斯推断（先验、似然、后验）
    3. 预测分布计算
    4. 参数后验分布可视化（注：仅支持查看前两个权重的分布）
    
    数学原理：
    - 先验分布：p(w) = N(w|0, α⁻¹I)
    - 似然函数：p(t|x,w,β) = N(w^Tφ(x), β⁻¹)
    - 后验分布：p(w|D) = N(w|m_N, S_N)
    其中：
        S_N⁻¹ = αI + βΦ^TΦ
        m_N = βS_NΦ^Tt
    
    预测分布：
    p(t_*|x_*,D) = N(m(x_*), s²(x_*))
    其中：
        m(x_*) = m_N^T φ(x_*)
        s²(x_*) = 1/β + φ(x_*)^T S_N φ(x_*)
    """
    
    def __init__(self, degree=3, alpha=2.0, beta=25.0):
        """
        初始化贝叶斯线性回归模型
        
        参数:
        degree: int, 多项式基函数的阶数，决定了模型的复杂度和参数维度(degree+1)
        alpha: float, 权重先验分布的精度参数，控制了先验的强度
        beta: float, 观测噪声的精度参数，控制了对数据的信任程度
        
        注意：
        - 虽然模型支持任意阶数的多项式，但参数可视化仅展示前两个权重
        - 当degree > 1时，参数后验分布的可视化是高维分布在前两维上的投影
        """
        self.degree = degree
        self.alpha = alpha
        self.beta = beta
        self.m_N = None  # 后验分布均值
        self.S_N = None  # 后验分布协方差

    def polynomial_basis(self, x):
        """计算多项式基函数"""
        return np.array([x**i for i in range(self.degree + 1)]).T

    def fit(self, X, t):
        """
        拟合模型
        
        参数:
        X: ndarray, 输入特征
        t: ndarray, 目标值
        """
        # 计算设计矩阵
        phi = self.polynomial_basis(X)
        
        # 计算后验分布参数
        S_N_inv = self.alpha * np.eye(phi.shape[1]) + self.beta * phi.T @ phi
        self.S_N = np.linalg.inv(S_N_inv)
        self.m_N = self.beta * self.S_N @ phi.T @ t
        
        return self

    def predict(self, X_test):
        """
        预测新数据点的分布
        
        参数:
        X_test: ndarray, 测试输入特征
        
        返回:
        mean: ndarray, 预测均值
        var: ndarray, 预测方差
        """
        if self.m_N is None or self.S_N is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        # 计算测试数据的基函数
        phi_test = self.polynomial_basis(X_test)
        
        # 计算预测分布
        mean = phi_test @ self.m_N
        var = 1/self.beta + np.sum(phi_test @ self.S_N * phi_test, axis=1)
        
        return mean, var

    def get_parameter_distribution(self, param_range=(-1, 1), n_points=100):
        """
        获取参数的后验分布（仅返回前两个参数的分布）
        
        注意：这是一个降维可视化方法。当多项式阶数 > 1 时，模型参数维度 > 2，
        此时返回的是高维参数空间在前两个权重上的投影（边缘分布）。这种可视化
        方法遵循了 Bishop 教材中图 3.7 的展示方式。
        
        参数:
        param_range: tuple, 参数空间的范围
        n_points: int, 网格点的数量
        
        返回:
        W: ndarray, 第一个参数(w₀)的网格
        B: ndarray, 第二个参数(w₁)的网格
        Z: ndarray, 每个网格点的概率密度
        
        注意：返回的概率密度是完整后验分布在前两个参数维度上的边缘分布
        """
        if self.m_N is None or self.S_N is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        # 创建参数空间的网格
        param_points = np.linspace(param_range[0], param_range[1], n_points)
        W, B = np.meshgrid(param_points, param_points)
        pos = np.dstack((W, B))
        
        # 计算概率密度
        rv = multivariate_normal(self.m_N[:2], self.S_N[:2, :2])
        Z = rv.pdf(pos)
        
        return W, B, Z