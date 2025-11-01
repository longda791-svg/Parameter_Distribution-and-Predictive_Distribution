import numpy as np
from scipy.stats import multivariate_normal

class BayesianRegression:
    def __init__(self, degree=3, alpha=2.0, beta=25.0):
        """
        初始化贝叶斯线性回归模型
        
        参数:
        degree: int, 多项式基函数的阶数
        alpha: float, 权重先验分布的精度参数
        beta: float, 观测噪声的精度参数
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
        
        参数:
        param_range: tuple, 参数空间的范围
        n_points: int, 网格点的数量
        
        返回:
        W: ndarray, 第一个参数的网格
        B: ndarray, 第二个参数的网格
        Z: ndarray, 每个网格点的概率密度
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