# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:31:10 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:34:51 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:46:40 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:39:05 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:51:51 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 21:35:54 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:07:16 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:48:00 2025

@author: zhuqi
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import torch.optim as optim
from torch.autograd import Variable
import scipy.io
import time
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.init as init

# 设置随机种子以确保结果可复现
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        
        # 定义网络层
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            self.layers.append(layer)
        
        # Xavier 初始化
        self.apply(self._initialize_weights)
        
        self.activation = nn.Tanh()

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x)

# 定义 φ 函数
def phi_minus(x, y):
    term1 = torch.sin(torch.tensor(torch.pi) * (x + y) / 4)
    term2 = torch.sin(torch.tensor(torch.pi) * (x - 2*y - 6) / 4)
    term3 = 3 * torch.sin(torch.tensor(torch.pi) * (x - y) / 4)
    term4 = 3 * torch.sin(torch.tensor(torch.pi) * (x - 2*y - 2) / 4)
    inside = term1 - term2 + term3 - term4 + 12 * torch.tensor(torch.pi)
    return -2 / torch.sqrt(3 * torch.tensor(torch.pi)) * torch.sqrt(inside)

def phi_plus(x, y):
    term1 = torch.sin(torch.tensor(torch.pi) * (x + y) / 4)
    term2 = 3 * torch.sin(torch.tensor(torch.pi) * (x - 2*y + 2) / 4)
    term3 = 3 * torch.sin(torch.tensor(torch.pi) * (x - y) / 4)
    term4 = torch.sin(torch.tensor(torch.pi) * (x - 2*y + 6) / 4)
    inside = term1 - term2 + term3 - term4 + 3 * torch.tensor(torch.pi)
    return 2 / torch.sqrt(3 * torch.tensor(torch.pi)) * torch.sqrt(inside)

def build_dataset(x_f_batch, x_add):
    return torch.cat((x_f_batch, x_add), dim=0)

def add_data(x_f_test, residual_test, add_k):
    abs_residual = np.abs(residual_test)
    topk_indices = np.argsort(-abs_residual)[:add_k]
    x_add = x_f_test[topk_indices]
    
    # 强制转换为二维数组 [新增]
    if x_add.ndim == 1:
        x_add = x_add.reshape(-1, 2)
    elif x_add.ndim == 3:
        x_add = x_add.reshape(-1, 2)
        
    return torch.tensor(x_add, dtype=torch.float32, requires_grad=True).to(device)

def main():
    # 参数设置
    L = 4.0  # x ∈ [-2, 2]
    T_max = 1.0
    initial_size_internal = 1800
    initial_size_boundary = 1000
    # initial_size_initial = 500
    max_iterations = 40
    residual_threshold = 1e-6
    inner_max_steps = 15000
    inner_loss_threshold = 1e-13
    add_k = 5
    lr = 0.001

    # 生成训练数据
    # 内部点
    x_f = torch.FloatTensor(initial_size_internal, 1).uniform_(-2, 2).to(device)
    t_f = torch.FloatTensor(initial_size_internal, 1).uniform_(0, T_max).to(device)
    X_f_train = torch.cat((x_f, t_f), dim=1).requires_grad_(True)

    
    # 修改后的周期性边界条件：h(-2, t) = h(2, t)
    # 生成左右边界点（x固定为-2和2，t随机）
    t_b = torch.FloatTensor(initial_size_boundary, 1).uniform_(0, T_max).to(device)
    x_left = -2.0 * torch.ones(initial_size_boundary, 1).to(device)  # 左边界x=-2
    x_right = 2.0 * torch.ones(initial_size_boundary, 1).to(device)   # 右边界x=2
    
    # 构建边界训练点
    X_b_train_left = torch.cat((x_left, t_b), dim=1).requires_grad_(True)
    X_b_train_right = torch.cat((x_right, t_b), dim=1).requires_grad_(True)
    
    
    # 初始化网络
    net = Net([2, 10, 10, 10,10,10,1]).to(device)
    loss_fn = nn.MSELoss(reduction='mean').to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_history = []

    # 训练循环
    # 训练循环
    train_start_time = time.time()
    n = 0
    err = residual_threshold + 1
    
    while n < max_iterations and err > residual_threshold:
        print(f"\n=== Adaptive Iteration {n+1}/{max_iterations} ===")
    
        # 构建训练集
        if n == 0:
            X_f_batch = X_f_train.clone().detach().requires_grad_(True).to(device)
        else:
            X_f_batch = build_dataset(X_f_batch, X_add).clone().detach().requires_grad_(True).to(device)
    
        # 内循环训练
        n_r = 0
        loss_r = float('inf')
        while n_r < inner_max_steps and loss_r >= inner_loss_threshold:
            optimizer.zero_grad()
    
            # 计算PDE残差
            u_pred = X_f_batch[:, 1:2]*net(X_f_batch)
            u_x = torch.autograd.grad(u_pred, X_f_batch, 
                                     grad_outputs=torch.ones_like(u_pred),
                                     create_graph=True)[0][:, 0:1]
            u_t = torch.autograd.grad(u_pred, X_f_batch,
                                     grad_outputs=torch.ones_like(u_pred),
                                     create_graph=True)[0][:, 1:2]
    
            phi_neg = phi_minus(X_f_batch[:, 0:1], u_pred)
            phi_pos = phi_plus(X_f_batch[:, 0:1], u_pred)
            residual = u_t - (u_x - 0.5) * (phi_neg + phi_pos)
            eq_loss = loss_fn(residual, torch.zeros_like(residual))
      
            # 边界条件损失（更新为左右边界比较）
            h_left = X_b_train_left[:, 1:2] * net(X_b_train_left)  # h(-2, t)
            h_right = X_b_train_right[:, 1:2] * net(X_b_train_right)  # h(2, t)
            bc_loss = loss_fn(h_left, h_right)  # 强制h(-2,t)=h(2,t)
    
            
            total_loss = eq_loss + bc_loss
            total_loss.backward()
            optimizer.step()
    
            loss_history.append(total_loss.item())
            loss_r = total_loss.item()  # 更新当前损失值
    
            if n_r % 500 == 0:
                print(f"  Step {n_r}/{inner_max_steps}, Loss: {total_loss.item():.3e}")
            
            n_r += 1  # 递增循环计数器

    

        # 残差评估
        # 生成二维随机采样点，范围分别为 [-2, 2] 和 [0, T_max]
        X_f_test = torch.rand(20000, 2).to(device)
        X_f_test[:, 0] = X_f_test[:, 0] * 4 - 2  # 将第一列调整到 [-2, 2]
        X_f_test[:, 1] = X_f_test[:, 1] * T_max  # 将第二列调整到 [0, T_max]
        X_f_test = X_f_test.requires_grad_(True)
        
        
        
        # 前向传播（保持梯度追踪）
        u_test = X_f_test[:, 1:2] *net(X_f_test)
        
        # 计算空间导数
        u_x = torch.autograd.grad(
            outputs=u_test,
            inputs=X_f_test,
            grad_outputs=torch.ones_like(u_test),
            create_graph=False,
            retain_graph=True
        )[0][:, 0:1]
        
        # 计算时间导数
        u_t = torch.autograd.grad(
            outputs=u_test,
            inputs=X_f_test,
            grad_outputs=torch.ones_like(u_test),
            create_graph=False,
            retain_graph=True
        )[0][:, 1:2]
        
        # 后续计算
        phi_neg = phi_minus(X_f_test[:, 0:1], u_test)
        phi_pos = phi_plus(X_f_test[:, 0:1], u_test)
        residual = (u_t - (u_x - 0.5)*(phi_neg + phi_pos)).detach().cpu().numpy()

        mean_residual = np.mean(np.abs(residual))
        print(f"  Mean Residual: {mean_residual:.3e}")
        
        if mean_residual < residual_threshold:
            print("  残差已低于阈值，停止训练。")
            break

        # 添加新采样点
        X_add = add_data(X_f_test.detach().cpu().numpy(), residual, add_k=add_k)
        print(f"  添加了 {add_k} 个新的采样点。")
        
        n += 1
        print(f"  当前训练数据点数量: {X_f_batch.shape[0]}")
        

    # 训练结果
    newtime=time.time()-train_start_time
    print(f"\nTraining Completed in {time.time()-train_start_time:.1f}s")
    
    print(f"Final Loss: {loss_history[-1]:.3e}")
    
    np.savez('2d_DAE_RAR_loss_history.npz', loss_history)
    # 可视化训练损失
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss History with Fixed Sampling')
    plt.grid(True)
    plt.show()

    # 统一可视化参数设置
    plt.rcParams.update({
        'font.size': 24,          # 全局字体大小
        'axes.labelsize': 24,     # 坐标轴标签大小
        'axes.titlesize': 30,     # 标题大小
        'xtick.labelsize': 25,    # x轴刻度大小
        'ytick.labelsize': 25,    # y轴刻度大小
        'axes.linewidth': 2,      # 坐标轴线宽
        'lines.linewidth': 8,     # 曲线线宽
        'legend.fontsize': 22     # 图例大小
    })

    # 自定义颜色映射
    custom_cmap = plt.cm.viridis  # 改用更科学的颜色映射


    def plot_3d_surface(X, Y, Z, title, cmap='plasma', elev=10, azim=-60,zlim_factor=0.2):
        # 增大画布宽度，同时保持高度
        fig = plt.figure(figsize=(18, 13))  # 宽度从16增大到18
        
        ax = fig.add_subplot(111, projection='3d')
        
        # 曲面设置
        surf = ax.plot_surface(X, Y, Z, 
                              cmap=cmap,
                              rstride=1, 
                              cstride=1,
                              antialiased=True,
                              edgecolor='none',
                              alpha=0.95)
        
        # ========== 坐标轴标签优化 ==========
        ax.set_xlabel('x', labelpad=25, fontsize=28, fontweight='bold')
        ax.set_ylabel('y', labelpad=25, fontsize=28, fontweight='bold')
        ax.set_zlabel('u', labelpad=20, fontsize=25, fontweight='bold')  # 增加z轴间距
        
        # 刻度设置
        ax.tick_params(axis='both', 
                      pad=12,          # 刻度标签与轴线的距离
                      labelsize=24,    # 刻度字体大小
                      width=1.5,       # 刻度线宽度
                      length=6)        # 刻度线长度
        
        # 视角设置
        ax.view_init(elev=elev, azim=azim)
        
        # ========== 关键修改：颜色条优化 ==========
        cbar = fig.colorbar(surf, ax=ax,
                           shrink=0.6,    # 缩短色条长度(原0.8)
                           aspect=12,     # 调整宽高比(原15)
                           pad=0.05,      # 增加与主图的间距(原0.01)
                           location='right')
        
        cbar.ax.tick_params(labelsize=24, width=1.5, length=5)
        # cbar.set_label('Value', fontsize=24, labelpad=15)  # 添加颜色条标签
        
        # ========== 画布布局优化 ==========
        plt.subplots_adjust(
            left=0.05,    # 左侧边距
            right=0.82,   # 右侧边距(原0.85)
            bottom=0.1,
            top=0.95
        )
        
        # # 背景优化
        # ax.xaxis.pane.set_edgecolor('gray')
        # ax.yaxis.pane.set_edgecolor('gray')
        # ax.zaxis.pane.set_edgecolor('gray')
        # ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # 移除轴面板边框颜色（可选）
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')

        # 关闭网格线
        ax.grid(False)
        plt.title(title, y=1.05, fontsize=35, fontweight='bold')
        return fig
    


    # 生成新的网格数据
    x_step = 0.1
    y_step = 0.1

    x_range = np.linspace(-2, 2, int((2 - (-2)) / x_step) + 1)
    y_range = np.linspace(-2, 2, int((2 - (-2)) / y_step) + 1)
    t_value = 0.5


    

    data = []
    for x in x_range:
        for y in y_range:
            data.append([x, y, t_value])

    df = pd.DataFrame(data, columns=['x', 'y', 't'])
    
    
    #读取和存储第一个数据

    # 读取真解数据1
    file_path_1 = '2d_39_1newdatamu2_ae_01_01_01.txt'
    true_data_1 = pd.read_csv(file_path_1, delim_whitespace=True, header=None, names=['u', 'x', 'y','t'])
    true_data_t_0_7_1 = true_data_1[true_data_1['t'] == t_value]
    true_data_t_0_7_1 = true_data_t_0_7_1.sort_values(by=['x', 'y'])
    # 提取真解的 u, x, y 数据
    true_u_t_0_7_1 = true_data_t_0_7_1['u'].values
    x_true_t_0_7 = true_data_t_0_7_1['x'].values
    y_true_t_0_7 = true_data_t_0_7_1['y'].values
    
    
    # 读取真解数据2
    file_path_2 = '2d_39_1newdatamu3_ae_01_01_01.txt'
    true_data_2 = pd.read_csv(file_path_2, delim_whitespace=True, header=None, names=['u', 'x', 'y','t'])
    true_data_t_0_7_2 = true_data_2[true_data_2['t'] == t_value]
    true_data_t_0_7_2 = true_data_t_0_7_2.sort_values(by=['x', 'y'])
    # 提取真解的 u, x, y 数据
    true_u_t_0_7_2 = true_data_t_0_7_2['u'].values
    
    
    # 读取真解数据3
    file_path_3 = '2d_39_1newdatamu4_ae_01_01_01.txt'
    true_data_3 = pd.read_csv(file_path_3, delim_whitespace=True, header=None, names=['u', 'x', 'y','t'])
    true_data_t_0_7_3 = true_data_3[true_data_3['t'] == t_value]
    true_data_t_0_7_3 = true_data_t_0_7_3.sort_values(by=['x', 'y'])
    # 提取真解的 u, x, y 数据
    true_u_t_0_7_3 = true_data_t_0_7_3['u'].values
    
    
    
    
    

    # 计算近似解 U01
    def compute_U0_1(x, y, h0, h0_x, phi_m, phi_p, phi_m_xy, phi_p_xy, k=2.0, mu=0.01):
        if y <= h0:
            U0 = phi_m_xy + (phi_p - phi_m) / (torch.exp((h0 - y) * (phi_p - phi_m) * (1 - k * h0_x) / (2 * mu)) + 1)
        else:
            U0 = phi_p_xy + (phi_m - phi_p) / (torch.exp((h0 - y) * (phi_m - phi_p) * (1 - k * h0_x) / (2 * mu)) + 1)
        return U0
    
    
    # 计算近似解 U02
    def compute_U0_2(x, y, h0, h0_x, phi_m, phi_p, phi_m_xy, phi_p_xy, k=2.0, mu=0.001):
        if y <= h0:
            U0 = phi_m_xy + (phi_p - phi_m) / (torch.exp((h0 - y) * (phi_p - phi_m) * (1 - k * h0_x) / (2 * mu)) + 1)
        else:
            U0 = phi_p_xy + (phi_m - phi_p) / (torch.exp((h0 - y) * (phi_m - phi_p) * (1 - k * h0_x) / (2 * mu)) + 1)
        return U0
    
    # 计算近似解 U03
    def compute_U0_3(x, y, h0, h0_x, phi_m, phi_p, phi_m_xy, phi_p_xy, k=2.0, mu=0.0001):
        if y <= h0:
            U0 = phi_m_xy + (phi_p - phi_m) / (torch.exp((h0 - y) * (phi_p - phi_m) * (1 - k * h0_x) / (2 * mu)) + 1)
        else:
            U0 = phi_p_xy + (phi_m - phi_p) / (torch.exp((h0 - y) * (phi_m - phi_p) * (1 - k * h0_x) / (2 * mu)) + 1)
        return U0

    # 提取预测解的 x, y, t 数据
    x_t_0_7 = df['x'].values
    y_t_0_7 = df['y'].values
    t_t_0_7 = df['t'].values

    # 初始化预测解数组
    U0_t_0_7_1 = np.zeros_like(x_t_0_7)
    
    # 初始化预测解数组
    U0_t_0_7_2 = np.zeros_like(x_t_0_7)
    
    # 初始化预测解数组
    U0_t_0_7_3 = np.zeros_like(x_t_0_7)

    # 设置模型为评估模式
    net.eval()

    # 计算预测解
    start_time = time.time()
    for i in range(len(x_t_0_7)):
        x_val = torch.tensor([[x_t_0_7[i]]], dtype=torch.float32, requires_grad=True, device=device)
        y_val = torch.tensor([[y_t_0_7[i]]], dtype=torch.float32, device=device)
        t_val = torch.tensor([[t_t_0_7[i]]], dtype=torch.float32, requires_grad=True, device=device)
        input_val = torch.cat([x_val, t_val], dim=1)

        h0_val = t_val*net(input_val)

        # 计算 h0 对 x 的梯度
        h0_x = torch.autograd.grad(h0_val, x_val, grad_outputs=torch.ones_like(h0_val), create_graph=True)[0]

        phi_m_x_y = phi_minus(x_val, y_val)
        phi_p_x_y = phi_plus(x_val, y_val)

        phi_m_val = phi_minus(x_val, h0_val)
        phi_p_val = phi_plus(x_val, h0_val)

        U0_val_1 = compute_U0_1(x_val, y_val, h0_val, h0_x, phi_m_val, phi_p_val, phi_m_x_y, phi_p_x_y)
        U0_t_0_7_1[i] = U0_val_1.item()
        
        U0_val_2 = compute_U0_2(x_val, y_val, h0_val, h0_x, phi_m_val, phi_p_val, phi_m_x_y, phi_p_x_y)
        U0_t_0_7_2[i] = U0_val_2.item()
        
        U0_val_3 = compute_U0_3(x_val, y_val, h0_val, h0_x, phi_m_val, phi_p_val, phi_m_x_y, phi_p_x_y)
        U0_t_0_7_3[i] = U0_val_3.item()

    end_time = time.time()
    print(f'Prediction time: {end_time - start_time} seconds')

    
    # 计算相对 L2 误差
    def relative_L2_error(true_values, predicted_values):
        return np.linalg.norm(true_values - predicted_values) / np.linalg.norm(true_values)

    error_1 = relative_L2_error(true_u_t_0_7_1, U0_t_0_7_1)
    print(f'Relative L2 Error: {error_1}')
    
    error_2 = relative_L2_error(true_u_t_0_7_2, U0_t_0_7_2)
    print(f'Relative L2 Error: {error_2}')


    error_3 = relative_L2_error(true_u_t_0_7_3, U0_t_0_7_3)
    print(f'Relative L2 Error: {error_3}')


    
    # 绘制误差等值线图
    error_values_1 = np.abs(U0_t_0_7_1 - true_u_t_0_7_1)
    # 打印最大的绝对误差
    max_absolute_error_1 = np.max(error_values_1)
    print(f"Maximum absolute error_1: {max_absolute_error_1}")
    
    # 绘制误差等值线图
    error_values_2 = np.abs(U0_t_0_7_2 - true_u_t_0_7_2)
    # 打印最大的绝对误差
    max_absolute_error_2 = np.max(error_values_2)
    print(f"Maximum absolute error_2: {max_absolute_error_2}")
    
    # 绘制误差等值线图
    error_values_3 = np.abs(U0_t_0_7_3 - true_u_t_0_7_3)
    # 打印最大的绝对误差
    max_absolute_error_3 = np.max(error_values_3)
    print(f"Maximum absolute error_3: {max_absolute_error_3}")


    def plot_U0(x, y, U0,file):
        # 转换为网格形式
        X_unique = np.unique(x)
        Y_unique = np.unique(y)
        X, Y = np.meshgrid(X_unique, Y_unique)
        Z = U0.reshape(len(Y_unique), len(X_unique))

        # 使用统一绘图函数
        fig = plot_3d_surface(X, Y, Z, 
                              "Predicted Solution of DAE with RAR (t=0.5)",
                              cmap='plasma',
                              elev=25, 
                              azim=135)
        
        # 保存和显示
        plt.tight_layout()
        plt.savefig(file, dpi=300, bbox_inches='tight')
        plt.show()

    # 修改后的真解绘制函数
    def plot_true_solution(x, y, u,file):
        # 转换为网格形式
        X_unique = np.unique(x)
        Y_unique = np.unique(y)
        X, Y = np.meshgrid(X_unique, Y_unique)
        Z = u.reshape(len(Y_unique), len(X_unique))

        # 使用统一绘图函数
        fig = plot_3d_surface(X, Y, Z, 
                              "Reference Solution (t=0.5)",
                              cmap='plasma',
                              elev=25,
                              azim=135)
        
        # 保存和显示
        plt.tight_layout()
        plt.savefig(file, dpi=300, bbox_inches='tight')
        plt.show()

    # 绘制结果
    plot_U0(x_t_0_7, y_t_0_7, U0_t_0_7_1,'2d_DAE_RAR_mu2_solution_t_0_5.eps')
    plot_true_solution(x_true_t_0_7, y_true_t_0_7, true_u_t_0_7_1,'2d_RAR_mu2_true_solution_t_0_5.eps')
    
    # 绘制结果
    plot_U0(x_t_0_7, y_t_0_7, U0_t_0_7_2,'2d_DAE_RAR_mu3_solution_t_0_5.eps')
    plot_true_solution(x_true_t_0_7, y_true_t_0_7, true_u_t_0_7_2,'2d_RAR_mu3_true_solution_t_0_5.eps')
    
    # 绘制结果
    plot_U0(x_t_0_7, y_t_0_7, U0_t_0_7_3,'2d_DAE_RAR_mu4_solution_t_0_5.eps')
    plot_true_solution(x_true_t_0_7, y_true_t_0_7, true_u_t_0_7_3,'2d_RAR_mu4_true_solution_t_0_5.eps')

    
   
    def plot_error(x, y, error,file):
        # 转换为网格形式
        X_unique = np.unique(x)
        Y_unique = np.unique(y)
        X, Y = np.meshgrid(X_unique, Y_unique)
        Z = error.reshape(len(Y_unique), len(X_unique))

        # ================= 误差分布图设置 =================
        plt.rcParams.update({'font.size': 24})  # 全局字体设置
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        
        # 使用plasma配色方案
        error_contour = ax.contourf(X, Y, Z, 
                                    levels=50, 
                                    cmap='plasma',
                                    antialiased=True)
        
        # ========== 坐标轴优化设置 ==========
        ax.set_xlabel('x', fontsize=28, labelpad=20, fontweight='bold')
        ax.set_ylabel('y', fontsize=28, labelpad=20, fontweight='bold')
        
        # 刻度参数调整
        ax.tick_params(axis='both',
                      which='major',
                      labelsize=24,
                      width=2,
                      length=6,
                      direction='inout')
        
        # ========== 颜色条专业设置 ==========
        cbar = fig.colorbar(error_contour, ax=ax, 
                            fraction=0.046, 
                            pad=0.04)
        cbar.ax.tick_params(labelsize=24, 
                            width=2, 
                            length=6,
                            direction='inout')
        
        # ========== 标题和布局优化 ==========
        plt.title('Absolute Error of DAE with RAR (t=0.5)', 
                  fontsize=28, 
                  pad=25,
                  fontweight='bold')
        
        plt.tight_layout(pad=3.0)
        plt.savefig(file, 
                    dpi=300, 
                    bbox_inches='tight',
                    transparent=True)
        plt.show()

    # 使用示例
    plot_error(x_t_0_7, y_t_0_7, error_values_1,'2d_DAE_RAR_mu2_error_contour.eps')
    
    # 使用示例
    plot_error(x_t_0_7, y_t_0_7, error_values_2,'2d_DAE_RAR_mu3_error_contour.eps')
    
    # 使用示例
    plot_error(x_t_0_7, y_t_0_7, error_values_3,'2d_DAE_RAR_mu4_error_contour.eps')
    
    
    
    print(f'Relative L2 Error: {error_1}')
    print(f'Relative L2 Error: {error_2}')
    print(f'Relative L2 Error: {error_3}')

    print(f"Maximum absolute error_1: {max_absolute_error_1}")
    print(f"Maximum absolute error_2: {max_absolute_error_2}")
    print(f"Maximum absolute error_3: {max_absolute_error_3}")

    print(newtime)
    print(f"Final Loss: {loss_history[-1]:.3e}")

    


if __name__ == '__main__':
    main()