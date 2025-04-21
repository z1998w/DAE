# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 21:23:40 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 16:31:54 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:24:00 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:43:58 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 13:27:41 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:06:16 2024

@author: zhuqi
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pyDOE import lhs
import torch.nn.init as init

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from matplotlib import cm

# 设置随机种子保证可复现性
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            self.layers.append(layer)
        self.apply(self._initialize_weights)
        self.activation = nn.Tanh()

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, y_t):
        for layer in self.layers[:-1]:
            y_t = self.activation(layer(y_t))
        raw_output = self.layers[-1](y_t)
        h0 = raw_output  # 精确确保h0(x,0)=0
        return h0

def phi_minus(x, y):
    pi = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
    expr = (
        16 * pi +
        2 * pi * torch.cos(pi * (-x + y) / 4) +
        pi * x * torch.cos(pi * (-x + y) / 4) -
        2 * torch.sin(pi * (-4 - x + y) / 4) +
        2 * torch.sin(pi * (x + y) / 4)
    )
    return - (1 / torch.sqrt(pi)) * torch.sqrt(expr)

def phi_plus(x, y):
    pi = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
    expr = (
        4 * pi -
        2 * pi * torch.cos(pi * (-x + y) / 4) +
        pi * x * torch.cos(pi * (-x + y) / 4) -
        2 * torch.sin(pi * (4 - x + y) / 4) +
        2 * torch.sin(pi * (x + y) / 4)
    )
    return (1 / torch.sqrt(pi)) * torch.sqrt(expr)

def pde_loss(net, y, t):
    y.requires_grad_(True)
    t.requires_grad_(True)
    y_t = torch.cat([y, t], dim=1)
    h0 = t*net(y_t)
    
    # 计算导数
    dh0_dt = grad(h0, t, grad_outputs=torch.ones_like(h0), create_graph=True)[0]
    dh0_dy = grad(h0, y, grad_outputs=torch.ones_like(h0), create_graph=True)[0]
    
    # 计算PDE残差
    phi_m = phi_minus(h0, y)
    phi_p = phi_plus(h0, y)
    residual = dh0_dt - (dh0_dy - 0.5) * (phi_m + phi_p)
    return torch.mean(residual**2)

def periodic_boundary_condition_loss(net, t):#用这个误差到4了
    # 定义边界点 x = -2 和 x = 2
    y_left = torch.full_like(t, -2.0)  # x = -2
    y_right = torch.full_like(t, 2.0)  # x = 2
    
    # 计算 h(-2, t) 和 h(2, t)
    h_left = t*net(torch.cat([y_left, t], dim=1))
    h_right = t*net(torch.cat([y_right, t], dim=1))
    
    # 计算边界条件损失
    return torch.mean((h_left - h_right)**2)
# 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
net = PINN([2, 10, 10, 10,10,10,1]).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)

def train(net, epochs):
    loss_history = []
    
    
    # 内部点（2000个样本）
    y_internal = (torch.rand(2000, 1) * 4 - 2).to(device)  # x ∈ [-2, 2)
    t_internal = torch.rand(2000, 1).to(device)            # t ∈ [0, 1)
       
    
    t_boundary = torch.rand(1000, 1).to(device)
    
    start_time = time.time()
    for epoch in range(epochs):
        # 使用固定采样点计算损失
        loss_pde = pde_loss(net, y_internal, t_internal)
        loss_bc = periodic_boundary_condition_loss(net, t_boundary)
        total_loss = loss_pde + loss_bc

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())
                                    
        # 每1000轮输出一次
        if epoch % 1000 == 0 or epoch == epochs-1:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch:5d} | Loss: {total_loss.item():.3e} | Time: {elapsed:.1f}s')
    print(f'\nFinal Loss after {epochs} epochs: {loss_history[-1]:.4e}')
    print(f'Total training time: {time.time()-start_time:.1f}s')
    
    newtime=time.time()-start_time
    
    np.savez('2d_new_loss_history.npz', loss_history)
    
    return loss_history,newtime

loss_history,newtime = train(net, epochs=15000)



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

y = np.linspace(-2, 2, 100)
t = np.linspace(0, 1, 100)
Y, T = np.meshgrid(y, t)
    
    # 转换为神经网络输入格式
Y_flat = Y.reshape(-1, 1)
T_flat = T.reshape(-1, 1)
    
    # 转换为Tensor并预测
with torch.no_grad():
        # 组合时空坐标 (N, 2)
    YT_tensor = torch.tensor(np.hstack((Y_flat, T_flat)), 
                                dtype=torch.float32).to(device)
        
    # 将 T_flat 转换为PyTorch张量
    T_flat_tensor = torch.tensor(T_flat, dtype=torch.float32).to(device)
    
   
    h0_pred = T_flat_tensor * net(YT_tensor)
        # 转换为numpy数组并重塑
    h0_pred = h0_pred.cpu().numpy().reshape(Y.shape)  # (100, 100)
        
def plot_h0_surface(X, Y, Z, title, xlabel, ylabel, zlabel, cmap='plasma'):
    # 增大画布宽度，同时保持高度
    fig = plt.figure(figsize=(14, 9))  # 宽度从16增大到18
    
    ax = fig.add_subplot(111)
    
    # 等值线图设置
    contour = ax.contourf(X, Y, Z, 
                          cmap=cmap,
                          levels=50,  # 等值线的数量
                          alpha=0.95)
    
    # ========== 坐标轴标签优化 ==========
    ax.set_xlabel(xlabel, labelpad=25, fontsize=22, fontweight='bold')
    ax.set_ylabel(ylabel, labelpad=25, fontsize=22, fontweight='bold')
    
    # 刻度设置
    ax.tick_params(axis='both', 
                  pad=12,          # 刻度标签与轴线的距离
                  labelsize=24,    # 刻度字体大小
                  width=1.5,       # 刻度线宽度
                  length=6)        # 刻度线长度
    
    # ========== 关键修改：颜色条优化 ==========
    cbar = fig.colorbar(contour, ax=ax,
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
    
    # 背景优化
    ax.grid(False)
    
    plt.title(title, y=1.05, fontsize=30, fontweight='bold')
    return fig        

fig_pred_3d = plot_h0_surface(Y, T, h0_pred, 
                              "Predicted $h_0(y,t)$",xlabel="y", ylabel="t", zlabel="h")
plt.savefig('2d_DAE_h0_3d_surface.eps', dpi=300, bbox_inches='tight')


def plot_3d_surface(X, Y, Z, title, xlabel, ylabel, zlabel, cmap='plasma', elev=10, azim=-60,zlim_factor=0.2):
    # 增大画布宽度，同时保持高度
    fig = plt.figure(figsize=(18, 13))  # 宽度从16增大到18
    
    ax = fig.add_subplot(111, projection='3d')
    
    # 曲面设置
    surf = ax.plot_surface(X, Y, Z, 
                          cmap=cmap,
                          rstride=1,          # 必须设为1
                          cstride=1,          # 必须设为1
                          edgecolor='none',   # 边缘透明
                          linewidth=0,        # 彻底消除线宽
                          antialiased=False,  # 关闭抗锯齿
                          alpha=0.95)
    
#     
    # ========== 坐标轴标签优化 ==========
    ax.set_xlabel(xlabel, labelpad=25, fontsize=28, fontweight='bold')
    ax.set_ylabel(ylabel, labelpad=25, fontsize=28, fontweight='bold')
    ax.set_zlabel(zlabel, labelpad=20, fontsize=25, fontweight='bold')  # 增加z轴间距
    
    
    
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
    
   
# 设置 3D 轴面板背景透明
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
t_value = 0.2






data = []
for y in y_range:
    for x in x_range:
        data.append([x, y, t_value])

df = pd.DataFrame(data, columns=['x', 'y', 't'])

file_path_1 = '2d_0410newdatamu2_ae_01_01_01.txt'
true_data_1 = pd.read_csv(file_path_1, delim_whitespace=True, header=None, names=['u', 'x', 'y','t'])
true_data_t_0_7_1 = true_data_1[true_data_1['t'] == t_value]
# true_data_t_0_7_1 = true_data_t_0_7_1.sort_values(by=['x', 'y'])
# 提取真解的 u, x, y 数据
true_u_t_0_7_1 = true_data_t_0_7_1['u'].values

x_true_t_0_7 = true_data_t_0_7_1['x'].values
y_true_t_0_7 = true_data_t_0_7_1['y'].values


# 读取真解数据2
file_path_2 = '2d_0410newdatamu3_ae_01_01_01.txt'
true_data_2 = pd.read_csv(file_path_2, delim_whitespace=True, header=None, names=['u', 'x', 'y','t'])
true_data_t_0_7_2 = true_data_2[true_data_2['t'] == t_value]
# true_data_t_0_7_2 = true_data_t_0_7_2.sort_values(by=['x', 'y'])
# 提取真解的 u, x, y 数据
true_u_t_0_7_2 = true_data_t_0_7_2['u'].values


# 读取真解数据3
file_path_3 = '2d_0410newdatamu4_ae_01_01_01.txt'
true_data_3 = pd.read_csv(file_path_3, delim_whitespace=True, header=None, names=['u', 'x', 'y','t'])
true_data_t_0_7_3 = true_data_3[true_data_3['t'] == t_value]
# true_data_t_0_7_3 = true_data_t_0_7_3.sort_values(by=['x', 'y'])
# 提取真解的 u, x, y 数据
true_u_t_0_7_3 = true_data_t_0_7_3['u'].values

# 计算近似解 U01
def compute_U0_1(x, y, h0, h0_y, phi_m, phi_p, phi_m_xy, phi_p_xy, mu=0.01):
    if x <= h0:
        U0 = phi_m_xy + (phi_p - phi_m) / (torch.exp((h0 - x) * (phi_p - phi_m) * (1 - h0_y) / (2 * mu)) + 1)
    else:
        U0 = phi_p_xy + (phi_m - phi_p) / (torch.exp((h0 - x) * (phi_m - phi_p) * (1 - h0_y) / (2 * mu)) + 1)
    return U0


# 计算近似解 U02
def compute_U0_2(x, y, h0, h0_y, phi_m, phi_p, phi_m_xy, phi_p_xy, mu=0.001):
    if x <= h0:
        U0 = phi_m_xy + (phi_p - phi_m) / (torch.exp((h0 - x) * (phi_p - phi_m) * (1 - h0_y) / (2 * mu)) + 1)
    else:
        U0 = phi_p_xy + (phi_m - phi_p) / (torch.exp((h0 - x) * (phi_m - phi_p) * (1 - h0_y) / (2 * mu)) + 1)
    return U0

# 计算近似解 U03
def compute_U0_3(x, y, h0, h0_y, phi_m, phi_p, phi_m_xy, phi_p_xy, mu=0.0001):
    if x <= h0:
        U0 = phi_m_xy + (phi_p - phi_m) / (torch.exp((h0 - x) * (phi_p - phi_m) * (1 - h0_y) / (2 * mu)) + 1)
    else:
        U0 = phi_p_xy + (phi_m - phi_p) / (torch.exp((h0 - x) * (phi_m - phi_p) * (1 - h0_y) / (2 * mu)) + 1)
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
for i in range(len(y_t_0_7)):
    x_val = torch.tensor([[x_t_0_7[i]]], dtype=torch.float32, device=device)
    y_val = torch.tensor([[y_t_0_7[i]]], dtype=torch.float32, requires_grad=True, device=device)
    t_val = torch.tensor([[t_t_0_7[i]]], dtype=torch.float32, requires_grad=True, device=device)
    input_val = torch.cat([y_val, t_val], dim=1)

    h0_val = t_val*net(input_val)

    # 计算 h0 对 x 的梯度
    h0_y = torch.autograd.grad(h0_val, y_val, grad_outputs=torch.ones_like(h0_val), create_graph=True)[0]

    phi_m_x_y = phi_minus(x_val, y_val)
    phi_p_x_y = phi_plus(x_val, y_val)

    phi_m_val = phi_minus(h0_val,y_val)
    phi_p_val = phi_plus(h0_val,y_val)

    U0_val_1 = compute_U0_1(x_val, y_val, h0_val, h0_y, phi_m_val, phi_p_val, phi_m_x_y, phi_p_x_y)
    U0_t_0_7_1[i] = U0_val_1.item()
    
    U0_val_2 = compute_U0_2(x_val, y_val, h0_val, h0_y, phi_m_val, phi_p_val, phi_m_x_y, phi_p_x_y)
    U0_t_0_7_2[i] = U0_val_2.item()
    
    U0_val_3 = compute_U0_3(x_val, y_val, h0_val, h0_y, phi_m_val, phi_p_val, phi_m_x_y, phi_p_x_y)
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
                          "Predicted Solution of DAE (t=0.2)",xlabel="x", ylabel="y", zlabel="u",
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
                          "Reference Solution (t=0.2)",xlabel="x", ylabel="y", zlabel="u",
                          cmap='plasma',
                          elev=25,
                          azim=135)
    
    # 保存和显示
    plt.tight_layout()
    plt.savefig(file, dpi=300, bbox_inches='tight')
    plt.show()

# 绘制结果
plot_U0(x_t_0_7, y_t_0_7, U0_t_0_7_1,'2d_DAE_mu2_solution_t_0_2.eps')
plot_true_solution(x_true_t_0_7, y_true_t_0_7, true_u_t_0_7_1,'2d_mu2_true_solution_t_0_2.eps')

# 绘制结果
plot_U0(x_t_0_7, y_t_0_7, U0_t_0_7_2,'2d_DAE_mu3_solution_t_0_2.eps')
plot_true_solution(x_true_t_0_7, y_true_t_0_7, true_u_t_0_7_2,'2d_mu3_true_solution_t_0_2.eps')

# 绘制结果
plot_U0(x_t_0_7, y_t_0_7, U0_t_0_7_3,'2d_DAE_mu4_solution_t_0_2.eps')
plot_true_solution(x_true_t_0_7, y_true_t_0_7, true_u_t_0_7_3,'2d_mu4_true_solution_t_0_2.eps')



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
    plt.title('Absolute Error of DAE (t=0.2)', 
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
plot_error(x_t_0_7, y_t_0_7, error_values_1,'2d_DAE_mu2_error_contour.eps')

# 使用示例
plot_error(x_t_0_7, y_t_0_7, error_values_2,'2d_DAE_mu3_error_contour.eps')

# 使用示例
plot_error(x_t_0_7, y_t_0_7, error_values_3,'2d_DAE_mu4_error_contour.eps')


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

print(f'\nFinal Loss after epochs: {loss_history[-1]:.4e}')
print(newtime)