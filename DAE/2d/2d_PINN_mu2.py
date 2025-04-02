# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:29:24 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:49:06 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:05:31 2024

@author: zhuqi
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import grad
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pyDOE import lhs
# 定义PINN模型

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
                                                                            
class PINN(torch.nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = torch.nn.Linear(layers[i], layers[i + 1])
            torch.nn.init.xavier_normal_(layer.weight)  # 使用Xavier初始化权重
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)  # 初始化偏置为0
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x
def f_star(x, y):
    return torch.cos(np.pi * x / 4) * torch.cos(np.pi * y / 4)

def u_init(x, y):#先不用精初
    return 3 * torch.tanh(x + y / 0.01) - 1

def compute_loss(model, x_internal, y_internal, t_internal, x_boundary, y_boundary, t_boundary, x_initial, y_initial, t_initial):
    # 内部点的计算
    x_internal = x_internal.clone().detach().requires_grad_(True)
    y_internal = y_internal.clone().detach().requires_grad_(True)
    t_internal = t_internal.clone().detach().requires_grad_(True)
    
    u_internal = model(torch.cat([x_internal, y_internal, t_internal], dim=1))
    u_x = grad(u_internal.sum(), x_internal, create_graph=True)[0]
    u_y = grad(u_internal.sum(), y_internal, create_graph=True)[0]
    u_xx = grad(u_x.sum(), x_internal, create_graph=True)[0]
    u_yy = grad(u_y.sum(), y_internal, create_graph=True)[0]
    u_t = grad(u_internal.sum(), t_internal, create_graph=True)[0]

    f = f_star(x_internal, y_internal)
    pde_residual = 0.01 * (u_xx + u_yy) - u_t + u_internal * (2 * u_x + u_y) - f
    loss_pde = (pde_residual ** 2).mean()

    # 边界条件的计算
    # u = model(torch.cat([x_boundary, y_boundary, t_boundary], dim=1))
    # u_left = model(torch.cat([x_boundary + 4, y_boundary, t_boundary], dim=1))
    
    # 这里周期边界精确一下
    u_left = model(torch.cat([torch.ones_like(x_boundary) * (-2), y_boundary, t_boundary], dim=1))
    u_right = model(torch.cat([torch.ones_like(x_boundary) * (2), y_boundary, t_boundary], dim=1))


    u_bottom = model(torch.cat([x_boundary, torch.ones_like(x_boundary) * (-2), t_boundary], dim=1))
    u_top = model(torch.cat([x_boundary, torch.ones_like(x_boundary) * 2, t_boundary], dim=1))

    # loss_zhouqi = ((u - u_left) ** 2).mean()
    # loss_boundary = ((u_bottom - (-4)) ** 2+(u_top - 2) ** 2+(u - u_left) ** 2).mean()
    
    loss_boundary = ((u_bottom - (-4)) ** 2+(u_top - 2) ** 2+(u_left - u_right) ** 2).mean()
    #两种写法是一样的

    # 初始条件的计算
    u_initial = model(torch.cat([x_initial, y_initial, t_initial], dim=1))
    loss_init = ((u_initial.squeeze() - u_init(x_initial, y_initial).squeeze()) ** 2).mean()

    return loss_pde + loss_boundary + loss_init

def train(model, epochs, optimizer):
    # ========== 原始版本 (LHS采样) ==========
    # 内部点
    x_internal = (torch.rand(2000, 1) * 4 - 2).to(device) 
    y_internal = (torch.rand(2000, 1) * 4 - 2).to(device) 
    t_internal = (torch.rand(2000, 1)).to(device) 
    
    # 边界点
    x_boundary = (torch.rand(2000//3, 1) * 4 - 2).to(device)#这个数量需要改,这样边界用了1500个点
    y_boundary = (torch.rand(2000//3, 1) * 4 - 2).to(device)
    t_boundary = (torch.rand(2000//3, 1)).to(device)
    
    # 初始点
    x_initial = (torch.rand(2000, 1) * 4 - 2).to(device)    # [-2, 2)
    y_initial = (torch.rand(2000, 1) * 4 - 2).to(device)    # [-2, 2)
    t_initial = torch.zeros(2000, 1).to(device)             # t=0
    
    
    loss_history = []

    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss = compute_loss(model, x_internal, y_internal, t_internal,
                            x_boundary, y_boundary, t_boundary,
                            x_initial, y_initial, t_initial)
        loss.backward()
        optimizer.step()

        # 记录损失
        loss_history.append(loss.item())
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    end_time = time.time()
    
    print(f'\nFinal Loss after {epochs} epochs: {loss_history[-1]:.4e}')
    print(f'Total training time: {time.time()-start_time:.1f}s')
    
    newtime=time.time()-start_time
    
    #np.savez('2d_PINN_mu2_loss_history.npz', loss_history)
    
    return loss_history,newtime

    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')
model = PINN(layers=[3, 16, 16, 16,16,16,1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_history,newtime=train(model, 15000, optimizer)

# # 可视化训练损失
# plt.figure(figsize=(10, 6))
# plt.semilogy(loss_history)
# plt.xlabel('Epoch')
# plt.ylabel('Loss (log scale)')
# plt.title('Training Loss History with Fixed Sampling')
# plt.grid(True)
# plt.show()


# 生成新的网格数据
x_step = 0.1
y_step = 0.1

x_range = np.linspace(-2, 2, int((2 - (-2)) / x_step) + 1)
y_range = np.linspace(-2, 2, int((2 - (-2)) / y_step) + 1)
t_value = 0.5


# t_value =0.1

data = []
for x in x_range:
    for y in y_range:
        data.append([x, y, t_value])

df = pd.DataFrame(data, columns=['x', 'y', 't'])

# 读取真解数据
file_path = '2d_39_1newdatamu2_ae_01_01_01.txt'
true_data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['u', 'x', 'y','t'])

# 筛选真解数据中 t = t_value 的数据
true_data_t_0_7 = true_data[true_data['t'] == t_value]

# 确保真解数据和预测解数据的网格点一致
# 将真解数据按 x 和 y 排序，以便与预测解数据对齐
true_data_t_0_7 = true_data_t_0_7.sort_values(by=['x', 'y'])

# 提取真解的 u, x, y 数据
true_u_t_0_7 = true_data_t_0_7['u'].values
x_true_t_0_7 = true_data_t_0_7['x'].values
y_true_t_0_7 = true_data_t_0_7['y'].values

#预测解
df_t_0_7 = df[df['t'] == t_value]
x_t_0_7 = torch.tensor(df_t_0_7['x'].values, dtype=torch.float32).unsqueeze(1).to(device)
y_t_0_7 = torch.tensor(df_t_0_7['y'].values, dtype=torch.float32).unsqueeze(1).to(device)
t_t_0_7 = torch.tensor(df_t_0_7['t'].values, dtype=torch.float32).unsqueeze(1).to(device)


# start_time = time.time()
u_pred_t_0_7 = model(torch.cat([x_t_0_7, y_t_0_7, t_t_0_7], dim=1)).detach().cpu().numpy()
u_pred_t_0_7 = u_pred_t_0_7.reshape(-1)  # 形状从 (40401,1) 变为 (40401,)
# end_time = time.time()
# print(f'Prediction time: {end_time - start_time} seconds')
# 计算相对 L2 误差
def relative_L2_error(true_values, predicted_values):
    return np.linalg.norm(true_values - predicted_values) / np.linalg.norm(true_values)

error = relative_L2_error(true_u_t_0_7, u_pred_t_0_7)
print(f'Relative L2 Error: {error}')

# 绘制误差等值线图
error_values = np.abs(u_pred_t_0_7 - true_u_t_0_7)
# 打印最大的绝对误差
max_absolute_error = np.max(error_values)
print(f"Maximum absolute error: {max_absolute_error}")

# x_t_0_7 = df['x'].values
# y_t_0_7 = df['y'].values
# t_t_0_7 = df['t'].values

# # 统一可视化参数设置
# plt.rcParams.update({
#     'font.size': 24,          # 全局字体大小
#     'axes.labelsize': 24,     # 坐标轴标签大小
#     'axes.titlesize': 30,     # 标题大小
#     'xtick.labelsize': 25,    # x轴刻度大小
#     'ytick.labelsize': 25,    # y轴刻度大小
#     'axes.linewidth': 2,      # 坐标轴线宽
#     'lines.linewidth': 8,     # 曲线线宽
#     'legend.fontsize': 22     # 图例大小
# })

# # 自定义颜色映射
# custom_cmap = plt.cm.viridis  # 改用更科学的颜色映射

# def plot_3d_surface(X, Y, Z, title, cmap='plasma', elev=10, azim=-60,zlim_factor=0.2):
#     # 增大画布宽度，同时保持高度
#     fig = plt.figure(figsize=(18, 13))  # 宽度从16增大到18
    
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 曲面设置
#     surf = ax.plot_surface(X, Y, Z, 
#                           cmap=cmap,
#                           rstride=1,          # 必须设为1
#                           cstride=1,          # 必须设为1
#                           edgecolor='none',   # 边缘透明
#                           linewidth=0,        # 彻底消除线宽
#                           antialiased=False,  # 关闭抗锯齿
#                           alpha=0.95)
    
#     # ========== 坐标轴标签优化 ==========
#     ax.set_xlabel('x', labelpad=25, fontsize=28, fontweight='bold')
#     ax.set_ylabel('y', labelpad=25, fontsize=28, fontweight='bold')
#     ax.set_zlabel('u', labelpad=20, fontsize=25, fontweight='bold')  # 增加z轴间距
    
#     # 刻度设置
#     ax.tick_params(axis='both', 
#                   pad=12,          # 刻度标签与轴线的距离
#                   labelsize=24,    # 刻度字体大小
#                   width=1.5,       # 刻度线宽度
#                   length=6)        # 刻度线长度
    
#     # 视角设置
#     ax.view_init(elev=elev, azim=azim)
    
#     # ========== 关键修改：颜色条优化 ==========
#     cbar = fig.colorbar(surf, ax=ax,
#                        shrink=0.6,    # 缩短色条长度(原0.8)
#                        aspect=12,     # 调整宽高比(原15)
#                        pad=0.05,      # 增加与主图的间距(原0.01)
#                        location='right')
    
#     cbar.ax.tick_params(labelsize=24, width=1.5, length=5)
#     # cbar.set_label('Value', fontsize=24, labelpad=15)  # 添加颜色条标签
    
#     # ========== 画布布局优化 ==========
#     plt.subplots_adjust(
#         left=0.05,    # 左侧边距
#         right=0.82,   # 右侧边距(原0.85)
#         bottom=0.1,
#         top=0.95
#     )
    
#     # 背景优化
#     # ax.xaxis.pane.set_edgecolor('gray')
#     # ax.yaxis.pane.set_edgecolor('gray')
#     # ax.zaxis.pane.set_edgecolor('gray')
#     # ax.grid(False)
#     ax.xaxis.pane.fill = False
#     ax.yaxis.pane.fill = False
#     ax.zaxis.pane.fill = False

#         # 移除轴面板边框颜色（可选）
#     ax.xaxis.pane.set_edgecolor('black')
#     ax.yaxis.pane.set_edgecolor('black')
#     ax.zaxis.pane.set_edgecolor('black')

#         # 关闭网格线
#     ax.grid(False)
#     plt.title(title, y=1.05, fontsize=35, fontweight='bold')
#     return fig
# def plot_U0(x, y, U0):
#     # 转换为网格形式
#     X_unique = np.unique(x)
#     Y_unique = np.unique(y)
#     X, Y = np.meshgrid(X_unique, Y_unique)
#     Z = U0.reshape(len(Y_unique), len(X_unique))

#     # 使用统一绘图函数
#     fig = plot_3d_surface(X, Y, Z, 
#                           "Predicted Solution of PINN (t=0.5)",
#                           cmap='plasma',
#                           elev=25, 
#                           azim=135)
    
#     # 保存和显示
#     plt.tight_layout()
#     plt.savefig('2d_PINN_mu2_solution_t_0_5.eps', dpi=300, bbox_inches='tight')
#     plt.show()

# # 修改后的真解绘制函数
# def plot_true_solution(x, y, u):
#     # 转换为网格形式
#     X_unique = np.unique(x)
#     Y_unique = np.unique(y)
#     X, Y = np.meshgrid(X_unique, Y_unique)
#     Z = u.reshape(len(Y_unique), len(X_unique))

#     # 使用统一绘图函数
#     fig = plot_3d_surface(X, Y, Z, 
#                           "Reference Solution (t=0.5)",
#                           cmap='plasma',
#                           elev=25,
#                           azim=135)
    
#     # 保存和显示
#     plt.tight_layout()
#     plt.savefig('2d_PINN_mu2_true_solution_t_0_5.eps', dpi=300, bbox_inches='tight')
#     plt.show()

# # 绘制结果
# plot_U0(x_t_0_7, y_t_0_7, u_pred_t_0_7)
# plot_true_solution(x_true_t_0_7, y_true_t_0_7, true_u_t_0_7)


# # 
# def plot_error(x, y, error):
#     # 转换为网格形式
#     X_unique = np.unique(x)
#     Y_unique = np.unique(y)
#     X, Y = np.meshgrid(X_unique, Y_unique)
#     Z = error.reshape(len(Y_unique), len(X_unique))

#     # ================= 误差分布图设置 =================
#     plt.rcParams.update({'font.size': 24})  # 全局字体设置
    
#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(111)
    
#     # 使用plasma配色方案
#     error_contour = ax.contourf(X, Y, Z, 
#                                 levels=50, 
#                                 cmap='plasma',
#                                 antialiased=True)
    
#     # ========== 坐标轴优化设置 ==========
#     ax.set_xlabel('x', fontsize=28, labelpad=20, fontweight='bold')
#     ax.set_ylabel('y', fontsize=28, labelpad=20, fontweight='bold')
    
#     # 刻度参数调整
#     ax.tick_params(axis='both',
#                   which='major',
#                   labelsize=24,
#                   width=2,
#                   length=6,
#                   direction='inout')
    
#     # ========== 颜色条专业设置 ==========
#     cbar = fig.colorbar(error_contour, ax=ax, 
#                         fraction=0.046, 
#                         pad=0.04)
#     cbar.ax.tick_params(labelsize=24, 
#                         width=2, 
#                         length=6,
#                         direction='inout')
#     # cbar.set_label('Absolute Error', 
#     #               fontsize=24, 
#     #               labelpad=20,
#     #               rotation=270)
    
#     # ========== 标题和布局优化 ==========
#     plt.title('Absolute Error of PINN (t=0.5)', 
#               fontsize=28, 
#               pad=25,
#               fontweight='bold')
    
#     plt.tight_layout(pad=3.0)
#     plt.savefig('2d_PINN_mu2_error_contour.eps', 
#                 dpi=300, 
#                 bbox_inches='tight',
#                 transparent=True)
#     plt.show()

# # 使用示例
# plot_error(x_t_0_7, y_t_0_7, error_values)
# # 已知预测解u_pred_t_0_7和真解true_u_t_0_7的求解过程



# print(f'Relative L2 Error: {error}')

# print(f"Maximum absolute error: {max_absolute_error}")

# print(f'\nFinal Loss after epochs: {loss_history[-1]:.4e}')

# print(newtime)

