# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:26:13 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:17:56 2025

@author: zhuqi
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
import time
import matplotlib.pyplot as plt
from pyDOE import lhs
from torch.autograd import grad
# 确保已经安装 kaleido 库
import kaleido

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# 设置设备为 GPU，如果可用，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义神经网络
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

# 定义源函数
def source_function(x, y, z):
    return torch.cos(np.pi * x) * torch.cos(np.pi * y) * torch.cos(np.pi * z)

# 初始条件
def initial_condition(x, y, z):
    return 3 * torch.tanh(x + y + z / 0.005) - 1

def compute_loss(model, x_internal, y_internal, z_internal, t_internal, x_boundary, y_boundary, z_boundary, t_boundary, x_initial, y_initial, z_initial, t_initial):
    # 内部点的计算
    x_internal = x_internal.clone().detach().requires_grad_(True)
    y_internal = y_internal.clone().detach().requires_grad_(True)
    z_internal = z_internal.clone().detach().requires_grad_(True)
    t_internal = t_internal.clone().detach().requires_grad_(True)

    u_internal = model(torch.cat([x_internal, y_internal, z_internal, t_internal], dim=1))
    u_x = grad(u_internal.sum(), x_internal, create_graph=True)[0]
    u_y = grad(u_internal.sum(), y_internal, create_graph=True)[0]
    u_z = grad(u_internal.sum(), z_internal, create_graph=True)[0]
    u_xx = grad(u_x.sum(), x_internal, create_graph=True)[0]
    u_yy = grad(u_y.sum(), y_internal, create_graph=True)[0]
    u_zz = grad(u_z.sum(), z_internal, create_graph=True)[0]
    u_t = grad(u_internal.sum(), t_internal, create_graph=True)[0]

    f = source_function(x_internal, y_internal, z_internal)
    pde_residual = 0.05 * (u_xx + u_yy + u_zz) - u_t + u_internal * (u_x + u_y + u_z) - f
    loss_pde = (pde_residual ** 2).mean()

    # 边界条件的计算
    u_left = model(torch.cat([torch.ones_like(x_boundary) * (-1), y_boundary, z_boundary, t_boundary], dim=1))
    u_right = model(torch.cat([torch.ones_like(x_boundary) * (1), y_boundary, z_boundary, t_boundary], dim=1))
    u_bottom = model(torch.cat([x_boundary, torch.ones_like(x_boundary) * (-1), z_boundary, t_boundary], dim=1))
    u_top = model(torch.cat([x_boundary, torch.ones_like(x_boundary) * 1, z_boundary, t_boundary], dim=1))
    u_z_left = model(torch.cat([x_boundary, y_boundary, torch.ones_like(z_boundary) * (-1), t_boundary], dim=1))
    u_z_right = model(torch.cat([x_boundary, y_boundary, torch.ones_like(z_boundary) * (1), t_boundary], dim=1))

    loss_boundary = ((u_bottom - u_top - 2) ** 2 + (u_left - u_right) ** 2 + (u_z_left - (-4)) ** 2 + (u_z_right - (2)) ** 2).mean()

    # 初始条件的计算
    u_initial = model(torch.cat([x_initial, y_initial, z_initial, t_initial], dim=1))
    loss_init = ((u_initial.squeeze() - initial_condition(x_initial, y_initial, z_initial).squeeze()) ** 2).mean()

    return loss_pde + loss_boundary + loss_init

# 训练 PINN 模型
def train(model, epochs, optimizer):
    # 内部点
    x_internal = (torch.rand(5000, 1) * 2 - 1).to(device)
    y_internal = (torch.rand(5000, 1) * 2 - 1).to(device)
    z_internal = (torch.rand(5000, 1) * 2 - 1).to(device)
    t_internal = (torch.rand(5000, 1)*0.5).to(device)

    # 边界点
    x_boundary = (torch.rand(5000//4, 1) * 2 - 1).to(device)
    y_boundary = (torch.rand(5000//4, 1) * 2 - 1).to(device)
    z_boundary = (torch.rand(5000//4, 1) * 2 - 1).to(device)
    t_boundary = (torch.rand(5000//4, 1)*0.5).to(device)

    # 初始点
    x_initial = (torch.rand(5000, 1) * 2 - 1).to(device)
    y_initial = (torch.rand(5000, 1) * 2 - 1).to(device)
    z_initial = (torch.rand(5000, 1) * 2 - 1).to(device)
    t_initial = torch.zeros(5000, 1).to(device)

    loss_history = []

    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = compute_loss(model, x_internal, y_internal, z_internal, t_internal,
                            x_boundary, y_boundary, z_boundary, t_boundary,
                            x_initial, y_initial, z_initial, t_initial)
        loss.backward()
        optimizer.step()

        # 记录损失
        loss_history.append(loss.item())

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    end_time = time.time()

    print(f'\nFinal Loss after {epochs} epochs: {loss_history[-1]:.4e}')
    print(f'Total training time: {time.time()-start_time:.1f}s')

    np.savez('3d_PINN_mu2_loss_history.npz', loss_history)
    
    newtime=end_time-start_time

    return loss_history,newtime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')
model = PINN(layers=[4, 10, 10, 10,10,10,10,1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_history,newtime=train(model, 40000, optimizer)

x_step = 0.08
y_step = 0.08
z_step = 0.05
x_range = np.linspace(-1, 1, int((1 - (-1)) / x_step) + 1)
y_range = np.linspace(-1, 1, int((1 - (-1)) / y_step) + 1)
z_range = np.linspace(-1, 1, int((1 - (-1)) / z_step) + 1)
t_value = 0.2



data = []
for y in y_range:
    for x in x_range:
        for z in z_range:
            data.append([x, y, z, t_value])

df = pd.DataFrame(data, columns=['x', 'y','z', 't'])



# 读取真解数据2
file_path = "3d_324newdatamu2_ae_008_008_005.txt"
true_data_2 = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['u', 'x', 'y','z'])
# true_data_t_0_7_2 = true_data_2[true_data_2['t'] == t_value]
# true_data_t_0_7_2 = true_data_t_0_7_2.sort_values(by=['x', 'y'])
# 提取真解的 u, x, y 数据
true_u = true_data_2['u'].values
x_true = true_data_2['x'].values
y_true = true_data_2['y'].values
z_true = true_data_2['z'].values

df_t_0_7 = df[df['t'] == t_value]
x_t_0_7 = torch.tensor(df_t_0_7['x'].values, dtype=torch.float32).unsqueeze(1).to(device)
y_t_0_7 = torch.tensor(df_t_0_7['y'].values, dtype=torch.float32).unsqueeze(1).to(device)
z_t_0_7 = torch.tensor(df_t_0_7['z'].values, dtype=torch.float32).unsqueeze(1).to(device)
t_t_0_7 = torch.tensor(df_t_0_7['t'].values, dtype=torch.float32).unsqueeze(1).to(device)



# start_time = time.time()
u_pred_t_0_7 = model(torch.cat([x_t_0_7, y_t_0_7, z_t_0_7, t_t_0_7], dim=1)).detach().cpu().numpy()
u_pred_t_0_7 = u_pred_t_0_7.reshape(-1)  # 形状从 (40401,1) 变为 (40401,)
# end_time = time.time()
# print(f'Prediction time: {end_time - start_time} seconds')
# 计算相对 L2 误差
def relative_L2_error(true_values, predicted_values):
    return np.linalg.norm(true_values - predicted_values) / np.linalg.norm(true_values)

error = relative_L2_error(true_u, u_pred_t_0_7)
print(f'Relative L2 Error: {error}')

# 绘制误差等值线图
error_values = np.abs(u_pred_t_0_7 - true_u)
# 打印最大的绝对误差
max_absolute_error = np.max(error_values)
print(f"Maximum absolute error: {max_absolute_error}")

#%% 可视化真解、预测解和误差的切片图 -------------------------------------------------
def plot_comparison(data_true, data_pred, error_data):
    # 定义提取切片的函数（保持与 h0 相同的切片位置）
    def scalar_f(x, y, z):
        # 将坐标映射到索引（假设数据按 x,y,z 维度排列）
        xi = ((x + 1)/0.08).astype(int)
        yi = ((y + 1)/0.08).astype(int)
        zi = ((z + 1)/0.05).astype(int) # 这里 z 对应原始数据中的 z 坐标
        return data_true[xi, yi, zi]

    # 定义切片生成函数（与 h0 相同）
    def get_the_slice(x, y, z, surfacecolor):
        return go.Surface(x=x, y=y, z=z, surfacecolor=surfacecolor, coloraxis='coloraxis')

    # t=0.25 切片（对应原始数据中的 z=0.25）
    x = np.linspace(-1, 1, 26)
    y = np.linspace(-1, 1, 26)
    x, y = np.meshgrid(x, y)
    z = 0 * np.ones(x.shape)
    surfcolor_z = scalar_f(x, y, z)
    slice_z = get_the_slice(x, y, z, surfcolor_z)

    # y=0 切片
    x = np.linspace(-1, 1, 26)
    z = np.linspace(-1, 1, 41)  # 这里 t 对应原始数据中的 z 坐标
    x, z = np.meshgrid(x, z)
    y = 0 * np.ones(x.shape)
    surfcolor_y = scalar_f(x, y, z)
    slice_y = get_the_slice(x, y, z, surfcolor_y)

    # x=0 切片
    y = np.linspace(-1, 1, 26)
    z = np.linspace(-1, 1, 41)
    y, z = np.meshgrid(y, z)
    x = 0 * np.ones(x.shape)
    surfcolor_x = scalar_f(x, y, z)
    slice_x = get_the_slice(x, y, z, surfcolor_x)

    # 统一颜色范围
    vmin = min([surfcolor_x.min(), surfcolor_y.min(), surfcolor_z.min()])
    vmax = max([surfcolor_x.max(), surfcolor_y.max(), surfcolor_z.max()])

    # 创建图表
    fig = go.Figure(data=[slice_x, slice_y, slice_z])
    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',  # 原 t 轴改为 z 轴
            zaxis_title='z',
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),  # z 轴范围改为 [-1, 1]
            zaxis=dict(range=[-1, 1]),
            aspectmode='cube'
        ),
        width=700,
        height=700,
        coloraxis=dict(colorscale='jet', colorbar_thickness=25, **dict(cmin=vmin, cmax=vmax))
    )
    return fig

#%% 数据预处理 --------------------------------------------------------------
# 将一维数据转换为三维网格（假设原始数据按 z, y, x 顺序排列）
true_data_3d = true_u.reshape(26, 26, 41)  # 假设 z 是第一个维度
pred_data_3d = u_pred_t_0_7.reshape(26, 26, 41)
error_data_3d = error_values.reshape(26, 26, 41)

#%% 生成并保存图像 ----------------------------------------------------------
# 真解切片图
fig_true = plot_comparison(true_data_3d, None, None)
fig_true.write_html('3d_PINN_mu2_true_solution.html')
pio.write_image(fig_true, '3d_PINN_mu2_true_solution.png')

# 预测解切片图 
fig_pred = plot_comparison(pred_data_3d, None, None)
fig_pred.write_html('3d_PINN_mu2_predicted_solution.html')
pio.write_image(fig_pred, '3d_PINN_mu2_predicted_solution.png')

# 误差切片图（单独设置颜色范围）
fig_error = plot_comparison(error_data_3d, None, None)
fig_error.write_html('3d_PINN_mu2_error_plot.html')
pio.write_image(fig_error, '3d_PINN_mu2_error_plot.png')


print(f'\nFinal Loss after epochs: {loss_history[-1]:.4e}')

print(newtime)


print(f'Relative L2 Error: {error}')

print(f"Maximum absolute error: {max_absolute_error}")















