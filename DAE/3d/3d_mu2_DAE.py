# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:27:06 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:04:27 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:58:26 2025

@author: zhuqi
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.autograd import grad
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
import time
import matplotlib.pyplot as plt
from pyDOE import lhs
import torch.nn.init as init

# 确保使用 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            self.layers.append(layer)

        # Xavier (Glorot) initialization for Tanh activation
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


# 定义 varphi 函数，使用蒙特卡洛积分
def monte_carlo_integration(func, a, b, num_samples=3000):#积分点数在这里给了和后面没关系
    samples = a + (b - a) * torch.rand(num_samples, 1).to(device)
    func_values = func(samples)
    integral = (b - a) * torch.mean(func_values)
    return integral


def varphi_minus(x, y, z):
    integrand = lambda s: torch.cos(np.pi * (x - 1 - z + s)) * torch.cos(np.pi * (y - 1 - z + s)) * torch.cos(np.pi * (-1 + s))
    integral = monte_carlo_integration(integrand, 0, 1 + z)
    return -torch.sqrt(16 + 2 * integral)


def varphi_plus(x, y, z):
    integrand = lambda s: torch.cos(np.pi * (x + 1 - z + s)) * torch.cos(np.pi * (y + 1 - z + s)) * torch.cos(np.pi * (1 + s))
    integral = monte_carlo_integration(integrand, z - 1, 0)
    return torch.sqrt(4 - 2 * integral)


# 定义损失函数
def compute_loss(model, x_internal, y_internal, t_internal, x_boundary, y_boundary, t_boundary):
    x_internal = x_internal.clone().detach().requires_grad_(True)
    y_internal = y_internal.clone().detach().requires_grad_(True)
    t_internal = t_internal.clone().detach().requires_grad_(True)

    u_internal = t_internal * model(torch.cat([x_internal, y_internal, t_internal], dim=1))
    u_x = grad(u_internal.sum(), x_internal, create_graph=True)[0]
    u_y = grad(u_internal.sum(), y_internal, create_graph=True)[0]
    u_t = grad(u_internal.sum(), t_internal, create_graph=True)[0]

    varphi_minus_vals = varphi_minus(x_internal, y_internal, u_internal)
    varphi_plus_vals = varphi_plus(x_internal, y_internal, u_internal)

    residual = u_t - 0.5 * (u_x + u_y - 1) * (varphi_minus_vals + varphi_plus_vals)

    loss_pde = (residual ** 2).mean()

    # # 这里周期边界精确一下，如果初始条件精确成立，边界条件在0处都为0，不需要这个边界条件了
    u_left = t_boundary * model(torch.cat([torch.ones_like(x_boundary) * (-1), y_boundary, t_boundary], dim=1))
    u_right = t_boundary * model(torch.cat([torch.ones_like(x_boundary) * (1), y_boundary, t_boundary], dim=1))

    u_bottom = t_boundary * model(torch.cat([x_boundary, torch.ones_like(x_boundary) * (-1), t_boundary], dim=1))
    u_top = t_boundary * model(torch.cat([x_boundary, torch.ones_like(x_boundary) * 1, t_boundary], dim=1))

    loss_boundary = ((u_bottom - u_top) ** 2 + (u_left - u_right) ** 2).mean()

    return loss_pde + loss_boundary


# 训练模型
def train_model(model, epochs, optimizer):
    # 内部点
    x_internal = (torch.rand(3000, 1) * 2 - 1).to(device)
    y_internal = (torch.rand(3000, 1) * 2 - 1).to(device)
    t_internal = (torch.rand(3000, 1) * 0.5).to(device)

    # 边界点
    x_boundary = (torch.rand(2000//2, 1) * 2 - 1).to(device)
    y_boundary = (torch.rand(2000//2, 1) * 2 - 1).to(device)
    t_boundary = (torch.rand(2000//2, 1) * 0.5).to(device)

    loss_history = []

    start_time = time.time()  # 开始计时（采样已完成）

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 计算损失
        loss = compute_loss(model, x_internal, y_internal, t_internal,
                            x_boundary, y_boundary, t_boundary)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())  # 保存损失值

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    end_time = time.time()  # 结束计时
    training_time = end_time - start_time  # 计算训练时间
    print(f"Training completed in: {training_time:.2f} seconds")
    print(f'\nFinal Loss after {epochs} epochs: {loss_history[-1]:.4e}')
    print(f'Total training time: {time.time() - start_time:.1f}s')

    np.savez('3d_mu2_loss_history.npz', loss_history)

    return loss_history, training_time


# 实例化并训练模型
model = PINN([3, 10, 10, 10, 10,10, 10,1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_history, training_time = train_model(model, 20000, optimizer)

# 可视化训练损失
plt.figure(figsize=(10, 6))
plt.semilogy(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('Training Loss History with Fixed Sampling')
plt.grid(True)
plt.show()

# x_step = 0.08
# y_step = 0.08
# t_step = 0.05
# x_range = np.linspace(-1, 1, int((1 - (-1)) / x_step) + 1)
# y_range = np.linspace(-1, 1, int((1 - (-1)) / y_step) + 1)
# t_range = np.linspace(0, 0.5, int((0.5 - (0)) / z_step) + 0.5)
# t_value = 0.2



# data = []
# for y in y_range:
#     for x in x_range:
#         for t in t_range:
#             data.append([x, y, t])


# df = pd.DataFrame(data, columns=['x', 'y','t'])

# points_t = torch.tensor(df, dtype=torch.float32).to(device)

# # Predict values
# model.eval()
# with torch.no_grad():
    
#     h0_pred = (points_t[:, 2:3]*model(points_t)).cpu().numpy()

# Define grid for plotting
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
t = np.linspace(0, 0.5, 101)

X, Y, T = np.meshgrid(x, y, t)
points = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)
points_t = torch.tensor(points, dtype=torch.float32).to(device)

# Predict values
model.eval()
with torch.no_grad():
    
    h0_pred = (points_t[:, 2:3]*model(points_t)).reshape(101, 101, 101).cpu().numpy()




# Transpose the data to match the example
#data2 = np.transpose(h0_pred, (2, 1, 0))
data2 = h0_pred

def get_the_slice(x, y, t, surfacecolor):
    return go.Surface(x=x, y=y, z=t, surfacecolor=surfacecolor, coloraxis='coloraxis')

def get_lims_colors(surfacecolor):  # color limits for a slice
    return np.min(surfacecolor), np.max(surfacecolor)

scalar_f = lambda x, y, t: data2[((x+1)/0.02).astype(int), ((y+1)/0.02).astype(int), (t/0.005).astype(int)]

# t=0. slice
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
x, y = np.meshgrid(x, y)
t = 0.25 * np.ones(x.shape)
surfcolor_t = scalar_f(x, y, t)
smint, smaxt = get_lims_colors(surfcolor_t)
slice_t = get_the_slice(x, y, t, surfcolor_t)

# y=0 slice
x = np.linspace(-1, 1, 101)
t = np.linspace(0, 0.5, 101)
x, t = np.meshgrid(x, t)
y = 0 * np.ones(x.shape)
surfcolor_y = scalar_f(x, y, t)
sminy, smaxy = get_lims_colors(surfcolor_y)
slice_y = get_the_slice(x, y, t, surfcolor_y)

# x=0 slice
y = np.linspace(-1, 1, 101)
t = np.linspace(0, 0.5, 101)
y, t = np.meshgrid(y, t)
x = 0 * np.ones(y.shape)
surfcolor_x = scalar_f(x, y, t)
sminx, smaxx = get_lims_colors(surfcolor_x)
slice_x = get_the_slice(x, y, t, surfcolor_x)

vmin = min([sminx, sminy, smint])
vmax = max([smaxx, smaxy, smaxt])

def colorax(vmin, vmax):
    return dict(cmin=vmin, cmax=vmax)

# 绘制 h0 切片图
fig1 = go.Figure(data=[slice_x, slice_y, slice_t])
fig1.update_layout(
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='t',
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        zaxis=dict(range=[0, 0.5]),
        aspectmode='cube'  # 添加 aspectmode 设置
    ),
    title_x=0.5,
    width=700,
    height=700,
    coloraxis=dict(colorscale='jet', colorbar_thickness=25, colorbar_len=0.75, **colorax(vmin, vmax))
)
#cividis
# Save the figure
fig1.write_html('3d_h0_plot.html')
pio.write_image(fig1, '3d_h0.png')
# 生成新的网格数据
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


# 计算近似解 U02
def compute_U0_2(x, y, z,h0, h0_x,h0_y, phi_m, phi_p, phi_m_xy, phi_p_xy, mu=0.05):
    if z <= h0:
        U0 = phi_m_xy + (phi_p - phi_m) / (torch.exp((h0 - z) * (phi_p - phi_m) * (1 - h0_x - h0_y) / (2 * mu)) + 1)
    else:
        U0 = phi_p_xy + (phi_m - phi_p) / (torch.exp((h0 - z) * (phi_m - phi_p) * (1 - h0_x - h0_y) / (2 * mu)) + 1)
    return U0


# 提取预测解的 x, y, t 数据
x_t_0_7 = df['x'].values
y_t_0_7 = df['y'].values
z_t_0_7 = df['z'].values
t_t_0_7 = df['t'].values


# 初始化预测解数组
U0_t_0_7_2 = np.zeros_like(x_t_0_7)


# 设置模型为评估模式
model.eval()

# 计算预测解
start_time = time.time()
for i in range(len(x_t_0_7)):
    x_val = torch.tensor([[x_t_0_7[i]]], dtype=torch.float32, requires_grad=True, device=device)
    y_val = torch.tensor([[y_t_0_7[i]]], dtype=torch.float32, requires_grad=True, device=device)
    z_val = torch.tensor([[z_t_0_7[i]]], dtype=torch.float32, device=device)
    t_val = torch.tensor([[t_t_0_7[i]]], dtype=torch.float32, requires_grad=True, device=device)
    input_val = torch.cat([x_val,y_val, t_val], dim=1)

    h0_val = t_val*model(input_val)

    # 计算 h0 对 x 的梯度
    h0_x = torch.autograd.grad(h0_val, x_val, grad_outputs=torch.ones_like(h0_val), create_graph=True)[0]
    h0_y = torch.autograd.grad(h0_val, y_val, grad_outputs=torch.ones_like(h0_val), create_graph=True)[0]

    phi_m_x_y = varphi_minus(x_val, y_val,z_val)
    phi_p_x_y = varphi_plus(x_val, y_val,z_val)

    phi_m_val = varphi_minus(x_val, y_val,h0_val)
    phi_p_val = varphi_plus(x_val, y_val,h0_val)

   
    U0_val_2 = compute_U0_2(x_val, y_val,z_val, h0_val, h0_x,h0_y, phi_m_val, phi_p_val, phi_m_x_y, phi_p_x_y)
    U0_t_0_7_2[i] = U0_val_2.item()
    
    
# 计算相对 L2 误差
def relative_L2_error(true_values, predicted_values):
    return np.linalg.norm(true_values - predicted_values) / np.linalg.norm(true_values)



error_2 = relative_L2_error(true_u, U0_t_0_7_2)
print(f'Relative L2 Error: {error_2}')


# 绘制误差等值线图
error_values_2 = np.abs(U0_t_0_7_2 - true_u)
# 打印最大的绝对误差
max_absolute_error_2 = np.max(error_values_2)
print(f"Maximum absolute error_2: {max_absolute_error_2}")    


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
pred_data_3d = U0_t_0_7_2.reshape(26, 26, 41)
error_data_3d = error_values_2.reshape(26, 26, 41)

#%% 生成并保存图像 ----------------------------------------------------------
# 真解切片图
fig_true = plot_comparison(true_data_3d, None, None)
fig_true.write_html('3d_mu2_true_solution.html')
pio.write_image(fig_true, '3d_mu2_true_solution.png')

# 预测解切片图 
fig_pred = plot_comparison(pred_data_3d, None, None)
fig_pred.write_html('3d_mu2_predicted_solution.html')
pio.write_image(fig_pred, '3d_mu2_predicted_solution.png')

# 误差切片图（单独设置颜色范围）
fig_error = plot_comparison(error_data_3d, None, None)
fig_error.write_html('3d_mu2_error_plot.html')
pio.write_image(fig_error, '3d_mu2_error_plot.png')

print(f'\nFinal Loss after epochs: {loss_history[-1]:.4e}')
print(training_time)
print(f'Relative L2 Error: {error_2}')
print(f"Maximum absolute error_2: {max_absolute_error_2}")    