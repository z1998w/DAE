# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:35:14 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:58:31 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:39:49 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:35:27 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:34:56 2025

@author: zhuqi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import grad
import time
import pandas as pd
# 设置随机种子
seed = 9999
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义物理参数和函数
def mu():
    return 0.01

def f_star(x):
    return x - x**2 + x**3

def u_init(x):
    return 7.5 * torch.tanh((x - 0.1) / 0.01) - 2.5

class SoftConstraintPINN(nn.Module):  # 修改类名以示区别
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            linear = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.layers.append(linear)
        self.activation = nn.Tanh()
        
    def forward(self, x, t):
        input = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            input = self.activation(layer(input))
        return self.layers[-1](input)  # 移除硬约束

def compute_loss(model, x_internal, t_internal, x_boundary, t_boundary, x_initial, t_initial):
    """包含初始条件的损失计算"""
    losses = {}
    
    # PDE残差计算
    x_internal.requires_grad_(True)
    t_internal.requires_grad_(True)
    u = model(x_internal, t_internal)
    u_x = grad(u.sum(), x_internal, create_graph=True)[0]
    u_xx = grad(u_x.sum(), x_internal, create_graph=True)[0]
    u_t = grad(u.sum(), t_internal, create_graph=True)[0]
    f = f_star(x_internal)
    pde_res = mu()*u_xx - u_t + u*u_x - f
    
    
    f_t=grad(pde_res.sum(), t_internal, create_graph=True)[0]
    f_x=grad(pde_res.sum(), x_internal, create_graph=True)[0]
    
    losses['pde'] = torch.mean(pde_res**2)+torch.mean(f_t**2)+torch.mean(f_x**2)

    # 边界条件计算
    u_boundary = model(x_boundary, t_boundary)
    u_left = model(torch.zeros_like(t_boundary), t_boundary)  # 左边界
    u_right = model(torch.ones_like(t_boundary), t_boundary)  # 右边界
    losses['bc'] = ((u_left + 10)**2).mean() + ((u_right - 5)**2).mean()

    # 初始条件计算（新增部分）
    u_initial = model(x_initial, t_initial)
    losses['ic'] = ((u_initial - u_init(x_initial))**2).mean()

    return sum(losses.values()), losses

def train(model, epochs, optimizer):
    """修改后的训练过程"""
    model.train()
    history = {'total': [], 'pde': [], 'bc': [], 'ic': []}
    

    # 预生成边界数据
    n_boundary = 2000
    boundary_data = {
        'x': torch.cat([
            torch.zeros(n_boundary//2, 1, device=device),
            torch.ones(n_boundary//2, 1, device=device)
        ]),
        't': torch.rand(n_boundary, 1, device=device) * 0.3,
        'u_true': torch.cat([
            -10 * torch.ones(n_boundary//2, 1, device=device),
            5 * torch.ones(n_boundary//2, 1, device=device)
        ])
    }
    
    # 生成各类数据点
        # 内部点
    x_internal = torch.rand(2000, 1, device=device)
    t_internal = torch.rand(2000, 1, device=device) * 0.3
        
        # 边界点
    x_boundary = boundary_data['x']
    t_boundary = boundary_data['t']
        
        # 初始条件点（新增部分）
    x_initial = torch.rand(2000, 1, device=device)  # t=0时的x坐标
    t_initial = torch.zeros(2000, 1, device=device)  # 固定t=0
    
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        

        # 计算损失
        total_loss, loss_components = compute_loss(
            model, x_internal, t_internal, 
            x_boundary, t_boundary,
            x_initial, t_initial
        )
        
        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 记录训练过程
        history['total'].append(total_loss.item())
        history['pde'].append(loss_components['pde'].item())
        history['bc'].append(loss_components['bc'].item())
        history['ic'].append(loss_components.get('ic', 0).item())

        if epoch % 5000 == 0:
            print(f"Epoch {epoch:5d} | Total Loss: {total_loss.item():.3e} | "
                  f"PDE: {loss_components['pde'].item():.3e} | "
                  f"BC: {loss_components['bc'].item():.3e} | "
                  f"IC: {loss_components['ic'].item():.3e} | "
                  f"Time: {time.time()-start_time:.1f}s")
            
    print(f"\nFinal Total Loss after {epochs} epochs: {history['total'][-1]:.3e}")
    
    end_time = time.time()
    
    newtime = end_time-start_time

    print(f"\nTotal training time: {time.time()-start_time:.1f} seconds")
    return history,newtime

# 初始化模型和优化器
model = SoftConstraintPINN([2, 10, 10,10,10,1]).to(device)  # 调整网络结构
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
history,newtime = train(model, 20000, optimizer)

# # 保存总体损失数据
# def save_total_loss(history, filename):
#     # 转换为numpy数组
#     loss_total = np.array(history['total'])
    
#     # 保存为压缩文件和CSV
#     np.savez(filename, total=loss_total)
#     pd.DataFrame({'Epoch': np.arange(len(loss_total)), 
#                  'Total_Loss': loss_total}
#                 ).to_csv(filename.replace('.npz', '.csv'), index=False)

# # 绘制总体损失曲线
# def plot_total_loss(history, filename=None):
#     plt.figure(figsize=(10, 6))
    
#     # 设置绘图样式
#     plt.semilogy(history['total'], 
#                 color='#2c3e50',      # 深灰色
#                 linewidth=2.5, 
#                 linestyle='-',
#                 marker='', 
#                 alpha=0.8,
#                 label='Total Loss')
    
#     # 图表美化设置
#     plt.xlabel('Training Epochs', fontsize=12, fontweight='bold', labelpad=10)
#     plt.ylabel('Total Loss', fontsize=12, fontweight='bold', labelpad=10)
#     plt.title('Total Training Loss Progression', 
#              fontsize=14, 
#              pad=15, 
#              fontweight='bold')
    
#     # 网格和边框设置
#     plt.grid(True, which='both', 
#             linestyle='--', 
#             linewidth=0.5,
#             alpha=0.6)
    
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     plt.gca().spines['left'].set_linewidth(1.2)
#     plt.gca().spines['bottom'].set_linewidth(1.2)
    
#     plt.tight_layout()
    
#     if filename:
#         plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.show()

# # ================= 使用示例 =================
# # 保存数据 (训练结束后调用)
# save_total_loss(history, '1d_PINN_mu2_total_loss_history.npz')

# # 绘制图像
# plot_total_loss(history, '1d_PINN_mu2_total_loss_curve.png')


# 设置均匀剖分的步长
x_step = 0.005
t_step = 0.005

# 生成数据对
x, t = np.mgrid[0:1+x_step:x_step, 0:0.3+t_step:t_step]
data = np.column_stack((x.ravel(), t.ravel()))

# 转换为 torch.Tensor
x_tensor = torch.tensor(data[:, 0], dtype=torch.float32).reshape(-1, 1).to(device)
t_tensor = torch.tensor(data[:, 1], dtype=torch.float32).reshape(-1, 1).to(device)

# 模型预测
# 模型预测（正确方式）
model.eval()
with torch.no_grad():
    u_pred = model(x_tensor, t_tensor).cpu().numpy()  # 直接传入两个参数

X_grid = x
T_grid = t
U0_grid = u_pred.reshape(X_grid.shape)



# 加载真实解
true_data = np.loadtxt('1d_0302newdatamu2_ae_0005_0005.txt')

# 分离真实解中的 u, x, t
u_true = true_data[:, 0]
x_true = true_data[:, 1]
t_true = true_data[:, 2]

# 创建 x 和 t 的网格
x_unique = np.unique(x_true)
t_unique = np.unique(t_true)
X_true, T_true = np.meshgrid(x_unique, t_unique)

# 创建 u 的网格
U_true = np.zeros_like(X_true)
for i in range(len(x_true)):
    xi = np.where(x_unique == x_true[i])[0][0]
    ti = np.where(t_unique == t_true[i])[0][0]
    U_true[ti, xi] = u_true[i]


# 计算预测解和真解的绝对误差
absolute_error = np.abs(U0_grid - U_true.T)
max_absolute_error = np.max(absolute_error)
print(f"Maximum absolute error: {max_absolute_error}")


    # 计算相对 L2 误差
relative_L2_error = np.linalg.norm(U0_grid - U_true.T) / np.linalg.norm(U_true.T)
print(f'Relative L2 error: {relative_L2_error}')
    

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


def plot_3d_surface(X, Y, Z, title, cmap='plasma', elev=20, azim=105):
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
    
    # ========== 坐标轴标签优化 ==========
    ax.set_xlabel('x', labelpad=15, fontsize=28, fontweight='bold')
    ax.set_ylabel('t', labelpad=25, fontsize=28, fontweight='bold')
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
                        pad=0.01,      # 增加与主图的间距(原0.01)
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
# 绘制预测解的3D图
fig_pred_3d = plot_3d_surface(X_grid, T_grid, U0_grid, 
                              "Predicted Solution of PINN")
plt.savefig('1d_PINN_mu2_3d_surface.eps', dpi=300, bbox_inches='tight')

# 绘制真解的3D图
fig_true_3d = plot_3d_surface(X_true, T_true, U_true, 
                              "Reference Solution")
plt.savefig('1d_true_mu2_3d_surface.eps', dpi=300, bbox_inches='tight')



# ================= 误差分布图 =================
fig_error = plt.figure(figsize=(12, 9))
ax = fig_error.add_subplot(111)

# 误差等值线
error_contour = ax.contourf(X_grid, T_grid, absolute_error, 
                            levels=50, 
                            cmap='plasma')

# 坐标轴设置
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('t', labelpad=20)

# 颜色条设置
cbar = fig_error.colorbar(error_contour, ax=ax)
# cbar.set_label('Absolute Error', labelpad=20)
cbar.ax.tick_params(labelsize=24, width=2, length=6)

plt.title('Absolute Error of PINN', pad=20)
plt.tight_layout()
plt.savefig('1d_PINN_mu2_error_contour.eps', dpi=300)

# ================= 时间切片对比图 =================

# ================= 时间切片对比图 =================
plt.rcParams.update({'font.size': 20})  # 全局字体放大

fig_comparison, axes = plt.subplots(1, 3, figsize=(18, 8))

# 统一设置子图参数
plot_params = {
    'xlabel': 'x',
    'ylabel': 'u',
    'xaxis.labelshift': 15,  # x轴标签偏移量
    'yaxis.labelshift': 15   # y轴标签偏移量
}

for ax, t_point in zip(axes, [0, 0.15, 0.3]):
    # 获取数据索引
    idx = np.argmin(np.abs(T_grid[0] - t_point))
    
    # 绘制曲线（保持不变）
    ax.plot(X_grid[:,0], U_true.T[:,idx], 'b-', label='Reference', linewidth=4)
    ax.plot(X_grid[:,0], U0_grid[:,idx], 'r--', label='PINN', linewidth=4)
    
    # ========== 坐标轴放大设置 ==========
    # 标签字体设置
    ax.set_xlabel('x', fontsize=24, labelpad=15)  # 标签字体24pt，间距15pt
    ax.set_ylabel('u', fontsize=24, labelpad=15)
    
    # 刻度设置
    ax.tick_params(axis='both', 
                  which='major', 
                  labelsize=20,       # 刻度数字大小
                  width=3,           # 刻度线宽度
                  length=8,          # 刻度线长度
                  direction='inout') # 刻度线样式
    
    # 坐标轴线宽设置
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # 坐标轴线宽2pt
    
    # 其他设置保持不变
    ax.set_title(f"t = {t_point}", fontsize=24, pad=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(-12, 7)
    
    if ax == axes[0]:
        ax.legend(loc='best', 
                  fontsize=20, 
                  frameon=True, 
                  framealpha=0.8)

plt.tight_layout(pad=3.0)  # 增加子图间距
# plt.savefig('1d_comparison.eps', dpi=300, bbox_inches='tight')
plt.show()

# ================= 保存数据 =================
# 在模型预测完成后添加以下代码
np.savez('1d_PINN_mu2_solution_data.npz',
          X_grid=X_grid,
          T_grid=T_grid,
          U0_grid=U0_grid,
          X_true=X_true,
          T_true=T_true,
          U_true=U_true)

# ================= 加载数据并绘图 =================
def load_and_plot_comparison():
    try:
        # 加载保存的数据
        data = np.load('1d_PINN_mu2_solution_data.npz')
        X_grid = data['X_grid']
        T_grid = data['T_grid']
        U0_grid = data['U0_grid']
        X_true = data['X_true']
        T_true = data['T_true']
        U_true = data['U_true']
    except FileNotFoundError:
        print("Error: Data file 'solution_data.npz' not found.")
        return

    plt.rcParams.update({'font.size': 20})
    fig_comparison, axes = plt.subplots(1, 3, figsize=(18, 8))

    # 统一设置子图参数
    plot_params = {
        'xlabel': 'x',
        'ylabel': 'u',
        'xaxis.labelshift': 15,
        'yaxis.labelshift': 15
    }

    for ax, t_point in zip(axes, [0, 0.15, 0.3]):
        # 获取数据索引
        idx = np.argmin(np.abs(T_grid[0] - t_point))
        
        # 绘制曲线
        ax.plot(X_grid[:,0], U_true.T[:,idx], 'b-', label='Reference', linewidth=4)
        ax.plot(X_grid[:,0], U0_grid[:,idx], 'r--', label='PINN', linewidth=4)
        
        # 坐标轴设置
        ax.set_xlabel('x', fontsize=24, labelpad=15)
        ax.set_ylabel('u', fontsize=24, labelpad=15)
        ax.tick_params(axis='both', 
                      which='major', 
                      labelsize=20,
                      width=3,
                      length=8,
                      direction='inout')
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        ax.set_title(f"t = {t_point}", fontsize=24, pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(-12, 7)
        
        if ax == axes[0]:
            ax.legend(loc='best', 
                      fontsize=20, 
                      frameon=True, 
                      framealpha=0.8)

    plt.tight_layout(pad=3.0)
    plt.savefig('1d_PINN_mu2_comparison_from_file.eps', dpi=300, bbox_inches='tight')
    plt.show()

# 调用绘图函数
load_and_plot_comparison()

print(f"Maximum absolute error: {max_absolute_error}")

print(f'Relative L2 error: {relative_L2_error}')

print(f"\nFinal Total Loss after epochs: {history['total'][-1]:.3e}")
    
print(newtime)