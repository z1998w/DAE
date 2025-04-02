# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:40:27 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:24:41 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:37:19 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:17:27 2024

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:17:43 2024

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:18:31 2024

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:35:19 2024

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 14:44:04 2024

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

import torch.nn.init as init
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        
        # 定义网络层
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            self.layers.append(layer)
        
        # Xavier (Glorot) 初始化，适用于 Tanh 激活函数
        self.apply(self._initialize_weights)
        
        self.activation = nn.Tanh()

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            # 使用 Xavier 正态分布初始化
            init.xavier_normal_(layer.weight)
            # 初始化偏置为零
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, x):
        # 对于除最后一层外的所有层，应用 Tanh 激活函数
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        # 最后一层不应用激活函数
        return self.layers[-1](x)


CONST_600 = torch.tensor(600., requires_grad=False)
CONST_6 = torch.tensor(6., requires_grad=False)
CONST_145 = torch.tensor(145., requires_grad=False)

def phi_l(x):
    return -torch.sqrt(CONST_600 + 6.*x**2 - 4.*x**3 + 3.*x**4) / torch.sqrt(CONST_6)

def phi_r(x):
    return torch.sqrt(CONST_145 + 6.*x**2 - 4.*x**3 + 3.*x**4) / torch.sqrt(CONST_6)


if __name__ == '__main__':
    
    size_internal = 1000
    size_init = 1000
    lr = 0.001
    
    t_train_internal = np.random.rand(size_internal, 1) * 0.3  # 生成[0,0.3)的均匀随机采样
    t_train_internal = torch.tensor(t_train_internal, dtype=torch.float32).to(device)

    # 生成初始条件点
    t_train_init = torch.zeros(size_init, 1).to(device)  # 初始条件点的 t 固定为 0
    
    net = Net([1, 12, 12,12,12,12,1]).to(device)
    loss_fn = nn.MSELoss(reduction='mean').to(device)
    
    # 定义 Adam 优化器
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_history = []
    # 记录训练开始时间
    train_start_time = time.time()
    # 训练循环
    num_epochs = 10000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 处理内部点的损失
        t_internal = t_train_internal.requires_grad_(True)
        x_internal = 0.1+t_internal*net(t_internal)
        dxdt_internal = torch.autograd.grad(x_internal, t_internal, grad_outputs=torch.ones_like(x_internal), create_graph=True)[0]
    
        # 计算方程的损失
        phi_l_val = phi_l(x_internal)
        phi_r_val = phi_r(x_internal)
        avg_phi = -0.5 * (phi_l_val + phi_r_val)
        eq_loss = loss_fn(dxdt_internal, avg_phi)
    
        # 计算初始条件的损失
        t_init = t_train_init.requires_grad_(False)
        x0_pred = net(t_init)
        init_loss = loss_fn(x0_pred, torch.tensor([[0.1]] * size_init, dtype=torch.float32).to(device))
    
        # 总损失
        loss = eq_loss
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

        if epoch % 5000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            
    train_end_time = time.time()    
    # 计算训练时间
    training_time = train_end_time - train_start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    print(f'Final Loss after {num_epochs} epochs: {loss.item()}')
    
    # 保存损失历史
    np.save('1d_DAE_mu2_loss_history.npy', loss_history)

    # 绘制损失曲线
    plt.figure()
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True)
    plt.legend()
    # plt.savefig('results/loss_curve.png')
    plt.show()

    # 生成测试数据
    t_test = torch.linspace(0, 0.3, steps=1000).view(-1, 1).to(device)
    net.eval()
    with torch.no_grad():
        x_test = 0.1+t_test*net(t_test)
    x_test = x_test.cpu().numpy()
    t_test = t_test.cpu().numpy()
    
    # 设置全局绘图参数
    plt.rcParams.update({
        'font.family': 'serif',        # 使用衬线字体
        'font.size': 14,               # 基础字号
        'axes.titlesize': 18,          # 标题字号
        'axes.labelsize': 18,          # 坐标轴标签字号
        'xtick.labelsize': 12,         # X轴刻度字号
        'ytick.labelsize': 12,         # Y轴刻度字号
        'axes.linewidth': 1.5,         # 坐标轴线宽
        'lines.linewidth': 2.5,        # 曲线线宽
        'legend.fontsize': 18,         # 图例字号
        'savefig.dpi': 300,            # 保存分辨率
        'savefig.bbox': 'tight',       # 自动裁剪白边
        'mathtext.fontset': 'stix'     # 数学公式字体
    })
    
    # 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6))  # 8x6英寸画布
    
    # 绘制预测曲线
    ax.plot(t_test, x_test, 
            color='#2c7bb6',          # 使用更专业的蓝色
            linestyle='-',
            marker=''
            )
    
    ax.set_title('Predicted $x_0(t)$', 
            pad=20,                # 标题与图的间距
            fontweight='bold',     # 加粗
            color='#2d3436')       # 使用深灰色

    
    # 坐标轴设置
    ax.set_xlabel('$t$', labelpad=10)  # 增加标签间距
    ax.set_ylabel('$x_0(t)$', labelpad=10)
    ax.set_xlim(0, 0.3)  # 明确设置坐标范围
    ax.set_ylim(np.min(x_test)-0.1, np.max(x_test)+0.1)
    
    
    # 保存和显示
    plt.tight_layout(pad=2.0)  # 增加布局边距
    plt.savefig('1d_mu2_x0_prediction.eps', format='eps')  # 矢量图格式
    plt.show()
    
    mu = 0.01

    # 设置均匀剖分的步长
    x_step = 0.005
    t_step = 0.005

    
    # 如果你需要 meshgrid
    
    x, t = np.mgrid[0:1+x_step:x_step, 0:0.3+t_step:t_step]
    data = np.column_stack((x.ravel(), t.ravel()))

    # 转换为 torch.Tensor 并转移到 GPU
    x_tensor = torch.tensor(data[:, 0], dtype=torch.float32).reshape(-1, 1).to(device)
    t_tensor = torch.tensor(data[:, 1], dtype=torch.float32).reshape(-1, 1).to(device)

    # 使用已训练的 net 获得 x0(t)
    with torch.no_grad():
        x0 = 0.1+t_tensor*net(t_tensor)

    phi_l_val = phi_l(x_tensor)
    phi_r_val = phi_r(x_tensor)
    phi_l_x0 = phi_l(x0)
    phi_r_x0 = phi_r(x0)

    U0 = torch.where(x_tensor <= x0, 
                     phi_l_val + (phi_r_x0 - phi_l_x0) / (torch.exp((x_tensor - x0) * ((phi_l_x0 - phi_r_x0) / (2 * mu))) + 1), 
                     phi_r_val - (phi_r_x0 - phi_l_x0) / (torch.exp((x_tensor - x0) * ((phi_r_x0 - phi_l_x0) / (2 * mu))) + 1))

    X_np, T_np, U0_np = x_tensor.cpu().numpy(), t_tensor.cpu().numpy(), U0.cpu().numpy()

    # Reshape the data for grid plotting
    X_grid = x
    T_grid = t
    U0_grid = U0_np.reshape(X_grid.shape)
    
    
    
    
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
    
    # 打印最大的绝对误差
    max_absolute_error = np.max(absolute_error)
    print(f"Maximum absolute error: {max_absolute_error}")


    # 计算相对 L2 误差
    relative_L2_error = np.linalg.norm(U0_grid - U_true.T) / np.linalg.norm(U_true.T)
    print(f'Relative L2 error: {relative_L2_error}')
    
    absolute_error_1=absolute_error.T
    
    
    
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
    # 绘制预测解的3D图
    fig_pred_3d = plot_3d_surface(X_grid, T_grid, U0_grid, 
                                  "Predicted Solution of DAE")
    plt.savefig('1d_DAE_mu2_3d_surface.eps', dpi=300, bbox_inches='tight')

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

    plt.title('Absolute Error of DAE', pad=20)
    plt.tight_layout()
    plt.savefig('1d_DAE_mu2_error_contour.eps', dpi=300)

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
    np.savez('1d_DAE_mu2_solution_data.npz',
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
            data = np.load('1d_DAE_mu2_solution_data.npz')
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
        plt.savefig('1d_DAE_mu2_comparison_from_file.eps', dpi=300, bbox_inches='tight')
        plt.show()

    # 调用绘图函数
    load_and_plot_comparison()
    
    print(f"Training time: {training_time:.2f} seconds")
    
    print(f'Final Loss after epochs: {loss.item()}')
    
    print(f"Maximum absolute error: {max_absolute_error}")


    print(f'Relative L2 error: {relative_L2_error}')
    
    
    

