# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:29:59 2025

@author: zhuqi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:39:06 2025

@author: zhuqi
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim
import time
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
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
        for i in range(len(layers)-1):
            layer = nn.Linear(layers[i], layers[i+1])
            self.layers.append(layer)
        self.apply(self._init_weights)
        self.activation = nn.Tanh()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
            
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

# 定义源项函数
def f_source(x, y):
    return torch.cos(np.pi*x/4) * torch.cos(np.pi*y/4)

# 初始化条件
def u_init(x, y, mu=0.01):
    return 3*torch.tanh((x + y/mu)) - 1

def build_dataset(x_f_batch, x_add):
    return torch.cat((x_f_batch, x_add), dim=0)

def add_data(x_f_test, residual_test, add_k):
    abs_residual = np.abs(residual_test)
    topk_indices = np.argsort(-abs_residual)[:add_k]
    x_add = x_f_test[topk_indices]
    
    # 强制转换为二维数组 [新增]
    if x_add.ndim == 1:
        x_add = x_add.reshape(-1, 3)
    elif x_add.ndim == 3:
        x_add = x_add.reshape(-1, 3)
        
    return torch.tensor(x_add, dtype=torch.float32, requires_grad=True).to(device)

def main():
    # 超参数设置
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    T_max = 1.0
    
    mu=0.01
    initial_size_internal = 2800
    initial_size_boundary = 1000#这样边界是3000个
    initial_size_initial = 3000
    max_iter = 40         # 外循环最大迭代次数
    residual_threshold = 1e-6
    inner_max_steps = 30000 # 内循环最大步数
    inner_loss_threshold = 1e-13
    add_k = 5
    lr = 0.001

    # 生成初始训练数据 ----------------------------------------------------------
    # 内部点 (三维空间)
    x_f = torch.FloatTensor(initial_size_internal, 1).uniform_(-2, 2).to(device)
    y_f = torch.FloatTensor(initial_size_internal, 1).uniform_(-2, 2).to(device)
    t_f = torch.FloatTensor(initial_size_internal, 1).uniform_(0, T_max).to(device)
    X_f_train = torch.cat((x_f,y_f, t_f), dim=1).requires_grad_(True)
    
    # x = np.linspace(x_min, x_max, initial_points).reshape(-1, 1)
    # y = np.linspace(y_min, y_max, initial_points).reshape(-1, 1)
    # t = np.linspace(0, t_max, initial_points).reshape(-1, 1)
    # X_train = np.hstack((x, y, t))
    # X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)
    
    # 边界条件 (y方向固定值)
    # y = -2边界
    
    t_b = torch.FloatTensor(initial_size_boundary, 1).uniform_(0, T_max).to(device)
    y_left = -2.0 * torch.ones(initial_size_boundary, 1).to(device)  # 左边界x=-2
    y_right = 2.0 * torch.ones(initial_size_boundary, 1).to(device)   # 右边界x=2
    x_b = torch.FloatTensor(initial_size_boundary, 1).uniform_(-2, 2).to(device)

# 构建边界训练点
    X_b_train_left = torch.cat((x_b,y_left, t_b), dim=1).requires_grad_(True)
    X_b_train_right = torch.cat((x_b,y_right, t_b), dim=1).requires_grad_(True)
    
    x_left = -2.0 * torch.ones(initial_size_boundary, 1).to(device)  # 左边界x=-2
    x_right = 2.0 * torch.ones(initial_size_boundary, 1).to(device)   # 右边界x=2
    y_b = torch.FloatTensor(initial_size_boundary, 1).uniform_(-2, 2).to(device)

# 构建边界训练点
    X_b_train_left_2 = torch.cat((x_left,y_b, t_b), dim=1).requires_grad_(True)
    X_b_train_right_2 = torch.cat((x_right,y_b, t_b), dim=1).requires_grad_(True)
    
    x0 = np.random.uniform(-2, 2, (initial_size_initial, 1))
    y0 = np.random.uniform(-2, 2, (initial_size_initial, 1))
    t0 = np.zeros_like(x0)
    X0_train = torch.tensor(np.hstack((x0,y0, t0)), dtype=torch.float32, requires_grad=True).to(device)
    U0_train = u_init(torch.tensor(x0), torch.tensor(y0)).detach().numpy()
    U0_train = torch.tensor(U0_train, dtype=torch.float32).to(device)

    # y_b1 = np.full((boundary_points//2, 1), y_min)
    # x_b1 = np.linspace(x_min, x_max, boundary_points//2).reshape(-1, 1)
    # t_b1 = np.linspace(0, t_max, boundary_points//2).reshape(-1, 1)
    # u_b1 = np.full((boundary_points//2, 1), -4.0)
    
    # # y = 2边界
    # y_b2 = np.full((boundary_points//2, 1), y_max)
    # x_b2 = np.linspace(x_min, x_max, boundary_points//2).reshape(-1, 1)
    # t_b2 = np.linspace(0, t_max, boundary_points//2).reshape(-1, 1)
    # u_b2 = np.full((boundary_points//2, 1), 2.0)
    
    # X_fixed = np.vstack((np.hstack((x_b1, y_b1, t_b1)),
    #                     np.hstack((x_b2, y_b2, t_b2))))
    # U_fixed = np.vstack((u_b1, u_b2))
    
    # periodic_num = boundary_points // 2  # 每个方向分配一半边界点
    
    # # 生成左边界点 (x=-2)
    # x_left = np.full((periodic_num, 1), x_min)
    # y_periodic = np.linspace(y_min, y_max, periodic_num).reshape(-1, 1)
    # t_periodic = np.linspace(0, t_max, periodic_num).reshape(-1, 1)
    # X_periodic_left = np.hstack((x_left, y_periodic, t_periodic))
    
    # # 生成对应的右边界点 (x=2)
    # x_right = np.full((periodic_num, 1), x_max)
    # X_periodic_right = np.hstack((x_right, y_periodic, t_periodic))
    
    # # 转换为Tensor
    # X_bc = torch.tensor(X_fixed, dtype=torch.float32, requires_grad=True).to(device)
    # U_bc = torch.tensor(U_fixed, dtype=torch.float32).to(device)
    
    # X_bc_left = torch.tensor(X_periodic_left, dtype=torch.float32, requires_grad=False).to(device)
    # X_bc_right = torch.tensor(X_periodic_right, dtype=torch.float32, requires_grad=False).to(device)
    
    # # 初始条件 (t=0)
    # x0 = np.linspace(x_min, x_max, initial_time_points).reshape(-1, 1)
    # y0 = np.linspace(y_min, y_max, initial_time_points).reshape(-1, 1)
    # t0 = np.zeros((initial_time_points, 1))
    # X_init = np.hstack((x0, y0, t0))
    # U_init = u_init(torch.tensor(x0), torch.tensor(y0)).detach().numpy()
    # X_init = torch.tensor(X_init, dtype=torch.float32, requires_grad=True).to(device)
    # U_init = torch.tensor(U_init, dtype=torch.float32).to(device)

    # 初始化网络
    net = Net([3, 10, 10, 10,10,10, 1]).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_history = []
    train_start_time = time.time()

    # 训练循环 ----------------------------------------------------------------
    n = 0  # 外循环计数器
    err = residual_threshold + 1
    
    # 外循环：控制自适应采样迭代
    while n < max_iter and err > residual_threshold:
        print(f"\n=== 自适应迭代 {n+1}/{max_iter} ===")
        
        # 构建当前训练集
        if n == 0:
            X_f_batch = X_f_train.clone().detach().requires_grad_(True).to(device)
        else:
            try:
                X_f_batch = build_dataset(X_f_batch, X_add).clone().detach().requires_grad_(True).to(device)
            except Exception as e:
                print(f"构建数据集错误: {e}")
                break

        # 内循环：模型训练
        n_r = 0
        loss_r = float('inf')
        while n_r < inner_max_steps and loss_r > inner_loss_threshold:
            optimizer.zero_grad()
            
            # 前向传播
            u_pred = net(X_f_batch)
            
            # 计算梯度
            u_x = torch.autograd.grad(u_pred, X_f_batch, 
                                     grad_outputs=torch.ones_like(u_pred),
                                     create_graph=True)[0][:, 0:1]
            
            u_y = torch.autograd.grad(u_pred, X_f_batch, 
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True)[0][:, 1:2]
            
            u_t = torch.autograd.grad(u_pred, X_f_batch,
                                     grad_outputs=torch.ones_like(u_pred),
                                     create_graph=True)[0][:, 2:3]
            
            # 二阶导数
            u_xx = torch.autograd.grad(u_x, X_f_batch,
                                         grad_outputs=torch.ones_like(u_x),
                                         create_graph=True)[0][:,0:1]
            
            
            u_yy = torch.autograd.grad(u_y, X_f_batch,
                                         grad_outputs=torch.ones_like(u_y),
                                         create_graph=True)[0][:,1:2]
            
            
            
            # PDE残差计算
            laplacian_u = u_xx + u_yy
            residual_pde = 0.01*laplacian_u - u_t + u_pred*(2*u_x + u_y) - f_source(X_f_batch[:, 0:1], X_f_batch[:, 1:2])
            loss_pde = loss_fn(residual_pde, torch.zeros_like(residual_pde))
            
            # 边界损失分解
            h_left = net(X_b_train_left)
            h_right = net(X_b_train_right)
            # (1) 固定值边界
            bc_loss = loss_fn(h_left- (-4), torch.zeros_like(h_left)) +loss_fn(h_right- (2), torch.zeros_like(h_right))
            
            # 边界损失分解
            h_left_2 = net(X_b_train_left_2)
            h_right_2 = net(X_b_train_right_2)
            # (1) 固定值边界
            bc_loss_2 = loss_fn(h_left_2, h_right_2) 
            
            

            
            # 初始条件损失
            U0_pred = net(X0_train)
            ic_loss = loss_fn(U0_pred, U0_train)
            
            # 总损失
            total_loss = loss_pde + bc_loss +bc_loss_2+ic_loss
            total_loss.backward()
            optimizer.step()
            
            loss_history.append(total_loss.item())
            loss_r = total_loss.item() 
            
            if n_r % 5000 == 0:
                print(f"  Step {n_r}/{inner_max_steps}, Loss: {total_loss.item():.3e}")
            
            n_r += 1  # 递增循环计数器

        # 残差评估与自适应采样 ------------------------------------------------
        # 生成测试点
        # x_test = np.random.uniform(x_min, x_max, (10000,1))
        # y_test = np.random.uniform(y_min, y_max, (10000,1))
        # t_test = np.random.uniform(0, t_max, (10000,1))
        # X_test = np.hstack((x_test, y_test, t_test))
        # X_test_tensor = torch.tensor(X_test, dtype=torch.float32, requires_grad=True).to(device)
        
        X_test = torch.rand(25000, 3).to(device)
        X_test[:, 0] = X_test[:, 0] * 4 - 2  # 将第一列调整到 [-2, 2]
        X_test[:, 1] = X_test[:, 1] * 4 - 2  # 将第一列调整到 [-2, 2]
        X_test[:, 2] = X_test[:, 2]  # 将第二列调整到 [0, T_max]
        X_test_tensor = X_test.requires_grad_(True)
        
        # 计算残差
        
        u_test = net(X_test_tensor)
        grad_test = torch.autograd.grad(u_test, X_test_tensor,
                                          grad_outputs=torch.ones_like(u_test),
                                          create_graph=True)[0]
            
        u_x_t = grad_test[:,0:1]
        u_y_t = grad_test[:,1:2]
        u_t_t = grad_test[:,2:3]
            
        grad_ux = torch.autograd.grad(u_x_t, X_test_tensor,
                                        grad_outputs=torch.ones_like(u_x_t),
                                        create_graph=True)[0]
        u_xx_t = grad_ux[:,0:1]
        
        grad_uy = torch.autograd.grad(u_y_t, X_test_tensor,
                                        grad_outputs=torch.ones_like(u_x_t),
                                        create_graph=True)[0]
        
        u_yy_t = grad_uy[:,1:2]
            
        residual = (mu*(u_xx_t + u_yy_t) - u_t_t + u_test*(2*u_x_t + u_y_t) 
                       - f_source(X_test_tensor[:,0:1], X_test_tensor[:,1:2])).detach().cpu().numpy()
        # residual = residual.detach().cpu().numpy()    
        
        mean_res = np.mean(np.abs(residual))
        print(f"平均残差: {mean_res:.3e}")
        
        if mean_res < residual_threshold:
            print("  残差已低于阈值，停止训练。")
            break
        
        # 添加新采样点
        
        X_add = add_data(X_test_tensor.detach().cpu().numpy(), residual, add_k=add_k)
        print(f"  添加了 {add_k} 个新的采样点。")
        
        n += 1
        print(f"当前训练集大小: {X_f_batch.shape[0]}")
        
    print(f"\nTraining Completed in {time.time()-train_start_time:.1f}s")
    print(f"Final Loss: {loss_history[-1]:.3e}")
    
    newtime=time.time()-train_start_time
    
    
    np.savez('2d_mu2_PINN_RAR_loss_history.npz', loss_history)
    # 可视化训练损失
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss History with Fixed Sampling')
    plt.grid(True)
    plt.show()
    
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
    u_pred_t_0_7 = net(torch.cat([x_t_0_7, y_t_0_7, t_t_0_7], dim=1)).detach().cpu().numpy()
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

    x_t_0_7 = df['x'].values
    y_t_0_7 = df['y'].values
    t_t_0_7 = df['t'].values

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
                              rstride=1,          # 必须设为1
                              cstride=1,          # 必须设为1
                              edgecolor='none',   # 边缘透明
                              linewidth=0,        # 彻底消除线宽
                              antialiased=False,  # 关闭抗锯齿
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
    def plot_U0(x, y, U0):
        # 转换为网格形式
        X_unique = np.unique(x)
        Y_unique = np.unique(y)
        X, Y = np.meshgrid(X_unique, Y_unique)
        Z = U0.reshape(len(Y_unique), len(X_unique))

        # 使用统一绘图函数
        fig = plot_3d_surface(X, Y, Z, 
                              "Predicted Solution of PINN with RAR (t=0.5)",
                              cmap='plasma',
                              elev=25, 
                              azim=135)
        
        # 保存和显示
        plt.tight_layout()
        plt.savefig('2d_PINN_RAR_mu2_solution_t_0_5.eps', dpi=300, bbox_inches='tight')
        plt.show()

    # 修改后的真解绘制函数
    def plot_true_solution(x, y, u):
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
        plt.savefig('2d_PINN_RAR_mu2_true_solution_t_0_5.eps', dpi=300, bbox_inches='tight')
        plt.show()

    # 绘制结果
    plot_U0(x_t_0_7, y_t_0_7, u_pred_t_0_7)
    plot_true_solution(x_true_t_0_7, y_true_t_0_7, true_u_t_0_7)


    # def plot_true_contour(x, y, u):
    #     # 转换为网格形式
    #     X_unique = np.unique(x)
    #     Y_unique = np.unique(y)
    #     X, Y = np.meshgrid(X_unique, Y_unique)
    #     Z = u.reshape(len(Y_unique), len(X_unique))

    #     # ================= 等值线图设置 =================
    #     plt.rcParams.update({'font.size': 24})  # 全局字体设置
        
    #     fig = plt.figure(figsize=(12, 9))
    #     ax = fig.add_subplot(111)
        
    #     # 使用plasma配色方案绘制等值线填充图
    #     contour = ax.contourf(X, Y, Z, 
    #                           levels=50, 
    #                           cmap='plasma',
    #                           antialiased=True)
        
    #     # ========== 坐标轴优化设置 ==========
    #     ax.set_xlabel('x', fontsize=28, labelpad=20, fontweight='bold')
    #     ax.set_ylabel('y', fontsize=28, labelpad=20, fontweight='bold')
        
    #     ax.tick_params(axis='both',
    #                   which='major',
    #                   labelsize=24,
    #                   width=2,
    #                   length=6,
    #                   direction='inout')
        
    #     # ========== 颜色条专业设置 ==========
    #     cbar = fig.colorbar(contour, ax=ax, 
    #                         fraction=0.046, 
    #                         pad=0.04)
    #     cbar.ax.tick_params(labelsize=24, 
    #                         width=2, 
    #                         length=6,
    #                         direction='inout')
    #     cbar.set_label('u', 
    #                   fontsize=24, 
    #                   labelpad=20,
    #                   rotation=270)
        
    #     # ========== 标题和布局优化 ==========
    #     plt.title('Reference Solution (t=0.4)', 
    #               fontsize=28, 
    #               pad=25,
    #               fontweight='bold')
        
    #     plt.tight_layout(pad=3.0)
    #     plt.savefig('2d_PINN_mu2_true_contour.eps', 
    #                 dpi=300, 
    #                 bbox_inches='tight',
    #                 transparent=True)
    #     plt.show()

    # # 使用示例
    # plot_true_contour(x_true_t_0_7, y_true_t_0_7, true_u_t_0_7)

    # def plot_appro_contour(x, y, u):
    #     # 转换为网格形式
    #     X_unique = np.unique(x)
    #     Y_unique = np.unique(y)
    #     X, Y = np.meshgrid(X_unique, Y_unique)
    #     Z = u.reshape(len(Y_unique), len(X_unique))

    #     # ================= 等值线图设置 =================
    #     plt.rcParams.update({'font.size': 24})  # 全局字体设置
        
    #     fig = plt.figure(figsize=(12, 9))
    #     ax = fig.add_subplot(111)
        
    #     # 使用plasma配色方案绘制等值线填充图
    #     contour = ax.contourf(X, Y, Z, 
    #                           levels=50, 
    #                           cmap='plasma',
    #                           antialiased=True)
        
    #     # ========== 坐标轴优化设置 ==========
    #     ax.set_xlabel('x', fontsize=28, labelpad=20, fontweight='bold')
    #     ax.set_ylabel('y', fontsize=28, labelpad=20, fontweight='bold')
        
    #     ax.tick_params(axis='both',
    #                   which='major',
    #                   labelsize=24,
    #                   width=2,
    #                   length=6,
    #                   direction='inout')
        
    #     # ========== 颜色条专业设置 ==========
    #     cbar = fig.colorbar(contour, ax=ax, 
    #                         fraction=0.046, 
    #                         pad=0.04)
    #     cbar.ax.tick_params(labelsize=24, 
    #                         width=2, 
    #                         length=6,
    #                         direction='inout')
    #     cbar.set_label('u', 
    #                   fontsize=24, 
    #                   labelpad=20,
    #                   rotation=270)
        
    #     # ========== 标题和布局优化 ==========
    #     plt.title('Predicted Solution of DAE (t=0.4)', 
    #               fontsize=28, 
    #               pad=25,
    #               fontweight='bold')
        
    #     plt.tight_layout(pad=3.0)
    #     plt.savefig('2d_PINN_mu2_appro_contour.eps', 
    #                 dpi=300, 
    #                 bbox_inches='tight',
    #                 transparent=True)
    #     plt.show()

    # # 使用示例
    # plot_appro_contour(x_t_0_7, y_t_0_7, u_pred_t_0_7)

    def plot_error(x, y, error):
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
        # cbar.set_label('Absolute Error', 
        #               fontsize=24, 
        #               labelpad=20,
        #               rotation=270)
        
        # ========== 标题和布局优化 ==========
        plt.title('Absolute Error of PINN with RAR (t=0.5)', 
                  fontsize=28, 
                  pad=25,
                  fontweight='bold')
        
        plt.tight_layout(pad=3.0)
        plt.savefig('2d_PINN_RAR_mu2_error_contour.eps', 
                    dpi=300, 
                    bbox_inches='tight',
                    transparent=True)
        plt.show()

    # 使用示例
    plot_error(x_t_0_7, y_t_0_7, error_values)
    # 已知预测解u_pred_t_0_7和真解true_u_t_0_7的求解过程
    
    
    print(f'Relative L2 Error: {error}')

    print(f"Maximum absolute error: {max_absolute_error}")

    print(f"Final Loss: {loss_history[-1]:.3e}")
    
    print(newtime)



if __name__ == '__main__':
    main()




        