import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def solve_ode_euler(step_num):
    """
    使用欧拉法求解弹簧-质点系统的常微分方程。

    参数:
    step_num (int): 模拟的步数

    返回:
    tuple: 包含时间数组、位置数组和速度数组的元组
    """
    # 参数设置
    t_max = 10.0  # 总模拟时间
    time_step = t_max / step_num
    
    # 初始化存储数组
    position = np.zeros(step_num + 1)
    velocity = np.zeros(step_num + 1)
    
    # 初始条件（初始位移1m，静止释放）
    position[0] = 1.0
    velocity[0] = 0.0
    
    # 时间数组
    time_points = np.linspace(0, t_max, step_num + 1)
    
    # 欧拉法迭代
    for i in range(step_num):
        dx_dt, dv_dt = spring_mass_ode_func([position[i], velocity[i]], 0)
        position[i+1] = position[i] + dx_dt * time_step
        velocity[i+1] = velocity[i] + dv_dt * time_step
    
    return time_points, position, velocity

def spring_mass_ode_func(state, time):
    """
    定义弹簧-质点系统的常微分方程。

    参数:
    state (list): 包含位置和速度的列表
    time (float): 时间

    返回:
    list: 包含位置和速度的导数的列表
    """
    x, v = state
    dx_dt = v  # 位置导数 = 速度
    dv_dt = -x  # 简谐运动方程 (k/m=1)
    return [dx_dt, dv_dt]

def solve_ode_odeint(step_num):
    """
    使用 odeint 求解弹簧-质点系统的常微分方程。

    参数:
    step_num (int): 模拟的步数

    返回:
    tuple: 包含时间数组、位置数组和速度数组的元组
    """
    t_max = 10.0  # 与欧拉法保持相同模拟时间
    initial_state = [1.0, 0.0]  # 初始条件与欧拉法一致
    time_points = np.linspace(0, t_max, step_num + 1)
    
    # 使用odeint求解
    solution = odeint(spring_mass_ode_func, initial_state, time_points)
    
    # 提取结果
    position = solution[:, 0]
    velocity = solution[:, 1]
    
    return time_points, position, velocity

def plot_ode_solutions(time_euler, position_euler, velocity_euler, 
                      time_odeint, position_odeint, velocity_odeint):
    """
    绘制两种方法求解结果的对比图
    """
    plt.figure(figsize=(12, 8))
    
    # 位置对比图
    plt.subplot(2, 1, 1)
    plt.plot(time_euler, position_euler, 'b--', label='Euler Method', linewidth=2)
    plt.plot(time_odeint, position_odeint, 'r-', label='odeint', alpha=0.7)
    plt.ylabel('Position (m)')
    plt.title('Position Comparison')
    plt.legend()
    plt.grid(True)
    
    # 速度对比图
    plt.subplot(2, 1, 2)
    plt.plot(time_euler, velocity_euler, 'b--', label='Euler Method', linewidth=2)
    plt.plot(time_odeint, velocity_odeint, 'r-', label='odeint', alpha=0.7)
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.title('Velocity Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 设置模拟参数
    step_count = 100
    
    # 使用不同方法求解
    time_euler, pos_euler, vel_euler = solve_ode_euler(step_count)
    time_odeint, pos_odeint, vel_odeint = solve_ode_odeint(step_count)
    
    # 绘制对比结果
    plot_ode_solutions(time_euler, pos_euler, vel_euler,
                      time_odeint, pos_odeint, vel_odeint)
