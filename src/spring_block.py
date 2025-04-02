import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def solve_ode_euler(step_num):
    """
    使用欧拉法求解弹簧 - 质点系统的常微分方程。

    参数:
    step_num (int): 模拟的步数

    返回:
    tuple: 包含时间数组、位置数组和速度数组的元组
    """
    # 创建存储位置和速度的数组
    position = np.zeros(step_num)
    velocity = np.zeros(step_num)

    # 计算时间步长 (模拟总时间设为10秒)
    total_time = 10.0
    time_step = total_time / step_num

    # 设置初始位置和速度
    position[0] = 1.0  # 初始位移
    velocity[0] = 0.0  # 初始速度

    # 使用欧拉法迭代求解微分方程
    for i in range(step_num - 1):
        # 计算当前加速度 (根据 Hooke's law: F = -kx, 这里k=1, m=1)
        acceleration = -position[i]
        
        # 更新速度和位置
        velocity[i+1] = velocity[i] + acceleration * time_step
        position[i+1] = position[i] + velocity[i] * time_step

    # 生成时间数组
    time_points = np.linspace(0, total_time, step_num)

    return time_points, position, velocity


def spring_mass_ode_func(state, time):
    """
    定义弹簧 - 质点系统的常微分方程。

    参数:
    state (list): 包含位置和速度的列表
    time (float): 时间

    返回:
    list: 包含位置和速度的导数的列表
    """
    # 从状态中提取位置和速度
    x, v = state
    
    # 计算位置和速度的导数
    dxdt = v
    dvdt = -x  # 弹簧常数为1，质量为1
    
    return [dxdt, dvdt]


def solve_ode_odeint(step_num):
    """
    使用 odeint 求解弹簧 - 质点系统的常微分方程。

    参数:
    step_num (int): 模拟的步数

    返回:
    tuple: 包含时间数组、位置数组和速度数组的元组
    """
    # 设置初始条件
    initial_state = [1.0, 0.0]  # 初始位置1.0，初始速度0.0
    
    # 创建时间点数组 (模拟总时间设为10秒)
    total_time = 10.0
    time_points = np.linspace(0, total_time, step_num)
    
    # 使用 odeint 求解微分方程
    solution = odeint(spring_mass_ode_func, initial_state, time_points)
    
    # 从解中提取位置和速度
    position = solution[:, 0]
    velocity = solution[:, 1]
    
    return time_points, position, velocity


def plot_ode_solutions(time_euler, position_euler, velocity_euler, time_odeint, position_odeint, velocity_odeint):
    """
    绘制欧拉法和 odeint 求解的位置和速度随时间变化的图像。

    参数:
    time_euler (np.ndarray): 欧拉法的时间数组
    position_euler (np.ndarray): 欧拉法的位置数组
    velocity_euler (np.ndarray): 欧拉法的速度数组
    time_odeint (np.ndarray): odeint 的时间数组
    position_odeint (np.ndarray): odeint 的位置数组
    velocity_odeint (np.ndarray): odeint 的速度数组
    """
    # 创建图形并设置大小
    plt.figure(figsize=(12, 6))
    
    # 绘制位置对比图
    plt.subplot(1, 2, 1)
    plt.plot(time_euler, position_euler, label='Euler Method', linestyle='--')
    plt.plot(time_odeint, position_odeint, label='odeint', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position vs Time')
    plt.legend()
    plt.grid(True)
    
    # 绘制速度对比图
    plt.subplot(1, 2, 2)
    plt.plot(time_euler, velocity_euler, label='Euler Method', linestyle='--')
    plt.plot(time_odeint, velocity_odeint, label='odeint', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Velocity vs Time')
    plt.legend()
    plt.grid(True)
    
    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 模拟步数
    step_count = 1000  # 增加步数以提高欧拉法的精度
    # 使用欧拉法求解
    time_euler, position_euler, velocity_euler = solve_ode_euler(step_count)
    # 使用 odeint 求解
    time_odeint, position_odeint, velocity_odeint = solve_ode_odeint(step_count)
    # 绘制对比结果
    plot_ode_solutions(time_euler, position_euler, velocity_euler, 
                      time_odeint, position_odeint, velocity_odeint)
