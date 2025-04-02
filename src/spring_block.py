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
    # TODO: 创建存储位置和速度的数组
    position = np.zeros(step_num + 1)
    velocity = np.zeros(step_num + 1)

    # TODO: 计算时间步长
    time_step = t_max / step_num

    # TODO: 设置初始位置和速度
    # 初始条件（初始位移1m，静止释放）
    position[0] = 1.0
    velocity[0] = 0.0
    # TODO: 使用欧拉法迭代求解微分方程
    # 欧拉法迭代
    for i in range(step_num):
        dx_dt, dv_dt = spring_mass_ode_func([position[i], velocity[i]], 0)
        position[i+1] = position[i] + dx_dt * time_step
        velocity[i+1] = velocity[i] + dv_dt * time_step
    # TODO: 生成时间数组
    time_points = np.linspace(0, t_max, step_num + 1)

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
    # TODO: 从状态中提取位置和速度
    x, v = state
    # TODO: 计算位置和速度的导数
    dx_dt = v  # 位置导数 = 速度
    dv_dt = -x  # 简谐运动方程 (k/m=1)
    
    return [dx_dt, dv_dt]  # 替换为正确的返回值


def solve_ode_odeint(step_num):
    """
    使用 odeint 求解弹簧 - 质点系统的常微分方程。

    参数:
    step_num (int): 模拟的步数

    返回:
    tuple: 包含时间数组、位置数组和速度数组的元组
    """
    # TODO: 设置初始条件
    initial_state = [1.0, 0.0]  # 初始条件与欧拉法一致
    
    # TODO: 创建时间点数组
    time_points = np.linspace(0, t_max, step_num + 1)
    
    # TODO: 使用 odeint 求解微分方程
    solution = odeint(spring_mass_ode_func, initial_state, time_points)
    
    # TODO: 从解中提取位置和速度
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
    # TODO: 创建图形并设置大小
    plt.figure(figsize=(12, 8))
    # TODO: 绘制位置对比图
    plt.subplot(2, 1, 1)
    plt.plot(time_euler, position_euler, 'b--', label='Euler Method', linewidth=2)
    plt.plot(time_odeint, position_odeint, 'r-', label='odeint', alpha=0.7)
    plt.ylabel('Position (m)')
    plt.title('Position Comparison')
    plt.legend()
    plt.grid(True)
    # TODO: 绘制速度对比图
    plt.subplot(2, 1, 2)
    plt.plot(time_euler, velocity_euler, 'b--', label='Euler Method', linewidth=2)
    plt.plot(time_odeint, velocity_odeint, 'r-', label='odeint', alpha=0.7)
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.title('Velocity Comparison')
    plt.legend()
    plt.grid(True)
    # TODO: 显示图形
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 模拟步数
    step_count = 100
    # TODO: 使用欧拉法求解
    time_euler, position_euler, velocity_euler = solve_ode_euler(step_count)
    # TODO: 使用 odeint 求解
    time_odeint, position_odeint, velocity_odeint = solve_ode_odeint(step_count)
    # TODO: 绘制对比结果
    plot_ode_solutions(time_euler, position_euler, velocity_euler, time_odeint, position_odeint, velocity_odeint)
