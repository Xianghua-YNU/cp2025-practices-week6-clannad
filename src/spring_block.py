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

    # 计算时间步长，假设总时间为10
    time_step = 10 / step_num

    # 设置初始位置和速度
    position[0] = 1.0
    velocity[0] = 0.0

    # 弹簧常数 k = 1, 质量 m = 1
    k = 1.0
    m = 1.0

    # 使用欧拉法迭代求解微分方程
    for i in range(step_num - 1):
        acceleration = -k * position[i] / m
        velocity[i + 1] = velocity[i] + acceleration * time_step
        position[i + 1] = position[i] + velocity[i] * time_step

    # 生成时间数组
    time_points = np.linspace(0, 10, step_num)

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

    # 计算位置和速度的导数，假设 k = 1, m = 1
    dxdt = v
    dvdt = -x

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
    initial_state = [1.0, 0.0]

    # 创建时间点数组，假设总时间为10
    time_points = np.linspace(0, 10, step_num)

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
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # 绘制位置对比图
    axes[0].plot(time_euler, position_euler, label='Euler Position')
    axes[0].plot(time_odeint, position_odeint, label='odeint Position')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Position')
    axes[0].legend()

    # 绘制速度对比图
    axes[1].plot(time_euler, velocity_euler, label='Euler Velocity')
    axes[1].plot(time_odeint, velocity_odeint, label='odeint Velocity')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Velocity')
    axes[1].legend()

    # 显示图形
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 模拟步数
    step_count = 100
    # 使用欧拉法求解
    time_euler, position_euler, velocity_euler = solve_ode_euler(step_count)
    # 使用 odeint 求解
    time_odeint, position_odeint, velocity_odeint = solve_ode_odeint(step_count)
    # 绘制对比结果
    plot_ode_solutions(time_euler, position_euler, velocity_euler, time_odeint, position_odeint, velocity_odeint)
