import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class TrajectoryGenerator:
    def __init__(self):
        pass

    def generate_trajectory(self, path, avg_speed=2.0, env=None):
        """
        在基础样条平滑基础上，增加了动力学计算和碰撞检查逻辑
        """
        path = np.array(path)
        if len(path) < 2:
            return None, None, None, None, False

        # 1. 时间分配 (基于弦长参数化)
        diffs = np.diff(path, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        waypoints_t = cum_dist / avg_speed
        total_time = waypoints_t[-1]

        # 2. 生成样条曲线 (C2 连续)
        # bc_type='clamped' 确保起终点速度为 0，符合机器人起停逻辑
        cs = CubicSpline(waypoints_t, path, axis=0, bc_type='clamped')

        # 3. 采样并计算导数 (位置、速度、加速度)
        sample_dt = 0.05
        t_eval = np.arange(0, total_time, sample_dt)
        trajectory = cs(t_eval)
        velocity = cs.derivative(1)(t_eval)   # 一阶导：速度
        acceleration = cs.derivative(2)(t_eval) # 二阶导：加速度

        # 4. 安全性二次校验 [新增]
        # 虽然路径点是无碰撞的，但样条插值出的曲线可能切过障碍物
        is_safe = True
        if env is not None:
            for pt in trajectory[::4]: # 步长采样检查，平衡效率与安全
                if env.is_collide(pt):
                    is_safe = False
                    break

        return t_eval, trajectory, velocity, acceleration, is_safe

    def visualize_dynamics(self, t_eval, trajectory, velocity, acceleration, path):
        """
        生成专业的实验分析图：包含位置、速度、加速度三个维度
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        labels = ['x', 'y', 'z']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 专业配色
        
        # 为了标记原始路径点，重新计算路径点的时间戳
        diffs = np.diff(path, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        avg_speed = cum_dist[-1] / t_eval[-1] if t_eval[-1] > 0 else 1
        path_t = cum_dist / avg_speed

        # 子图 1: 位置 (Position)
        for i in range(3):
            axes[0].plot(t_eval, trajectory[:, i], color=colors[i], label=f'Pos {labels[i]}')
            axes[0].scatter(path_t, path[:, i], color='black', s=20, zorder=5)
        axes[0].set_ylabel('Position (m)')
        axes[0].set_title('Trajectory Dynamics Analysis')
        axes[0].legend(loc='right')

        # 子图 2: 速度 (Velocity)
        for i in range(3):
            axes[1].plot(t_eval, velocity[:, i], color=colors[i], linestyle='--', label=f'Vel {labels[i]}')
        axes[1].set_ylabel('Velocity (m/s)')
        axes[1].legend(loc='right')

        # 子图 3: 加速度 (Acceleration)
        for i in range(3):
            axes[2].plot(t_eval, acceleration[:, i], color=colors[i], linestyle=':', label=f'Acc {labels[i]}')
        axes[2].set_ylabel('Acc (m/s²)')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend(loc='right')

        for ax in axes: ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("trajectory_analysis.png", dpi=300) # 建议保存为高分辨率图用于报告
        plt.show()