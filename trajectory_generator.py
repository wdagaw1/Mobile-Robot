import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class TrajectoryGenerator:
    def __init__(self):
        pass

    def generate_trajectory(self, path, avg_speed=2.0):
        """
        生成通过给定路径点的平滑轨迹。
        
        :param path: Nx3 numpy array, 离散路径点
        :param avg_speed: 假设的平均飞行速度 (m/s)，用于分配时间
        :return: (times, trajectory_points)
                 times: 时间序列
                 trajectory_points: Nx3 array, 平滑轨迹上的点
        """
        path = np.array(path)
        if len(path) < 2:
            return np.zeros((1, 3))

        # 1. 时间分配
        # 计算路径点之间的累积距离（弦长参数化）
        diffs = np.diff(path, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        
        # 根据距离和平均速度计算到达每个路点的时间
        # 这种方法比简单的等时间间隔更能反映几何形状
        waypoints_t = cum_dist / avg_speed
        total_time = waypoints_t[-1]

        # 2. 生成样条曲线 (Cubic Spline)
        # scipy 的 CubicSpline 默认处理 C2 连续性（位置、速度、加速度连续）
        # axis=0 表示对 [x, y, z] 分别进行插值
        # bc_type='clamped' 强制起点和终点的速度（一阶导数）为 0，这在机器人控制中很常见（起步和停止）
        cs = CubicSpline(waypoints_t, path, axis=0, bc_type='clamped')

        # 3. 采样轨迹
        # 生成高密度的时间点用于绘图和控制
        sample_dt = 0.05 # 采样间隔 0.05秒
        t_eval = np.arange(0, total_time, sample_dt)
        trajectory = cs(t_eval)

        return t_eval, trajectory, path

    def visualize(self, t_eval, trajectory, path):
        """
        可视化轨迹：x, y, z 分别随时间的变化
        """
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        
        labels = ['x', 'y', 'z']
        colors = ['r', 'g', 'b']
        
        # 原始路径点对应的时间估计（为了绘图对比）
        # 我们需要重新计算一下路径点的时间戳以便在图上标记原始点
        diffs = np.diff(path, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        avg_speed = cum_dist[-1] / t_eval[-1] if t_eval[-1] > 0 else 1
        path_t = cum_dist / avg_speed

        for i in range(3):
            # 绘制连续轨迹
            axes[i].plot(t_eval, trajectory[:, i], color=colors[i], linewidth=2, label=f'Trajectory {labels[i]}')
            # 绘制原始离散路径点
            axes[i].scatter(path_t, path[:, i], color='k', marker='o', s=30, label='Path Points', zorder=5)
            
            axes[i].set_ylabel(f'{labels[i]} (m)')
            axes[i].grid(True)
            axes[i].legend(loc='upper left')

        axes[2].set_xlabel('Time (s)')
        plt.suptitle('Trajectory vs Time (x, y, z)')
        plt.tight_layout()
        plt.show()