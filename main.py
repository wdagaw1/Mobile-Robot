from flight_environment import FlightEnvironment
from path_planner import PathPlanner
from trajectory_generator import TrajectoryGenerator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

env = FlightEnvironment(50)
start = (1, 2, 0)
goal = (18, 18, 3)

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here.
# The planner should return a collision-free path and store it in the variable `path`.
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.

# path = [[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
# Call your path planning algorithm here.
print("Planning path...")
planner = PathPlanner(env)

# --- 1. 运行原始 A* 算法 ---
raw_path = planner.plan(start, goal)
print(f"Original A* path found with {len(raw_path)} points.")

# --- 2.  运行路径平滑算法 ---
# 这会让路径看起来更直，减少不必要的转弯
smooth_path = planner.simplify_path(raw_path)
print(f"Smoothed A* path has {len(smooth_path)} points.")

# --- 3.  运行 RRT 算法 (用于对比实验) ---
# 这是一个完全不同的算法，可以在报告中对比它和 A* 的区别
print("Running RRT...")
# RRT 是随机算法，如果不成功可以尝试增大 plan_rrt 中的 max_iter
rrt_path = planner.plan_rrt(start, goal)
print(f"RRT path found with {len(rrt_path)} points.")

# --- 4. 决定最终用于可视化的路径 ---
# 我们使用平滑后的路径，因为它效果最好
path = smooth_path
# 确保 path 是 numpy array 格式
path = np.array(path)

# --------------------------------------------------------------------------------------------------- #
# [修改] 为了区分颜色，我们在这里手动绘制 3D 场景，而不是调用 env.plot_cylinders(path)
# print("Displaying 3D Path Comparison...")
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # 1. 绘制障碍物 (复用 env 中的数据)
# for cx, cy, h, r in env.cylinders:
#     z = np.linspace(0, h, 30)
#     theta = np.linspace(0, 2 * np.pi, 30)
#     theta, z = np.meshgrid(theta, z)
#     x = cx + r * np.cos(theta)
#     y = cy + r * np.sin(theta)
#     ax.plot_surface(x, y, z, color='skyblue', alpha=0.3)
#     # 绘制顶盖
#     theta2 = np.linspace(0, 2 * np.pi, 30)
#     x_top = cx + r * np.cos(theta2)
#     y_top = cy + r * np.sin(theta2)
#     z_top = np.ones_like(theta2) * h
#     ax.plot_trisurf(x_top, y_top, z_top, color='steelblue', alpha=0.3)

# # 2. 绘制 A* 路径 (蓝色)
# if len(smooth_path) > 1:
#     p1 = np.array(smooth_path)
#     ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], label='A* (Smoothed)', color='blue', linewidth=3, marker='o', markersize=3)

# # 3. 绘制 RRT 路径 (红色虚线) - 只有找到路径才画
# if len(rrt_path) > 1:
#     p2 = np.array(rrt_path)
#     ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], label='RRT', color='red', linewidth=3, linestyle='--', marker='^',
#             markersize=3)
# else:
#     print("Warning: RRT failed to generate a valid path, skipping 3D plot for RRT.")

# # 设置坐标轴和比例 (复用 env 的 helper 函数)
# ax.set_xlim(0, env.env_width)
# ax.set_ylim(0, env.env_length)
# ax.set_zlim(0, env.env_height)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title("3D Path Comparison: A* (Blue) vs RRT (Red)")
# ax.legend(loc='upper right')  # 指定位置避免 warning
# env.set_axes_equal(ax)
# plt.show()

# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.

# 1. 引入类 (放在文件顶部或这里)
print("Generating trajectory...")
traj_gen = TrajectoryGenerator()

# [修改] 分别生成两条轨迹，增加对 RRT 是否成功的判断
t_eval_a, traj_a, vel_a, acc_a, safe_a = traj_gen.generate_trajectory(smooth_path, avg_speed=3.0, env=env)

print(f"A* Trajectory generated. Safety check: {'Pass' if safe_a else 'Fail'}")

# 初始化 RRT 轨迹数据为 None
t_eval_r, traj_r, vel_r, acc_r, safe_r = None, None, None, None, False
if len(rrt_path) > 1:
    # 同样修改为 5 个返回值
    t_eval_r, traj_r, vel_r, acc_r, safe_r = traj_gen.generate_trajectory(rrt_path, avg_speed=3.0, env=env)
    print(f"RRT Trajectory generated. Safety check: {'Pass' if safe_r else 'Fail'}")
else:
    print("Warning: RRT path too short, skipping RRT trajectory generation.")

#   After generating the trajectory, plot it in a new figure.
# [修改] 手动绘制对比图，区分颜色，并处理 RRT 可能为空的情况
print("Displaying 3D Path Comparison (Poly3DCollection Optimized)...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# --- 1. 使用 Poly3DCollection 批量绘制障碍物 ---
all_faces = []
res_theta = 8  # 八棱柱，兼顾视觉和性能

for cx, cy, h, r in env.cylinders:
    # 计算底面和顶面的顶点
    theta = np.linspace(0, 2 * np.pi, res_theta, endpoint=False)
    x_base = cx + r * np.cos(theta)
    y_base = cy + r * np.sin(theta)
    
    # 构造侧面面片
    for i in range(res_theta):
        next_i = (i + 1) % res_theta
        # 每个侧面由 4 个顶点组成
        face = [
            [x_base[i], y_base[i], 0],
            [x_base[next_i], y_base[next_i], 0],
            [x_base[next_i], y_base[next_i], h],
            [x_base[i], y_base[i], h]
        ]
        all_faces.append(face)
    
    # 构造顶盖面片
    top_face = [[x_base[i], y_base[i], h] for i in range(res_theta)]
    all_faces.append(top_face)

# 创建集合：设置不透明度 alpha=1, 关闭阴影
poly_collection = Poly3DCollection(all_faces, facecolors='skyblue', edgecolors='steelblue', alpha=1.0, shade=False)
ax.add_collection3d(poly_collection)

# --- 2. 绘制路径 (保持不变，但增加 zorder 确保在障碍物上方) ---
if len(smooth_path) > 1:
    p1 = np.array(smooth_path)
    ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], label='A* (Smoothed)', color='blue', linewidth=3, zorder=10)

if len(rrt_path) > 1:
    p2 = np.array(rrt_path)
    ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], label='RRT', color='red', linewidth=2, linestyle='--', zorder=11)

# --- 3. 场景设置 ---
ax.set_xlim(0, env.env_width)
ax.set_ylim(0, env.env_length)
ax.set_zlim(0, env.env_height)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Optimized 3D View: A* vs RRT")

if traj_a is not None:
    ax.plot(traj_a[:, 0], traj_a[:, 1], traj_a[:, 2], color='cyan', linewidth=1, alpha=0.8, label='A* Trajectory')

# 绘制 RRT 的连续平滑轨迹 (用浅红色实线)
if traj_r is not None:
    ax.plot(traj_r[:, 0], traj_r[:, 1], traj_r[:, 2], color='salmon', linewidth=1, alpha=0.8, label='RRT Trajectory')

ax.legend()
env.set_axes_equal(ax)
plt.show()

if traj_a is not None:
    # 使用我们新改进的动力学分析绘图
    traj_gen.visualize_dynamics(t_eval_a, traj_a, vel_a, acc_a, smooth_path)

# --------------------------------------------------------------------------------------------------- #


# You must manage this entire project using Git.
# When submitting your assignment, upload the project to a code-hosting platform
# such as GitHub or GitLab. The repository must be accessible and directly cloneable.
#
# After cloning, running `python3 main.py` in the project root directory
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.