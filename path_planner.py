"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""

import numpy as np
import heapq
import math

import numpy as np
import heapq
import math


class PathPlanner:
    def __init__(self, env):
        self.env = env
        # 设置网格分辨率，越小越精确但越慢。0.5是一个平衡的选择。
        self.resolution = 0.5
        # 定义26个运动方向（包括对角线），使路径更平滑
        self.motions = self._get_motions()

    def plan(self, start_idx, goal_idx):
        """
        A* 路径规划算法实现
        :param start_idx: 起点坐标 (x, y, z)
        :param goal_idx: 终点坐标 (x, y, z)
        :return: 路径点列表 (Nx3 numpy array)
        """
        # 将起点和终点对齐到网格中心
        start_node = self._discretize(start_idx)
        goal_node = self._discretize(goal_idx)

        # 优先队列 (cost, current_node)
        # cost = g (从起点移动的代价) + h (到终点的启发式估算)
        open_list = []
        heapq.heappush(open_list, (0, start_node))

        # 记录每个节点的最小代价 g，避免重复搜索
        g_costs = {start_node: 0}

        # 记录父节点以便回溯路径: parent[child] = father
        came_from = {start_node: None}

        # 简单的启发式函数缓存
        goal_arr = np.array(goal_node)

        while open_list:
            _, current = heapq.heappop(open_list)

            # 如果到达终点附近（考虑到离散化误差，允许一定范围的误差）
            if self._dist(current, goal_node) < self.resolution:
                return self._reconstruct_path(came_from, current, goal_idx)

            current_g = g_costs[current]

            # 遍历所有可能的移动方向
            for dx, dy, dz, move_cost in self.motions:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                # 检查边界和碰撞
                # 注意：将网格索引转换回实际坐标进行检查
                check_point = neighbor  # 这里的 neighbor 已经是实际坐标了，因为我们是累加分辨率

                # 预先检查边界，节省碰撞检测开销
                if self.env.is_outside(neighbor):
                    continue

                # 检查碰撞
                if self.env.is_collide(neighbor):
                    continue

                new_g = current_g + move_cost

                # 如果发现了更短的路径到达该邻居，或者该邻居未被访问过
                if neighbor not in g_costs or new_g < g_costs[neighbor]:
                    g_costs[neighbor] = new_g
                    priority = new_g + self._heuristic(neighbor, goal_arr)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current

        print("Failed to find a path!")
        return np.array([start_idx])  # 如果失败，返回起点

    def _get_motions(self):
        """生成3D空间的26个移动方向及其代价"""
        motions = []
        for dx in [-self.resolution, 0, self.resolution]:
            for dy in [-self.resolution, 0, self.resolution]:
                for dz in [-self.resolution, 0, self.resolution]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cost = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    motions.append((dx, dy, dz, cost))
        return motions

    def _discretize(self, point):
        """将连续坐标吸附到最近的网格点"""
        x, y, z = point
        return (round(x / self.resolution) * self.resolution,
                round(y / self.resolution) * self.resolution,
                round(z / self.resolution) * self.resolution)

    def _heuristic(self, node, goal):
        """欧几里得距离启发式函数"""
        return np.linalg.norm(np.array(node) - goal)

    def _dist(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _reconstruct_path(self, came_from, current, real_goal):
        """从终点回溯到起点重建路径"""
        path = []
        path.append(real_goal)  # 确保终点是精确的输入终点
        while current is not None:
            # 只有当当前点与路径上一个点距离足够大时才添加，避免路径点过密
            if not path or self._dist(current, path[-1]) > 1e-3:
                path.append(current)
            current = came_from[current]
        path.reverse()
        return np.array(path)









