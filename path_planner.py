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
import random  # [新增] 为了 RRT 算法引入

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

    # -----------------------------------------------------------------------
    # [新增功能] 下面的代码是为了丰富作业内容而新增的高级功能
    # -----------------------------------------------------------------------

    def simplify_path(self, path):
        """
        路径平滑/剪枝算法 (Floyd's Algorithm 变体)
        原理：如果节点 i 和节点 k 之间没有障碍物，则直接连接 i->k，跳过中间的节点 j。
        这可以将 A* 的"锯齿状"路径拉直。
        """
        if len(path) < 3:
            return np.array(path)

        # 转换为列表便于操作
        path = [np.array(p) for p in path]
        simplified = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # 贪心策略：从当前点尽可能往后找，找到最远的一个可见点
            # 我们倒序遍历，一旦找到一个无碰撞连接，就是最远的
            for check_idx in range(len(path) - 1, current_idx, -1):
                if not self._check_line_collision(path[current_idx], path[check_idx]):
                    # 发现从 current 直接连到 check_idx 是安全的
                    simplified.append(path[check_idx])
                    current_idx = check_idx
                    break
            else:
                # 理论上只要相邻点连通就不会进这里，但为了安全起见：
                current_idx += 1
                simplified.append(path[current_idx])

        return np.array(simplified)

    def plan_rrt(self, start_idx, goal_idx, max_iter=3000, step_size=1.0):
        """
         RRT (Rapidly-exploring Random Tree) 算法
        这是一个基于采样的算法，与 A* (基于搜索) 形成鲜明对比。
        """
        start_node = np.array(start_idx)
        goal_node = np.array(goal_idx)

        # 树结构: 列表存储节点，每个节点是 (坐标, 父节点索引)
        # 初始时只有起点，父节点索引为 None
        node_list = [(start_node, None)]

        print("Running RRT planner...")

        for i in range(max_iter):
            # 1. 采样 (Sampling)
            # 50% 概率采样目标点 (Goal Bias)，加快收敛；50% 随机采样
            if random.random() < 0.5:
                rnd_point = goal_node
            else:
                rnd_point = np.array([
                    random.uniform(0, self.env.env_width),
                    random.uniform(0, self.env.env_length),
                    random.uniform(0, self.env.env_height)
                ])

            # 2. 寻找最近邻 (Nearest Neighbor)
            # 计算采样点与树中所有节点的距离
            dists = [np.linalg.norm(node[0] - rnd_point) for node in node_list]
            nearest_idx = np.argmin(dists)
            nearest_node = node_list[nearest_idx][0]

            # 3. 扩展 (Steer)
            # 从最近节点向采样点延伸 step_size 的距离
            direction = rnd_point - nearest_node
            length = np.linalg.norm(direction)

            if length == 0:
                continue

            # 归一化并计算新位置
            new_pos = nearest_node + (direction / length) * min(step_size, length)

            # 4. 碰撞检测 (Collision Check)
            if not self.env.is_outside(new_pos) and \
               not self.env.is_collide(new_pos) and \
               not self._check_line_collision(nearest_node, new_pos):

                # 添加新节点到树中
                node_list.append((new_pos, nearest_idx))

                # 5. 判断是否到达目标 (Goal Reached)
                if np.linalg.norm(new_pos - goal_node) <= step_size:
                    # 尝试直接连接到终点
                    if not self._check_line_collision(new_pos, goal_node):
                        print(f"RRT reached goal in {i} iterations!")
                        # 添加终点并回溯路径
                        node_list.append((goal_node, len(node_list)-1))
                        return self._backtrack_rrt(node_list)

        print("RRT Failed to find a path!")
        return np.array([start_idx])

    def _check_line_collision(self, p1, p2):
        """
         辅助函数：检测两点 p1, p2 之间的线段是否与障碍物碰撞
        通过在两点之间进行插值采样来检查
        """
        p1 = np.array(p1)
        p2 = np.array(p2)
        dist = np.linalg.norm(p1 - p2)
        if dist < 1e-3: return False

        # 采样步长，必须小于障碍物的最小特征，这里选0.05以提高精度
        step = 0.05
        num_samples = int(math.ceil(dist / step))

        for i in range(num_samples + 1):
            t = i / num_samples
            point = p1 + (p2 - p1) * t
            if self.env.is_collide(point):
                return True
        return False

    def _backtrack_rrt(self, node_list):
        """ 从 RRT 树中回溯路径"""
        path = []
        current_idx = len(node_list) - 1
        while current_idx is not None:
            node, parent_idx = node_list[current_idx]
            path.append(node)
            current_idx = parent_idx
        path.reverse()
        return np.array(path)