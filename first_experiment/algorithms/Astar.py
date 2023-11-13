import copy
import heapq
import time


class PriorityQueue:
    def __init__(self):
        self.priorityqueue = []
        self.index = 0

    def add(self, item, priority):
        heapq.heappush(self.priorityqueue, (priority, self.index, item))
        self.index += 1
        return

    def pop(self):
        return heapq.heappop(self.priorityqueue)[-1]


def get_children(current, game_map, last):
    children = []
    temp_children = [[current[0] + 1, current[1]], [current[0], current[1] + 1],
                     [current[0] - 1, current[1]], [current[0], current[1] - 1]]
    path = get_path(current, last)
    for child in temp_children:
        if game_map[child[0]][child[1]] and child not in path:
            children.append(tuple(child))
    return children


def get_path(current, last):
    path = [current]
    while last[current] != current:
        current = last[current]
        path.append(current)
    path.append(last[current])
    return path


def h_cost(child, goal):
    [x1, y1] = child
    [x2, y2] = goal
    return abs(x1 - x2) + abs(y1 - y2)


# 0：屏障 1：可走 2：起点 3：终点 4：已访问点 5：访问边界
def Astar(game_map, start, goal):
    open_list = PriorityQueue()
    close_list = []
    visited = {}
    last = {}
    ans_process = []
    game_map = copy.deepcopy(game_map)

    open_list.add(start, 0)
    last[start] = start
    visited[start] = 0

    start_time = time.time()
    while open_list.index:
        current = open_list.pop()
        if current != start and current != goal:
            game_map[current[0]][current[1]] = 4
        close_list.append(current)
        if current == goal:
            end_time = time.time()
            print(f'Astar {end_time - start_time}s')
            # print(get_path(goal, last))
            return True, ans_process
        else:
            for child in get_children(current, game_map, last):
                g_cost = visited[current] + 1
                if child not in visited or g_cost < visited[child]:
                    child_cost = g_cost + h_cost(child, goal)
                    visited[child] = g_cost
                    open_list.add(child, child_cost)
                    last[child] = current
                    if child != goal:
                        game_map[child[0]][child[1]] = 5
            ans_process.append(copy.deepcopy(game_map))

    return False, None


if __name__ == '__main__':
    maze_map = [[0] * 21,
                     [0] + [1] * 9 + [0] + [1] * 9 + [0],
                     [0] + [1] + [0] * 3 + [1, 0, 1, 0, 1] + [0] * 7 + [1, 0, 1, 0],
                     [0, 1, 0] + [1] * 3 + [0, 1, 0] + [1] * 7 + [0, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 0, 0, 1] + [0] * 7 + [1, 0, 1, 0, 1, 0],
                     [0, 2, 0, 1, 0, 1, 1, 1, 0] + [1] * 5 + [0, 1, 1, 1, 0, 1, 0],
                     [0, 0, 0, 1] + [0] * 5 + [1, 0, 1] + [0] * 7 + [1, 0],
                     [0] + [1] * 7 + [0, 1, 0] + [1] * 7 + [0, 1, 0],
                     [0] * 7 + [1, 0, 1] + [0] * 5 + [1, 0, 1, 0, 1, 0],
                     [0] + [1] * 7 + [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
                     [0, 1] + [0] * 5 + [1, 0, 1, 0, 0, 0, 1] + [0] * 7,
                     [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0] + [1] * 7 + [0],
                     [0, 1, 0, 1, 0, 1] + [0] * 5 + [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
                     [0, 1] * 6 + [0, 0, 0, 1] * 2 + [0],
                     [0, 1] * 5 + [1, 1, 0] + [1] * 3 + [0] + [1] * 3 + [0],
                     [0, 1] * 2 + [0] * 3 + [1] + [0] * 5 + [1] + [0] * 3 + [1] + [0] * 3,
                     [0] + [1] * 5 + [0] + [1] * 9 + [0, 1, 1, 3, 0],
                     [0] * 21
                     ]
    start = (5, 1)
    end = (17, 19)
    flag, ans = Astar(maze_map, start, end)
    for maze in ans:
        for line in maze:
            print(line)
        print("\n")
