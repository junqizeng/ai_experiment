import copy
from queue import Queue
import time


def get_child_list(game_map, cur_pos):
    child_list = []
    change_list = [[0, -1], [1, 0], [-1, 0], [0, 1]]

    for change in change_list:
        next_row = cur_pos[0] + change[0]
        next_col = cur_pos[1] + change[1]
        next_val = game_map[next_row][next_col]
        if next_val == 1 or next_val == 3:
            child_list.append([next_row, next_col])
    return child_list


# 0：屏障 1：可走 2：起点 3：终点 4：已访问点 5：访问边界
def BFS_main(game_map, start_pos, target_pos):
    game_map = copy.deepcopy(game_map)
    bfs_queue = Queue()
    ans_process = []
    start_pos = list(start_pos)
    target_pos = list(target_pos)

    bfs_queue.put(start_pos)
    start_time = time.time()

    while not bfs_queue.empty():
        cur_pos = bfs_queue.get()
        cur_row = cur_pos[0]
        cur_col = cur_pos[1]
        cur_val = game_map[cur_row][cur_col]

        if cur_val == 3:
            end_time = time.time()
            print(f'BFS {end_time - start_time}s')
            return True, ans_process

        if cur_val != 2:
            game_map[cur_row][cur_col] = 4

        child_list = get_child_list(game_map=game_map, cur_pos=cur_pos)
        for child_pos in child_list:
            child_row = child_pos[0]
            child_col = child_pos[1]
            child_val = game_map[child_row][child_col]
            if child_val != 3:
                game_map[child_row][child_col] = 5
            bfs_queue.put(child_pos)
        ans_process.append(copy.deepcopy(game_map))

    return True, ans_process


if __name__ == '__main__':
    game_map = [[0] * 21,
                         [0] + [1] * 4 + [2] + [1] * 4 + [0] + [1] * 9 + [0],
                         [0] + [1] + [0] * 3 + [1, 0, 1, 0, 1] + [0] * 7 + [1, 0, 1, 0],
                         [0, 1, 0] + [1] * 3 + [0, 1, 0] + [1] * 7 + [0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 0, 0, 1] + [0] * 7 + [1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 1, 1, 0] + [1] * 5 + [0, 1, 1, 1, 0, 1, 0],
                         [0, 0, 0, 1] + [0] * 5 + [1, 0, 1] + [0] * 7 + [1, 0],
                         [0] + [1] * 7 + [0, 1, 0] + [1] * 7 + [0, 1, 0],
                         [0] * 9 + [1] + [0] * 5 + [1, 0, 1, 0, 1, 0],
                         [0] + [1] * 7 + [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
                         [0, 1] + [0] * 5 + [1, 0, 1, 0, 0, 0, 1] + [0] * 7,
                         [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0] + [1] * 7 + [0],
                         [0, 1, 0, 1, 0, 1] + [0] * 5 + [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
                         [0, 1] * 6 + [0, 0, 0, 1] * 2 + [0],
                         [0, 1] * 5 + [1, 1, 0] + [1] * 3 + [0] + [1] * 3 + [0],
                         [0, 1] * 2 + [0] * 3 + [1] + [0] * 5 + [1] + [0] * 3 + [1] + [0] * 3,
                         [0] + [1] * 15 + [0, 1, 1, 3, 0],
                         [0] * 21
                         ]
    start = (1, 5)
    end = (17, 19)
    ans_process = BFS_main(game_map=game_map, start_pos=start, target_pos=end)
