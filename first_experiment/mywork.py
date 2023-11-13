import os
from algorithms import Astar, DFS, BFS
from vis_windows import maze_panel, main_windows, wel_panel
from PyQt5.QtWidgets import QMessageBox, QPushButton
from PyQt5.QtGui import QIcon
from vis_windows.images import whu_ico
import base64


class MyWork:
    def __init__(self):
        self.maze_map = [[0] * 21,
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
        self.start = (1, 5)
        self.end = (17, 19)

        self.DFS_flag, self.DFS_ans = DFS.DFS_main(self.maze_map, self.start, self.end)
        self.BFS_flag, self.BFS_ans = BFS.BFS_main(self.maze_map, self.start, self.end)
        self.Astar_flag, self.Astar_ans = Astar.Astar(self.maze_map, self.start, self.end)

        self.step_index = 0
        self.ans_process = self.Astar_ans

        self.main_win = main_windows.MainWindow()

        tmp = open(f'./whu.ico', 'wb')
        tmp.write(base64.b64decode(whu_ico.imgs))
        tmp.close()
        self.main_win.setWindowIcon(QIcon('./whu.ico'))
        os.remove(f'./whu.ico')

        self.maze_panel = maze_panel.MazePanel(self.main_win)
        self.maze_panel.next_btn.clicked.connect(self.maze_next)
        self.maze_panel.last_btn.clicked.connect(self.maze_last)

        self.wel_panel = wel_panel.WelPanel(self.main_win)

        self.main_win.Astar_btn.clicked.connect(self.jump_to_Astar_maze)
        self.main_win.DFS_btn.clicked.connect(self.jump_to_DFS_maze)
        self.main_win.BFS_btn.clicked.connect(self.jump_to_BFS_maze)

        self.wel_panel.show()
        self.maze_panel.hide()

    def jump_to_Astar_maze(self):
        self.maze_panel.hide()
        self.step_index = -1

        self.maze_panel.steps = len(self.Astar_ans)
        self.maze_panel.algorithm = 'Astar'
        self.ans_process = self.Astar_ans
        self.maze_update(0)

        self.wel_panel.hide()
        self.maze_panel.show()

    def jump_to_DFS_maze(self):
        self.maze_panel.hide()
        self.step_index = -1

        self.maze_panel.steps = len(self.DFS_ans)
        self.maze_panel.algorithm = 'DFS'
        self.ans_process = self.DFS_ans
        self.maze_update(0)

        self.wel_panel.hide()
        self.maze_panel.show()

    def jump_to_BFS_maze(self):
        self.maze_panel.hide()
        self.step_index = -1

        self.maze_panel.steps = len(self.BFS_ans)
        self.maze_panel.algorithm = 'BFS'
        self.ans_process = self.BFS_ans
        self.maze_update(0)

        self.wel_panel.hide()
        self.maze_panel.show()

    def maze_update(self, index_change):
        if index_change == 0:
            self.step_index = -1
        elif self.step_index == 0 and index_change == -1:
            message_box = QMessageBox(self.main_win)
            message_box.setWindowTitle("运行提示")
            message_box.setText('<h3>%s</h3>' % "A*运行到第一步")
            message_box.setInformativeText("点击上一步，退出运行")
            message_box.setIcon(QMessageBox.Information)
            message_box.addButton(QPushButton('确定', message_box), QMessageBox.YesRole)
            message_box.show()
            self.step_index = - 1
        elif self.step_index == len(self.ans_process) - 1 and index_change == 1:
            message_box = QMessageBox(self.main_win)
            message_box.setWindowTitle("运行提示")
            message_box.setText('<h3>%s</h3>' % "运行到最后一步")
            message_box.setInformativeText("点击下一步，重新运行")
            message_box.setIcon(QMessageBox.Information)
            message_box.addButton(QPushButton('确定', message_box), QMessageBox.YesRole)
            message_box.show()
            self.step_index = -1
        elif self.step_index == -1 and index_change == -1:
            message_box = QMessageBox(self.main_win)
            message_box.setWindowTitle("运行提示")
            message_box.setText('<h3>%s</h3>' % "请点击下一步开始运行")
            message_box.setIcon(QMessageBox.Information)
            message_box.addButton(QPushButton('确定', message_box), QMessageBox.YesRole)
            message_box.show()
        else:
            self.step_index = self.step_index + index_change
        self.maze_panel.update(self.maze_map, self.ans_process, self.step_index)

    def maze_next(self):
        self.maze_update(index_change=1)
        self.maze_panel.show()

    def maze_last(self):
        self.maze_update(index_change=-1)
        self.maze_panel.show()


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    mywork = MyWork()
    mywork.main_win.show()

    sys.exit(app.exec_())
