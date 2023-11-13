from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt


class WelPanel:
    def __init__(self, main_window):
        self.wel_slogan = QLabel(main_window)
        self.wel_slogan.resize(500, 35)
        self.wel_slogan.setStyleSheet('font-size:16px;font-weight:bold;font-family:SimHei;')
        self.wel_slogan.move(20, 10)

        self.account_slogan = QLabel(main_window)
        self.account_slogan.resize(620, 300)
        self.account_slogan.setAlignment(Qt.AlignCenter)
        self.account_slogan.setStyleSheet('font-size:18px;font-weight:bold;font-family:SimHei;')
        self.account_slogan.move(200, 55)

        self.wel_slogan.setText('欢迎使用！')
        self.account_slogan.setText('作者：曾俊淇\n'
                                    '学号：2023202210155\n\n'
                                    '路径规划问题（迷宫寻路）算法演示\n\n'
                                    'DFS：点击“DFS”进入深度优先算法演示\n'
                                    'BFS：点击“BFS”进入宽度优先算法演示\n'
                                    'A* ：点记“A*”进入启发式A*算法演示')

    def hide(self):
        self.wel_slogan.hide()
        self.account_slogan.hide()

    def show(self):
        self.wel_slogan.show()
        self.account_slogan.show()
