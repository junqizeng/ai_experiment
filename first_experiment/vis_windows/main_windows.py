from PyQt5.QtWidgets import QApplication, QWidget, QPushButton


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(850, 800)
        self.setWindowTitle('路径规划问题算法演示')
        self.move(int((QApplication.desktop().width()-self.width())*0.5),
                  int((QApplication.desktop().height()-self.height())*0.5))

        self.DFS_btn = QPushButton(self)
        self.DFS_btn.resize(180, 50)
        self.DFS_btn.move(15, 55)
        self.DFS_btn.setStyleSheet('background-color:white;font-size:16px;')
        self.DFS_btn.setText('DFS 算法')

        self.BFS_btn = QPushButton(self)
        self.BFS_btn.resize(180, 50)
        self.BFS_btn.move(15, 115)
        self.BFS_btn.setStyleSheet('background-color:white;font-size:16px')
        self.BFS_btn.setText('BFS 算法')

        self.Astar_btn = QPushButton(self)
        self.Astar_btn.resize(180, 50)
        self.Astar_btn.move(15, 175)
        self.Astar_btn.setStyleSheet('background-color:white;font-size:16px')
        self.Astar_btn.setText('A* 算法')

