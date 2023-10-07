import sys
import subprocess
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QHBoxLayout, QVBoxLayout, QMessageBox

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 创建UI组件
        self.setWindowTitle("调用Python脚本")
        self.setGeometry(100, 100, 500, 300)

        self.file_label = QLabel("选择脚本文件:")
        self.file_button = QPushButton("浏览")
        self.file_button.clicked.connect(self.selectFile)

        self.video_label = QLabel("选择视频文件:")
        self.video_button = QPushButton("浏览")
        self.video_button.clicked.connect(self.selectVideo)

        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.startProcess)

        # 布局UI组件
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.file_label)
        hbox1.addWidget(self.file_button)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.video_label)
        hbox2.addWidget(self.video_button)

        hbox3 = QHBoxLayout()
        hbox3.addStretch(1)
        hbox3.addWidget(self.start_button)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        self.setLayout(vbox)

        self.show()

    def selectFile(self):
        # 弹出文件选择对话框，获取脚本文件路径
        file_path, _ = QFileDialog.getOpenFileName(self, "选择脚本文件")
        if file_path:
            self.file_label.setText(f"已选择脚本文件: {file_path}")
            self.script_file = file_path

    def selectVideo(self):
        # 弹出文件选择对话框，获取视频文件路径
        video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件")
        if video_path:
            self.video_label.setText(f"已选择视频文件: {video_path}")
            self.video_file = video_path

    def startProcess(self):
        # 执行脚本文件，并传递参数
        try:
            cmd = [
                "python",
                self.script_file,
                "--video-path",
                self.video_file,
                "--out-video-root",
                "vis_results_dyhead"
            ]
            subprocess.run(cmd, check=True)
            QMessageBox.information(self, "提示", "处理完成")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())
