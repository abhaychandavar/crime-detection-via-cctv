from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget


def initUi(self):
    centralwidget = QWidget()
    button = QPushButton('PyQt5 button', self)
    button.setToolTip('This is an example button')
    button.clicked.connect(self.on_click)
    button.setMaximumWidth(int(self.width / 2))
    layout = QVBoxLayout(centralwidget)
    layout.addWidget(button)
    layout.setAlignment(Qt.AlignCenter)
    self.setLayout(layout)