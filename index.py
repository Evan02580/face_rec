from PyQt5 import QtCore

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon, QTextCursor
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QTableWidgetItem, QAbstractItemView
from PyQt5.uic import loadUi
import dataManage
import core
import dataRecord

import logging.config
import sys
import multiprocessing



class IndexUI(QWidget):
    logQueue = multiprocessing.Queue()  # 日志队列
    receiveLogSignal = pyqtSignal(str)  # 日志信号

    def __init__(self):
        super(IndexUI, self).__init__()
        loadUi('./ui/index.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.core.clicked.connect(self.turnto_core)
        self.record.clicked.connect(self.turnto_record)
        self.manage.clicked.connect(self.turnto_manage)
        self.quit.clicked.connect(self.clickButtonCloseWindow)

    def clickButtonCloseWindow(self):
        qApp = QApplication.instance()
        qApp.quit()

    def turnto_core(self, cameral_state):
        window = core.CoreUI()
        window.show()

    def turnto_record(self, cameral_state):
        window = dataRecord.DataRecordUI()
        window.show()

    def turnto_manage(self):
        window = dataManage.DataManageUI()
        window.show()

if __name__ == '__main__':
    logging.config.fileConfig('./config/logging.cfg')
    #对高分辨率屏幕的支持
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = IndexUI()
    window.show()
    sys.exit(app.exec())
