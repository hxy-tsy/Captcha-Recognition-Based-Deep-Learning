import re
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pymysql
from utils import Utils


class Ui_Change(object):
    util = Utils()

    def setupUi(self, Change):
        Change.setObjectName("Change")
        Change.resize(677, 508)
        self.label_2 = QtWidgets.QLabel(Change)
        self.label_2.setGeometry(QtCore.QRect(180, 190, 121, 51))
        self.label_2.setStyleSheet("font: 16pt \"Adobe Heiti Std\";")
        self.label_2.setObjectName("label_2")
        self.lineEdit = QtWidgets.QLineEdit(Change)
        self.lineEdit.setGeometry(QtCore.QRect(280, 200, 221, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.label_3 = QtWidgets.QLabel(Change)
        self.label_3.setGeometry(QtCore.QRect(180, 240, 121, 51))
        self.label_3.setStyleSheet("font: 16pt \"Adobe Heiti Std\";")
        self.label_3.setObjectName("label_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(Change)
        self.lineEdit_2.setGeometry(QtCore.QRect(280, 250, 221, 31))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.setEchoMode(QLineEdit.Password)
        self.pushButton = QtWidgets.QPushButton(Change)
        self.pushButton.setGeometry(QtCore.QRect(290, 320, 75, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.change)

        self.pushButton_2 = QtWidgets.QPushButton(Change)
        self.pushButton_2.setGeometry(QtCore.QRect(410, 320, 75, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.exit_change)  # 添加退出按钮的连接

        self.retranslateUi(Change)
        QtCore.QMetaObject.connectSlotsByName(Change)

    def retranslateUi(self, Change):
        _translate = QtCore.QCoreApplication.translate
        Change.setWindowTitle(_translate("Change", "修改密码"))
        self.label_2.setText(_translate("Change", "账号"))
        self.label_3.setText(_translate("Change", "密码"))
        self.pushButton.setText(_translate("Change", "修改"))
        self.pushButton_2.setText(_translate("Change", "退出"))

    def change(self):
        username = self.lineEdit.text()
        password = self.lineEdit_2.text()
        if self.check_password_strength(password) == False:
            msg = QMessageBox(QMessageBox.Warning, "警告",
                              "密码过于简单，密码不能少于8位且必须包含字母、数字、特殊字符三种类型")
            msg.exec_()
        else:
            pwd = self.util.encrypt_data(password)
            print(pwd)
            db = pymysql.connect(host='localhost', user='root', password='root', database='mysql')
            try:
                cursor = db.cursor()
                sql = "UPDATE captcha_user SET password=%s WHERE username=%s;"
                print("密码修改成功")
                cursor.execute(sql, (pwd, username))
                db.commit()
                msg_box = QMessageBox(QMessageBox.Information, '成功', '密码修改成功')
                msg_box.exec_()
                # self.close()  # 关闭当前窗口
            except Exception as e:
                db.rollback()  # 回滚
                msg_box = QMessageBox(QMessageBox.Critical, '错误', '密码修改失败')
                msg_box.exec_()
                print("密码修改失败", e)
            finally:
                db.close()  # 确保数据库连接只关闭一次

    def check_password_strength(self, password):
        has_letter = re.compile(r'[a-zA-Z]')
        has_digit = re.compile(r'\d')
        has_special_char = re.compile(r'[!@#$%^&*()]')

        if has_letter.search(password) and has_digit.search(password) and has_special_char.search(password) and len(password) >= 8:
            return True
        else:
            return False

    def exit_change(self):
        self.close()  # 关闭当前窗口
