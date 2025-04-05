import sys
import pymysql 
import traceback
from Ui_login import Ui_Login
from Ui_predict import Ui_Predict
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import *  
from PyQt5.QtCore import *
from Ui_change import  Ui_Change
from PyQt5.QtWidgets import QApplication, QMainWindow
from Ui_index import  Ui_MainWindow
from Ui_list import Ui_List
from Ui_intorduce import Ui_introduce
import re
from utils import Utils

class Login(QMainWindow, Ui_Login):
    util = Utils()

    def __init__(self):
        super(Login, self).__init__()
        self.setupUi(self)
        self.mainwindow1 = None
        self.mainwindow2 = None
        self.mainwindow3 = None
        self.mainwindow4 = None
        self.mainwindow5 = None

        self.pushButton.clicked.connect(self.login)
        self.pushButton_2.clicked.connect(self.register)

    def login(self):
        username = self.lineEdit.text()
        password = self.lineEdit_2.text()
        try:
            db = pymysql.connect(host='localhost', user='root', password='root', database='captcha')
            cursor = db.cursor()
            sql = "SELECT password FROM captcha_user WHERE username=%s;"
            cursor.execute(sql, (username,))
            pas = cursor.fetchone()

            # 检查查询结果是否为 None
            if pas is None:
                msg_box = QMessageBox(QMessageBox.Critical, '错误', '用户名不存在')
                msg_box.exec_()
                return

            # 检查账户名是否为数字
            if not self.check_password_strength(password, username):
                msg_box = QMessageBox(QMessageBox.Critical, '错误', '账号只能为数字')
                msg_box.exec_()
                return

            decryt_pas = self.util.decrypt_data(pas[0])  # 如果 pas 不为 None，可以继续使用
            print(pas[0])

            if decryt_pas == password:
                print("密码正确")
                self.mainwindow2 = Index()
                self.mainwindow1 = Predict(username)
                self.mainwindow2.show()
                self.close()
                self.mainwindow2.pushButton.clicked.connect(self.jump1)
                self.mainwindow1.pushButton_4.clicked.connect(self.jump2)

                self.mainwindow2.manage_action.triggered.connect(self.jump3)
                self.mainwindow2.exit_acction.triggered.connect(self.jump4)  # 退出系统
                self.mainwindow2.about_menu.addAction(self.mainwindow2.manage_action)
                self.mainwindow2.about_menu.addAction(self.mainwindow2.exit_acction)

                self.mainwindow1.introduction_action.triggered.connect(self.jump8)
                self.mainwindow1.predict_menu.addAction(self.mainwindow1.introduction_action)


                self.mainwindow3 = Change()
                self.mainwindow3.pushButton_2.clicked.connect(self.jump5)

                self.mainwindow4 = List()
                self.mainwindow1.predict_action.triggered.connect(lambda :self.jump6(username))  # 退出系统
                self.mainwindow1.predict_menu.addAction(self.mainwindow1.predict_action)
                self.mainwindow4.pushButton.clicked.connect(self.jump7)

                self.mainwindow5 = Introduce()
                self.mainwindow5.pushButton.clicked.connect(self.jump9)

            else:
                msg_box = QMessageBox(QMessageBox.Critical, '错误', '用户名或密码错误')
                msg_box.exec_()

        except Exception as e:
            db.rollback()
            traceback.print_exc()
            msg_box = QMessageBox(QMessageBox.Critical, '错误', '用户名或密码错误')
            msg_box.exec_()

    def register(self):
        username = self.lineEdit.text()
        password = self.lineEdit_2.text()
        # 修复参数顺序
        if self.check_password_strength(password, username) == False:
            msg = QMessageBox(QMessageBox.Warning, "警告",
                              "密码过于简单，密码不能少于8位且必须包含字母、数字、特殊字符三种类型且账号只能为数字")
            msg.exec_()
        else:
            pwd = self.util.encrypt_data(password)
            print(pwd)
            db = pymysql.connect(host='localhost', user='root', password='root', database='captcha')
            try:
                cursor = db.cursor()
                sql = "insert into captcha_user(username,password)values(%s,%s);"
                print("注册成功")
                cursor.execute(sql, (username, pwd))
                db.commit()
                msg_box = QMessageBox(QMessageBox.Information, '成功', '注册成功')
                msg_box.exec_()
            except Exception as e:
                msg_box = QMessageBox(QMessageBox.Critical, '错误', e)
                msg_box.exec_()
                db.rollback()  # 回滚
                print("注册失败", e)
            db.close()

    def check_password_strength(self, password, account):
        # 定义正则表达式模式
        has_letter = re.compile(r'[a-zA-Z]')
        has_digit = re.compile(r'\d')
        has_special_char = re.compile(r'[!@#$%^&*()]')

        # 检查账户是否仅包含数字
        # if not account.isdigit():
        #     return False

        # 检查密码是否满足每个条件
        if has_letter.search(password) and has_digit.search(password) and has_special_char.search(password) and len(
                password) >= 8:
            return True
        else:
            return False

    def jump1(self):
        self.mainwindow2.close()
        self.mainwindow1.show()

    def jump2(self):
        self.mainwindow1.close()
        self.mainwindow2.show()

    def jump3(self):
        self.mainwindow2.close()
        self.mainwindow3.show()
        self.lineEdit.setText("")
        self.lineEdit_2.setText("")

    def jump4(self):
        self.mainwindow2.close()
        self.show()
        self.lineEdit.setText("")
        self.lineEdit_2.setText("")

    def jump5(self):
        self.mainwindow3.close()
        self.show()
        self.lineEdit.setText("")
        self.lineEdit_2.setText("")

    def jump6(self,username):
        self.mainwindow1.close()
        self.mainwindow4.load(username)
        self.mainwindow4.show()

    def jump7(self):
        self.mainwindow4.close()
        self.mainwindow1.show()

    def jump8(self):
        # self.mainwindow1.close()
        # self.mainwindow1.show()
        self.mainwindow1.close()
        self.mainwindow5.show()

    def jump9(self):
        self.mainwindow5.close()
        self.mainwindow1.show()

class Predict(QMainWindow, Ui_Predict):
    def __init__(self,username):
        super(Predict, self).__init__()
        self.setupUi(self)
        self.soltInit(username)

    def soltInit(self,username):
        self.pushButton_3.clicked.connect(lambda: self.save(username))

class Index(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(Index,self).__init__()
        self.setupUi(self)

class Change(QMainWindow,Ui_Change):
    def __init__(self):
        super(Change,self).__init__()
        self.setupUi(self)




class List(QMainWindow,Ui_List):
    def __init__(self):
        super(List,self).__init__()
        self.setupUi(self)
        # self.load(username)

class Introduce(QMainWindow,Ui_introduce):
    def __init__(self):
        super(Introduce,self).__init__()
        self.setupUi(self)

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    login = Login()
    login.show()
    sys.exit(app.exec_())