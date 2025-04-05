import os
import random
import re

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import pymysql
from Crypto import Random
from skimage import morphology
import time
import torch
import cv2 as cv
from PyQt5.QtWidgets import QMessageBox
from torchvision import transforms
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
import string
import torch.nn as nn
import PIL.Image as Image

from CNN import CNN
from CNN_GRU import CNN_GRU
from torchvision.transforms import Compose, ToTensor
import torch
from ultralytics import YOLO
from pathlib import Path

from GESTURE.Gesture import UNet


class Utils:
    characters = '-' + string.digits + string.ascii_uppercase
    width, height, n_len, n_classes = 192, 64, 4, len(characters)  # 生成图片宽高定、定长验证码、字符数
    n_input_length = 12
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = [str(i) for i in range(0, 10)]
    source += [chr(i) for i in range(97, 97 + 26)]
    source += [chr(i) for i in range(65, 65 + 26)]
    alphabet = ''.join(source)

    def login(self, username):
        print("login")
        return True

    def encrypt_data(self, msg):
        public_key_file = 'D:\\code\\python\\Outsourcing\\captcha\\rsa_public_key.pem'
        public_key = self.get_key(public_key_file, is_private=False)
        cipher = PKCS1_OAEP.new(public_key)
        encrypt_text = base64.b64encode(cipher.encrypt(msg.encode())).decode()
        return encrypt_text

    def decrypt_data(self, encrypt_msg):
        private_key_file = 'D:\\code\\python\\Outsourcing\\captcha\\rsa_private_key.pem'
        private_key = self.get_key(private_key_file, is_private=True)
        cipher = PKCS1_OAEP.new(private_key)
        back_text = cipher.decrypt(base64.b64decode(encrypt_msg)).decode()
        return back_text

    def get_key(self, key_file, is_private=False):
        with open(key_file, 'rb') as f:
            data = f.read()
            if is_private:
                key = RSA.import_key(data)
            else:
                key = RSA.importKey(data)
        return key

    # def CNN_LSTM(self, img_path):
    #     # 加载模型并确保在 CPU 上加载
    #     print('sb')
    #     train_weights='D:\\code\\python\\Outsourcing\\captcha\\ctc.pth'
    #     train_weights_dict = torch.load(train_weights, map_location=torch.device('cpu'))
    #     model = CNN_LSTM(37)
    #     model.load_state_dict(train_weights_dict)
    #     print('ss')
    #     # Move model to the selected device (GPU or CPU)
    #     model = model.to('cpu')  # If CUDA is available, model will be moved to GPU, else to CPU
    #     model.eval()  # Set model to evaluation mode
    #     print("开始预测")
    #     # Load and preprocess the image
    #     image = cv.imread(img_path)  # Load the image
    #     image = cv.resize(image, (self.width, self.height))
    #
    #     image = image[..., ::-1]  # BGR to RGB
    #     image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    #
    #     transform = transforms.Compose([transforms.ToTensor()])
    #     image = transform(image).unsqueeze(0).to(self.device)  # Add batch dimension
    #
    #     # Perform inference
    #     output = model(image)
    #     output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    #     print(output_argmax)
    #     # Decode and print the prediction
    #     pred = self.decode(output_argmax[0])
    #     print(f'Pred: {pred}')
    #     return pred

    def decode(self,output):
        """
        处理 CTC 输出，去除重复字符和空白符（假设空白符是第一个字符）
        """
        output_argmax = output.argmax(dim=-1)  # [seq_len, batch_size, n_classes] → [seq_len, batch_size]
        decoded = []
        for batch in output_argmax.transpose(0, 1):  # 遍历批次
            batch_decoded = []
            prev_char = -1
            for idx in batch:
                if idx != prev_char and idx != 0:  # 跳过空白符（假设空白符索引为0）
                    batch_decoded.append(self.characters[idx])
                prev_char = idx
            decoded.append(''.join(batch_decoded))
        return decoded

    def decode_target(self, sequence):
        return ''.join([self.characters[x] for x in sequence]).replace(' ', '')

    def CNN_GRU(self, image_path):
        train_weights = "D:\\code\\python\\Outsourcing\\captcha\\vary_base_cov_mnist.pt"
        train_weights_dict = torch.load(train_weights,map_location=torch.device('cpu'))
        model = CNN_GRU(11)
        model.load_state_dict(train_weights_dict, strict=True)
        model.to(self.device)
        model.eval()

        # Preprocess the image
        image = Image.open(image_path)
        pre_process = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.5], std=[0.5])]
        )
        image = pre_process(image)
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            predicts = model(image)[0]
            predicts = predicts.cpu()
            output = torch.argmax(predicts, dim=-1)
            output = output.permute(1, 0)
            predict = torch.unique_consecutive(output[0])
            predict = predict[predict != (10)]
        pred = map(str, predict.numpy().tolist())
        # Print the prediction
        pred=''.join(pred)
        print('Prediction:',pred )
        return pred

    def YOLO_CLICK(self, image_path):
        model_path = 'D:\\code\\python\\Outsourcing\\captcha\\click_captcha.pt'
        model = YOLO(model_path)

        # 检查图片路径是否存在
        if not os.path.exists(image_path):
            print(f"错误: 图片路径 {image_path} 无效!")
            return "", None

        # 进行预测并保存结果
        results = model.predict(source=image_path, save=True, save_txt=True)

        box = ""
        # 打印检测到目标的坐标
        for *xyxy, conf, cls in results[0].boxes.data:
            print(f'类别：{results[0].names[int(cls)]}，置信度：{conf:.2f}，坐标：{xyxy}')
            box += str(xyxy) + "\n"

        # 获取保存的图像路径
        save_paths = results[0].save("D:\\code\\python\\Outsourcing\\captcha\\CLICK_CAPTCHA\\result.png")
        print(save_paths)

        # 返回预测框和保存后的图像路径
        return box, save_paths

    def YOLO_DRAG(self,image_path):
        model_path = 'D:\\code\\python\\Outsourcing\\captcha\\best.pt'
        model = YOLO(model_path)

        # 检查图片路径是否存在
        if not os.path.exists(image_path):
            print(f"错误: 图片路径 {image_path} 无效!")
            return "", None

        # 进行预测并保存结果
        results = model.predict(source=image_path, save=True, save_txt=True)

        box = ""
        # 打印检测到目标的坐标
        for *xyxy, conf, cls in results[0].boxes.data:
            print(f'类别：{results[0].names[int(cls)]}，置信度：{conf:.2f}，坐标：{xyxy}')
            box += str(xyxy) + "\n"

        # 获取保存的图像路径
        save_paths = results[0].save("D:\\code\\python\\Outsourcing\\captcha\\QK_CAPTCHA\\result.png")  # 调用 save() 方法获取保存路径
        print(save_paths)

        # 返回预测框和保存后的图像路径
        return box, save_paths

    def CNN(self,image_path):
        model_path='D:\\code\python\\Outsourcing\\captcha\\cnn.pth'
        image=Image.open(image_path)
        trans = ToTensor()
        img = trans(image)
        cnn=CNN()
        cnn.eval()
        model=torch.load(model_path,map_location='cpu')
        cnn.load_state_dict(model)
        img_tensor = img.view(1, 3, 40, 120)
        output = cnn(img_tensor)
        output = output.view(-1, 62)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        output = output.view(-1, 4)[0]
        label = ''.join([self.alphabet[i] for i in output.cpu().numpy()])
        print(label)

        return label


    ##保存预测结果
    def save(self, username, captcha_type, model, pred):
        try:
            # 连接到数据库
            with pymysql.connect(host='localhost', user='root', password='root', database='captcha') as db:
                # 创建一个新的数据库游标对象 cursor
                with db.cursor() as cursor:
                    # 获取当前时间戳并格式化

                    # 使用参数化查询，避免SQL注入，并确保数据被正确转义
                    sql = """
                        INSERT INTO captcha_prediction (pred, username, model, captcha_type)
                        VALUES (%s, %s, %s, %s)
                    """

                    # 执行插入操作
                    cursor.execute(sql, (pred,username, model, captcha_type))

                    # 提交事务
                    db.commit()

                    print("保存成功")
                    msg = QMessageBox(QMessageBox.Information, '成功', "保存成功")
                    msg.exec_()

        except pymysql.MySQLError as e:
            # 如果数据库操作失败，进行事务回滚
            if db:
                db.rollback()

            print("数据库错误:", e)
            msg_box = QMessageBox(QMessageBox.Critical, '错误', '保存失败: 数据库错误')
            msg_box.exec_()

        except Exception as e:
            # 捕获其他异常
            print("发生未知错误:", e)
            msg_box = QMessageBox(QMessageBox.Critical, '错误', '保存失败: 未知错误')
            msg_box.exec_()

    def getInfo(self,path):
        df2 = cv.imread(path)
        print("action")
        model=UNet(3,2)
        madel_weight=torch.load("D:\\code\\python\\Outsourcing\\captcha\\GESTURE\\unet.pth", map_location=torch.device('cpu'))
        # model = torch.load("D:\\code\\python\\Outsourcing\\captcha\\GESTURE\\fumx.pth", map_location=torch.device('cpu'))
        model.load_state_dict(madel_weight)
        print("over")
        image = cv.resize(df2, (280, 176))  # Resize the image to 280x176

        # Apply the ToTensor transformation
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image)  # Shape will be [C, H, W] i.e. [3, 176, 280]

        # Move the tensor to the correct device (GPU/CPU)
        image_tensor = image_tensor.unsqueeze(0).to('cpu')  # Add batch dimension: [1, 3, 176, 280]
        print("kasi")
        # Forward pass through the model
        out = model(image_tensor)
        print("没到")
        out = torch.argmax(out, dim=1)  # Assuming output is a segmentation map

        # Create an empty mask and set the predicted region to 255
        j = torch.zeros((1, 176, 280))
        j[out == 1] = 255  # Assuming the class of interest is '1'

        # Repeat the mask to create a 3-channel image for visualization
        kp = j.repeat(3, 1, 1)
        kp = torch.permute(kp, (1, 2, 0))  # Change dimensions to [H, W, C] for visualization

        # Skeletonize the mask
        gj = morphology.skeletonize(np.array(kp.tolist(), dtype=np.uint8))

        # Extract the skeletonized points
        hn = np.transpose(np.nonzero(gj))
        print(hn)

        # Visualize the result
        es = np.array(kp.tolist(), dtype=np.uint8)
        for i in hn:
            es[i[0], i[1], :] = 100  # Modify the skeleton points to highlight them

        cv.imwrite("D:\\code\\python\\Outsourcing\\captcha\\GESTURE\\result.png", es)

    def find_all(self, username):
        db = pymysql.connect(host='localhost', user='root', password='root', database='captcha')
        try:
            # 创建一个新的数据库游标对象 cursor
            cursor = db.cursor()
            sql_1 = "SELECT * FROM captcha_prediction where username=%s"
            cursor.execute(sql_1, (username))
            res = cursor.fetchall()
            return res
            # print(res)
            # for row, row_data in enumerate(res):
            #     self.tableWidget.setRowCount(row + 1)
            #     # 创建按钮
            #     button = QtWidgets.QPushButton("删除", self)
            #     button.setFixedSize(50, 25)  # 设置按钮的大小
            #     # 将按钮设置为单元格小部件
            #     self.tableWidget.setCellWidget(row, 5, button)
            #     button.clicked.connect(lambda: self.on_button_clicked(row, account))
            #     for col, data in enumerate(row_data):
            #         # item = QTableWidgetItem(str(data))
            #         self.tableWidget.setItem(row, col, QTableWidgetItem(str(data)))
            print("查询成功")
        except Exception as e:
            db.rollback()
            print("发生错误", e)

    def extract_numbers_from_tensor_string(self,tensor_string):
        # 使用正则表达式匹配 tensor 中的数值
        numbers = re.findall(r'tensor\((.*?)\)', tensor_string)
        # 将匹配到的字符串转换为浮点数
        float_numbers = [float(num) for num in numbers]
        return float_numbers

if __name__ == "__main__":
    util = Utils()
    img_path = 'D:\\code\\python\\Outsourcing\\captcha\\GESTURE\\119.jpg'
    # util.CNN(img_path)
    # util.CNN_GRU(img_path)
    # util.YOLO_CLICK(img_path)
    # util.save(111111,'test','test','test')
    # util.getInfo(img_path)
    tensor_string = "[tensor(209.0465), tensor(148.8065), tensor(295.3108), tensor(225.8940)]"

    # 调用函数并打印结果
    result = util.extract_numbers_from_tensor_string(tensor_string)
    print(result)  #
