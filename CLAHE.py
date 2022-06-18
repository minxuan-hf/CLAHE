"""
利用OpenCV算法库实现CLAHE(限制对比度自适应直方图均衡化)算法 
"""
import os

import cv2 as cv
import numpy as np


# 对整个存放图片文件夹进行CLAHE算法操作
def all_img_clahe(file_path, new_file_path):
    """
    file_path: 原始图片文件夹存放路径
    new_file_path: 新图片文件夹存放路径
    """
    file_list = os.listdir(file_path)
    for file in file_list:
        src = os.path.join(os.path.abspath(file_path), file)
        file_part = file.split('.')[0] + '_clahe.png'
        # 读取图片
        test = cv.imread(src, -1)
        # 将图片高和宽分别赋值给x，y
        x, y = test.shape[0:2]
        # 拆分每个通道
        B, G, R = cv.split(test)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_B = clahe.apply(B)
        clahe_G = clahe.apply(G)
        clahe_R = clahe.apply(R)
        clahe_test = cv.merge((clahe_B, clahe_G, clahe_R))
        # cv.imshow('clahe_test', clahe_test)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        cv.imwrite(os.path.join(new_file_path, file_part), clahe_test)


# 对单一图片进行CLAHE算法操作
def single_img_clahe(file_path, new_file_path):
    """
    file_path: 原始图片存放路径
    new_file_path: 新图片存放路径
    """
    # 读取图片
    test = cv.imread(file_path, -1)
    # 将图片高和宽分别赋值给x，y
    x, y = test.shape[0:2]
    # 拆分每个通道
    B, G, R = cv.split(test)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_B = clahe.apply(B)
    clahe_G = clahe.apply(G)
    clahe_R = clahe.apply(R)
    clahe_test = cv.merge((clahe_B, clahe_G, clahe_R))
    # cv.imshow('clahe_test', clahe_test)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # 保存图片
    cv.imwrite(new_file_path, clahe_test)


if __name__ == "__main__":
    all_img_clahe(file_path=r"E:\Disney",
                  new_file_path=r"E:\Disney_clahe")
    # single_img_clahe(file_path=r'E:\micky.png',
    #                  new_file_path=r"E:\micky_clahe.png")
