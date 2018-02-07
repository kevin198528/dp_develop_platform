import tensorflow as tf
import numpy as np
import os
import cv2

# pic = cv2.imread('./f_01.jpeg')
#
# coutn = 368*368*1
#
# s = 0.8
#
# for k in range(0, coutn):
#     # get the random point
#     xi = int(np.random.uniform(0, pic.shape[1]))
#     xj = int(np.random.uniform(0, pic.shape[0]))
#     # xi = np.linspace(0, pic.shape[1] - 1, 368).astype(np.int32)
#     # xj = np.linspace(0, pic.shape[0] - 1, 368).astype(np.int32)
#     # add noise
#     if pic.ndim == 2:
#         pic[xj, xi] = 255
#     elif pic.ndim == 3:
#         pic[xj, xi, 0] = 25
#         pic[xj, xi, 1] = 20
#         pic[xj, xi, 2] = 20
#
# figure = cv2.namedWindow('win', flags=0)
#
# cv2.imshow('win', pic)
#
# cv2.waitKey(0)

class MyType(type):

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        print('在这里面..')
        print('==========================')
        print('来咬我呀')
        obj.__init__(*args, **kwargs)
        return obj


class Foo(metaclass=MyType):

    def __init__(self):
        self.name = 'alex'


f = Foo()
# print(f.name)
