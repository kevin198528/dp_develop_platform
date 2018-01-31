import tensorflow as tf
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# class ImageGenerator(object):
#
#     def __init__(self):
#         self._dis_rotate = {'front': 8,
#                             'left': 1,
#                             'right': 1}
#         self._dis_property = {'brightness': 1,
#                               'contrast': 1,
#                               'saturation': 1,
#                               'hue': 1}
#         self._dis_scale = {'small': 18,
#                            'big': 30}

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr




img_file = '/home/kevin/face/wider_face/WIDER_train/images'
label_file = './wider_face_train.txt'

with open(label_file, 'r') as f:
    annotations = f.readlines()
#
# num = len(annotations)
# print("%d pics in total" % num)
#
# ret = IoU(np.array([0, 0, 24, 24]), np.array([[0, 0, 24, 24], [0, 0, 24, 24]]))
#
# print(ret)
#
annotation = annotations[9].strip().split(' ')
#
# print(annotation)
#
#
#
#

box_size = 48.0

join = lambda a: os.path.join(img_file, a + '.jpg')
#
pt = np.array(annotation[1:]).astype(np.float32).astype(np.int32).reshape([-1, 4])

face_width = pt[0][2] - pt[0][0]
face_high = pt[0][3] - pt[0][1]

scale = box_size / max(face_high, face_width)

print(scale)

# time.sleep(1000)

#
# print(pt)

# pt = np.array(annotation[5:9]).astype(np.float32).astype(np.int32)
#
# pt = np.array([448, 329, 448+24, 329+24])
#
pic = cv2.imread(join(annotation[0]))

pic_width = pic.shape[1]*scale
pic_high = pic.shape[0]*scale

pic = cv2.resize(pic, (int(pic_width), int(pic_high)), interpolation=cv2.INTER_AREA)



#
# num = 1000
#
start_pt = np.array([0, 0, box_size, box_size])

# ret_x = np.linspace(start=0, stop=pic_width - 24, num=pic_width - 23)
# ret_y = np.linspace(start=0, stop=pic_high - 24, num=pic_high - 23)

num = 100

ret_r = np.random.rand(num, 2)


ret_r[:, 0] = ret_r[:, 0]*pic_width - box_size
ret_r[:, 1] = ret_r[:, 1]*pic_high - box_size

# print(ret_y)

# iou_ret = IoU()


#

#
# center = np.array([pt[0] + int(pt_width/2)-12, pt[1] + int(pt_high/2)-12, pt[0] + int(pt_width/2) + 12, pt[1] + int(pt_high/2) + 12])
#
# num = 10
# # random_rect =
# ret_n = np.random.randn(num, 2)*(max(pt_width, pt_high)/2)
# # print(annotation)
#
# ret_n = np.hstack((ret_n, ret_n)).astype(np.int32)
#
# print(ret_n)
#
# ret_n = ret_n + center
#
# # for i in range(10):
# #     pt1[0] += ret_n[i][0]
# #     pt1[1] += ret_n[i][1]
# #     pt2[0] += ret_n[i][0]
# #     pt2[1] += ret_n[i][1]
#

# for i in range(num):
#     # pt1_tmp = tuple(ret_n[i][0:2])
#     # pt2_tmp = tuple(ret_n[i][2:4])
#     # print(pt1_tmp)
#
#     pt1_tmp = tuple(ret_r[i][0:].astype(np.int32))
#     pt2_tmp = tuple(ret_r[i].astype(np.int32) + 24)
#
#     cv2.rectangle(pic, pt1_tmp, pt2_tmp, (0, 255, 0), 1)

# # pic = cv2.imread(join(annotation[0]))
# #
# # for box in pt:
# #     print(box)
# #     cv2.rectangle(pic, tuple(box[0:2]), tuple(box[2:4]), (0, 255, 0), 1)
#
figure = cv2.namedWindow('win', flags=0)

cv2.imshow('win', pic)


cv2.waitKey(0)


