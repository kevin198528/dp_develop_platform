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

def img_random_operate(pic, idx):
    op = np.random.randint(0, 4, 1)[0]

    op0 = tf.image.random_brightness(pic, 0.6, seed=idx)
    op1 = tf.image.random_contrast(pic, lower=0.2, upper=2.0, seed=idx)
    op2 = tf.image.random_saturation(image=pic, lower=0.1, upper=1.5, seed=idx)
    op3 = tf.image.random_hue(image=pic, max_delta=0.2, seed=idx)

    with tf.Session() as sess:

        if (op == 0):
            ret = sess.run(op0)
            print('brightness')
        elif (op == 1):
            ret = sess.run(op1)
            print('contrast')
        elif (op == 2):
            ret = sess.run(op2)
            print('saturation')
        elif (op == 3):
            ret = sess.run(op3)
            print('hue')
    return ret

# zero bounding box use global random select
def get_zero_bounding_box(width, high, annotations):
    pass

# normal bounding box use face random.randn select
def get_normal_bounding_box(width, high, annotations, iou):
    pass

def get_bounding_box(width, high, annotations, iou):
    if iou == 0.0:
        return get_zero_bounding_box(width, high, annotations)
    else:
        return get_normal_bounding_box(width, high, annotations, iou)

iou_field = np.linspace(0, 1.0, 11)

print(iou_field)

box_size = 24

img_file = '/home/kevin/face/wider_face/WIDER_train/images'
label_file = './wider_face_train.txt'

join = lambda a: os.path.join(img_file, a + '.jpg')

with open(label_file, 'r') as f:
    annotations = f.readlines()

# select random img
idx_random_img = np.random.randint(0, len(annotations), 1)[0]

annotation = annotations[idx_random_img].strip().split(' ')

# read img
pic = cv2.imread(join(annotation[0]))

# random img convert
pic = img_random_operate(pic, idx_random_img)

pic_width = pic.shape[1]
pic_high = pic.shape[0]

annotations = np.array(annotation[1:]).astype(np.float32).astype(np.int32).reshape([-1, 4])

zero_box = get_bounding_box(pic_width, pic_high, annotations, iou=0.0)



# figure = cv2.namedWindow('win', flags=0)
#
# cv2.imshow('win', pic)
#
# cv2.waitKey(0)

print(pt.shape)

idx_random_face = np.random.randint(0, pt.shape[0], 1)[0]

print(idx_random_face)

# time.sleep(1000)

box_size = 48.0

# print(annotations[0:10])

#
# num = len(annotations)
# print("%d pics in total" % num)
#
# ret = IoU(np.array([0, 0, 24, 24]), np.array([[0, 0, 24, 24], [0, 0, 24, 24]]))
#
# print(ret)
#

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


