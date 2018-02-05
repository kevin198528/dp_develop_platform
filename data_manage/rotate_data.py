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

iou_field = np.linspace(0.1, 1.0, 10)

print(iou_field)

box_size = 24

img_file = '/home/kevin/face/wider_face/WIDER_train/images'
label_file = './wider_face_train.txt'
save_top_path = '/home/kevin/face/face_data'

total_idx = 0

total_idx += 1

join = lambda a: os.path.join(img_file, a + '.jpg')

join_save = lambda a: os.path.join(save_top_path, a+ '.jpg')

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

boxes = get_bounding_box(pic_width, pic_high, annotations, iou=0.0)

# shape is [scale, pt1.x, pt1.y, pt2.x, pt2.y]
# scale is 1

for iou in iou_field:
    box = get_bounding_box(pic_width, pic_high, annotations, iou=iou)
    boxes = np.vstack((boxes, box))

# time.sleep(1000)

# figure = cv2.namedWindow('win', flags=0)
#
# cv2.imshow('win', pic)
#
# cv2.waitKey(0)

# print(pt.shape)

# idx_random_face = np.random.randint(0, pt.shape[0], 1)[0]

# print(idx_random_face)

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

# face_width = pt[0][2] - pt[0][0]
# face_high = pt[0][3] - pt[0][1]

# scale = box_size / max(face_high, face_width)

# print(scale)

# time.sleep(1000)

#
# print(pt)

# pt = np.array(annotation[5:9]).astype(np.float32).astype(np.int32)
#
# pt = np.array([448, 329, 448+24, 329+24])
#

# pic_width = pic.shape[1]*scale
# pic_high = pic.shape[0]*scale

local_index = 0

for box in boxes:
    local_index += 1

    scale = box[0]

    pic_resized = cv2.resize(pic, (int(pic_width*scale), int(pic_high*scale)), interpolation=cv2.INTER_AREA)

    x_s = int(box[2])
    x_e = int(box[4])
    y_s = int(box[3])
    y_e = int(box[5])

    crop = pic_resized[y_s:y_e, x_s:x_e]

    save_path = join_save(str(total_idx) + '_' + str(local_index) + '_' + str(box[0]) + '_' + str(box[1]))

    print(save_path)

    cv2.imwrite(save_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # figure = cv2.namedWindow('win', flags=0)
    #
    # cv2.imshow('win', crop)
    #
    # cv2.waitKey(0)

figure = cv2.namedWindow('win', flags=0)

cv2.imshow('win', pic)

cv2.waitKey(0)

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
#
# iou_ret = IoU()
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


