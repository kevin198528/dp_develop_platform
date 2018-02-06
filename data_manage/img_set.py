import numpy as np
import tensorflow as tf
import cv2
import os
import copy
import sys
import time

class ImgBase(object):

    def __init__(self, img, annotation):
        """
        raw img annotation and std size
        each picture process one face annotation

        img: (high, width, channel)
        (0,0)   +x
            ------->
            |
            |
          + |
          y v
        annotation: numpy array type [x1, y1, x2, y2]
        std_window_size: 24.0

        """
        self._img = img
        self._resize_img = self._img
        self._bounding_box = np.array([0, 0, 0, 0])
        self._annotation = annotation

        """
        _zoom_ratio: -0.2~0.2 describe the picture zoom degree
         
        """
        self._zoom_ratio = 0.0

        """
        get img width hight and face width high
        
        """
        self._img_high = self._img.shape[0]
        self._img_width = self._img.shape[1]
        self._resize_img_high = self._img_high
        self._resize_img_width = self._img_width
        self._face_width = self._annotation[2] - self._annotation[0]
        self._face_high = self._annotation[3] - self._annotation[1]
        self._resize_face_width = self._face_width
        self._resize_face_high = self._face_high

    def get_zoom_ratio(self):
        return self._zoom_ratio

    def get_iou_ratio(self):
        """
        iou_current: the real iou between bounding_box and annotation
        iou_max: the max iou between bounding_box and annotation
        iou_ration: 0~1 describe the actual position accuracy

        """
        iou_current = self.iou(self._annotation, self._bounding_box.reshape([-1, 4]))
        box_tmp = self._annotation.copy()
        box_tmp[2] = box_tmp[0] + 24
        box_tmp[3] = box_tmp[1] + 24
        iou_max = self.iou(self._annotation, box_tmp.reshape([-1, 4]))
        return iou_current / iou_max

    def get_face_probability(self):
        """
        use iou_ratio and zoom_ratio to measure the face probability in 24x24 window
        the value is 0~1 1: the iou_ratio = 1.0 and zoom_ratio = 0.0

        """
        face_prob = self.get_iou_ratio() - np.abs(self._zoom_ratio)
        if face_prob < 0:
            face_prob = 0
        return face_prob

    def get_position_ratio(self):
        """
        get regression bounding box ratio on 24x24 picture

        """
        return np.array([(x - y)/24 for x, y in zip(self._annotation, self._bounding_box)])

    def check_face_size(self, min_size=16):
        """
        check the face in picture, the support min size is 16

        """
        if min(self._face_width, self._face_high) < min_size:
            return False
        return True

    def zoom_change(self, std_size=24.0, random_scale=False):
        """
        compute scale use std_size and face size
        ex. std_size = 24 face_size = 240 scale = 24 / 240 = 0.1
        then change img size and annotation size

        the rand_factor is -0.2 ~ +0.2

        """
        rand_factor = 0.0
        if random_scale is False:
            rand_factor = 0.0
        elif random_scale is True:
            rand_factor = np.random.randn(1)*0.2

        if max(self._face_high, self._face_width) < std_size:
            scale = 1.0
            self._zoom_ratio = 0.0
        else:
            scale = (std_size / max(self._face_width, self._face_high))*(1.0 + rand_factor)
            self._zoom_ratio = rand_factor

        self._img = self.img_resize(self._img, self._img_width, self._img_high, scale)
        self._img_high = self._img.shape[0]
        self._img_width = self._img.shape[1]
        self._annotation = self._annotation * scale
        self._face_width = self._annotation[2] - self._annotation[0]
        self._face_high = self._annotation[3] - self._annotation[1]

    def property_change(self):
        """
        random select one of property in brightness(0), contrast(1), saturation(2), hue(3)
        but this slow in data process schedule, so put this in training process with tensorflow

        """
        seed = np.random.randint(0, 1000, 1)[0]
        op = np.random.randint(0, 4, 1)[0]
        op0 = tf.image.random_brightness(self._img, 0.5, seed=seed)
        op1 = tf.image.random_contrast(self._img, lower=0.2, upper=2.0, seed=seed)
        op2 = tf.image.random_saturation(image=self._img, lower=0.1, upper=1.5, seed=seed)
        op3 = tf.image.random_hue(image=self._img, max_delta=0.2, seed=seed)

        with tf.Session() as sess:
            if op == 0:
                ret = sess.run(op0)
                print('brightness')
            elif op == 1:
                ret = sess.run(op1)
                print('contrast')
            elif op == 2:
                ret = sess.run(op2)
                print('saturation')
            elif op == 3:
                ret = sess.run(op3)
                print('hue')

        self._img = ret

    def show(self):
        """
        show img use opencv api

        """
        cv2.namedWindow('win', flags=0)
        cv2.imshow('win', self._img)
        cv2.waitKey(0)

    def draw_annotation_box(self):
        """
        draw rectangle in img use opencv api

        """
        pt1 = tuple(self._annotation[0:2].astype(np.int32))
        pt2 = tuple(self._annotation[2:4].astype(np.int32))
        cv2.rectangle(self._img, pt1, pt2, (0, 255, 0), 1)

    def draw_bounding_box(self):
        """
        draw bounding_ box in img

        """
        pt1 = tuple(self._bounding_box[0:2].astype(np.int32))
        pt2 = tuple(self._bounding_box[2:4].astype(np.int32))
        cv2.rectangle(self._img, pt1, pt2, (0, 255, 0), 1)

    def img_resize(self, img, width, high, scale):
        return cv2.resize(img, (int(width*scale), int(high*scale)), interpolation=cv2.INTER_AREA)

    def iou(self, box, boxes):
        """
        Compute IoU between detect box and gt boxes

        Parameters:
        ----------
        box: numpy array , shape (4, ): x1, y1, x2, y2
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

    def random_bounding_box(self):
        factor = np.random.randn(2)*0.2
        self._bounding_box[0] = self._annotation[0] + factor[0]*24
        self._bounding_box[1] = self._annotation[1] + factor[1]*24
        self._bounding_box[2] = self._bounding_box[0] + 24
        self._bounding_box[3] = self._bounding_box[1] + 24

        if (self._bounding_box[0] < 0) or (self._bounding_box[0] >= self._img_width) or \
                (self._bounding_box[1] < 0) or (self._bounding_box[1] >= self._img_high) or \
                (self._bounding_box[2] < 0) or (self._bounding_box[2] >= self._img_width) or \
                (self._bounding_box[3] < 0) or (self._bounding_box[3] >= self._img_high):
            factor = np.random.randn(2) * 0.1
            self._bounding_box[0] = self._annotation[0] + factor[0] * 24
            self._bounding_box[1] = self._annotation[1] + factor[1] * 24
            self._bounding_box[2] = self._bounding_box[0] + 24
            self._bounding_box[3] = self._bounding_box[1] + 24

        if (self._bounding_box[0] < 0) or (self._bounding_box[0] >= self._img_width) or \
                (self._bounding_box[1] < 0) or (self._bounding_box[1] >= self._img_high) or \
                (self._bounding_box[2] < 0) or (self._bounding_box[2] >= self._img_width) or \
                (self._bounding_box[3] < 0) or (self._bounding_box[3] >= self._img_high):
            factor = np.random.randn(2) * 0.01
            self._bounding_box[0] = self._annotation[0] + factor[0] * 24
            self._bounding_box[1] = self._annotation[1] + factor[1] * 24
            self._bounding_box[2] = self._bounding_box[0] + 24
            self._bounding_box[3] = self._bounding_box[1] + 24

        if (self._bounding_box[0] < 0) or (self._bounding_box[0] >= self._img_width) or \
                (self._bounding_box[1] < 0) or (self._bounding_box[1] >= self._img_high) or \
                (self._bounding_box[2] < 0) or (self._bounding_box[2] >= self._img_width) or \
                (self._bounding_box[3] < 0) or (self._bounding_box[3] >= self._img_high):
            return False

        self.get_iou_ratio()
        self.get_position_ratio()
        return True

    def save_bounding_box(self, file):
        x_s = self._bounding_box[0]
        x_e = self._bounding_box[2]
        y_s = self._bounding_box[1]
        y_e = self._bounding_box[3]
        bounding_box = self._img[y_s:y_e, x_s:x_e]
        cv2.imwrite(file, bounding_box, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

# first test one img change
# second test img set change
class ImgSet(ImgBase):

    def __init__(self):
        pass


if __name__ == '__main__':
    img_file = '/home/kevin/face/wider_face/WIDER_train/images'
    label_file = './wider_face_train.txt'
    save_top_path = '/home/kevin/face/face_data'
    join = lambda a: os.path.join(img_file, a + '.jpg')
    join_save = lambda a: os.path.join(save_top_path, a + '.jpg')

    # join_save = lambda a: os.path.join(save_top_path, a + '.jpg')

    with open(label_file, 'r') as f:
        annotations = f.readlines()

    # select random img
    # idx_random_img = np.random.randint(0, len(annotations), 1)[0]
    idx_random_img = 922
    annotation = annotations[idx_random_img].strip().split(' ')

    """
    select random annotation
    """
    annotations = np.array(annotation[1:]).astype(np.float32).astype(np.int32).reshape([-1, 4])

    print(annotations.shape)

    time.sleep(1000)

    # idx_anno = np.random.randint(0, annotations.shape[0], 1)[0]
    idx_anno = 8

    # read img
    pic = cv2.imread(join(annotation[0]))

    img = ImgBase(pic, annotations[idx_anno])

    if img.check_face_size() is False:
        print('img face is smaller than 16')
        sys.exit()

    # img.property_change()
    for i in range(200):
        img_tmp = copy.copy(img)
        # img_tmp.property_change()
        img_tmp.zoom_change(24, random_scale=True)
        while img_tmp.random_bounding_box() is not True:
            img_tmp = copy.copy(img)
            # img_tmp.property_change()
            img_tmp.zoom_change(24, random_scale=True)

        face_ratio = img_tmp.get_face_probability()
        zoom_ratio = img_tmp.get_zoom_ratio()
        iou_ratio = img_tmp.get_iou_ratio()

        pos = img_tmp.get_position_ratio()

        if face_ratio > 0.9:
            save_file = join_save(str(i) + '=====' + str(face_ratio))
        else:
            save_file = join_save(str(i) + '_' + str(zoom_ratio) + '_' + str(iou_ratio) + '_' +
                              str(face_ratio) + '+' + str(pos[0]) + '_' + str(pos[1]) + '_' +
                              str(pos[2]) + '_' + str(pos[3]))



        img_tmp.save_bounding_box(save_file)

    # save_path = join_save(str(total_idx) + '_' + str(local_index) + '_' + str(box[0]) + '_' + str(box[1]))
    #
    # print(save_path)
    #
    # cv2.imwrite(save_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    img.show()
