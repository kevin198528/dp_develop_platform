import numpy as np
import tensorflow as tf
import cv2
import os
import copy
import sys

class ImgBase(object):

    def __init__(self, img, annotation):
        """
        raw img annotation and std size
        there is one face in img

        img: (high, width, channel)
        (0,0)   +x
            ------->
            |
            |
          + |
          y v
        annotation: [x1, y1, x2, y2]
        std_size: 24.0

        """
        self._img = img
        self._annotation = annotation
        self._scale = 1.0

        """
        get img width hight and face width high
        
        """
        self._img_high = self._img.shape[0]
        self._img_width = self._img.shape[1]
        self._face_width = self._annotation[2] - self._annotation[0]
        self._face_high = self._annotation[3] - self._annotation[1]

        self._bounding_box = np.array([0, 0, 0, 0])

        self._iou_current = 0.0
        self._iou_max = 0.0
        # self._iou_rat = self._iou_current + 0.001 / self._iou_max + 0.001

    def check_face_size(self, min_size = 16):
        if min(self._face_width, self._face_high) < min_size:
            return False
        return True

    def zoom_change(self, std_size=24.0, random_scale=False):
        """
        compute scale use std_size and face size
        ex. std_size = 24 face_size = 240 scale = 24 / 240 = 0.1
        then change img size and annotation size

        """
        rand_factor = 0.0
        if random_scale is False:
            rand_factor = 0.0
        elif random_scale is True:
            # rand_factor = np.random.randint(-1, 1, 1)[0] * 0.3
            rand_factor = np.random.randn(1)*0.2

        if max(self._face_high, self._face_width) < std_size:
            self._scale = 1.0
        else:
            self._scale = (std_size / max(self._face_width, self._face_high))*(1.0 + rand_factor)

        self._img = self.img_resize(self._img, self._img_width, self._img_high, self._scale)
        self._img_high = self._img.shape[0]
        self._img_width = self._img.shape[1]
        self._annotation = self._annotation * self._scale
        self._face_width = self._annotation[2] - self._annotation[0]
        self._face_high = self._annotation[3] - self._annotation[1]

    def property_change(self):
        """
        random select one of property in brightness(0), contrast(1), saturation(2), hue(3)

        """
        seed = np.random.randint(0, 1000, 1)[0]
        op = np.random.randint(0, 4, 1)[0]
        op0 = tf.image.random_brightness(self._img, 0.6, seed=seed)
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

        """
        if bounding_box is out of img must set bounding_box as annotation
        
        """
        if (self._bounding_box[0] < 0) or (self._bounding_box[0] >= self._img_width) or \
           (self._bounding_box[1] < 0) or (self._bounding_box[1] >= self._img_high) or \
           (self._bounding_box[2] < 0) or (self._bounding_box[2] >= self._img_width) or \
           (self._bounding_box[3] < 0) or (self._bounding_box[3] >= self._img_high):
            self._bounding_box[0] = self._annotation[0]
            self._bounding_box[1] = self._annotation[1]
            self._bounding_box[2] = self._bounding_box[0] + 24
            self._bounding_box[3] = self._bounding_box[1] + 24

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
    idx_random_img = 800
    annotation = annotations[idx_random_img].strip().split(' ')

    """
    select random annotation
    """
    annotations = np.array(annotation[1:]).astype(np.float32).astype(np.int32).reshape([-1, 4])
    # idx_anno = np.random.randint(0, annotations.shape[0], 1)[0]
    idx_anno = 0

    # read img
    pic = cv2.imread(join(annotation[0]))

    img = ImgBase(pic, annotations[idx_anno])

    if img.check_face_size() is False:
        print('img face is smaller than 16')
        sys.exit()

    # img.property_change()



    for i in range(100):
        img_tmp = copy.copy(img)
        img_tmp.property_change()
        img_tmp.zoom_change(24, random_scale=True)
        img_tmp.random_bounding_box()


        save_file = join_save(str(i))
        img_tmp.save_bounding_box(save_file)

    # save_path = join_save(str(total_idx) + '_' + str(local_index) + '_' + str(box[0]) + '_' + str(box[1]))
    #
    # print(save_path)
    #
    # cv2.imwrite(save_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    img.show()
