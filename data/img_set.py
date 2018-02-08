import numpy as np
import tensorflow as tf
import cv2
import os
import copy
import sys
import time
import pickle
import matplotlib.pyplot as plt


class ImgBase(object):

    def __init__(self, img, anno_boxes, face_idx):
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
        self._bounding_box = np.array([0, 0, 0, 0])
        self._zero_box = np.array([0, 0, 0, 0])
        self._annotation = anno_boxes[face_idx]
        self._anno_boxes = anno_boxes

        """
        _zoom_ratio: -0.2~0.2 describe the picture zoom degree
         
        """
        self._zoom_ratio = 0.0

        """
        get img width hight and face width high
        
        """
        self._img_high = self._img.shape[0]
        self._img_width = self._img.shape[1]
        self._face_width = self._annotation[2] - self._annotation[0]
        self._face_high = self._annotation[3] - self._annotation[1]

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
        return (iou_current / iou_max)[0]

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
            rand_factor = np.random.randn(1)[0]*0.2

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
        self._anno_boxes = self._anno_boxes * scale
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

    def random_zero_box(self):
        local_count = 0
        while True:
            local_count += 1
            if local_count > 100:
                break

            if self._img_width < 50 or self._img_high < 50:
                break

            random_x = np.random.randint(0, self._img_width - 24, 1)[0]
            random_y = np.random.randint(0, self._img_high - 24, 1)[0]
            zero_box = np.array([random_x, random_y, random_x + 24, random_y + 24])
            zero_iou = self.iou(zero_box, self._anno_boxes)

            if 0.0 == np.sum(zero_iou):
                self._zero_box = zero_box.copy()
                return True

        return False

    def random_bounding_box(self):
        factor = np.random.randn(2)*0.3
        self._bounding_box[0] = self._annotation[0] + factor[0]*24
        self._bounding_box[1] = self._annotation[1] + factor[1]*24
        self._bounding_box[2] = self._bounding_box[0] + 24
        self._bounding_box[3] = self._bounding_box[1] + 24

        if (self._bounding_box[0] < 0) or (self._bounding_box[0] >= self._img_width) or \
                (self._bounding_box[1] < 0) or (self._bounding_box[1] >= self._img_high) or \
                (self._bounding_box[2] < 0) or (self._bounding_box[2] >= self._img_width) or \
                (self._bounding_box[3] < 0) or (self._bounding_box[3] >= self._img_high):
            factor = np.random.randn(2) * 0.10
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

    def get_bounding_box_img(self):
        x_s = self._bounding_box[0]
        x_e = self._bounding_box[2]
        y_s = self._bounding_box[1]
        y_e = self._bounding_box[3]
        bounding_box = self._img[y_s:y_e, x_s:x_e]
        bounding_box = np.reshape(bounding_box, [1, 24, 24, 3])
        return bounding_box

    def save_zero_box(self, file):
        x_s = self._zero_box[0]
        x_e = self._zero_box[2]
        y_s = self._zero_box[1]
        y_e = self._zero_box[3]
        zero_box = self._img[y_s:y_e, x_s:x_e]
        cv2.imwrite(file, zero_box, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def get_zero_box_img(self):
        x_s = self._zero_box[0]
        x_e = self._zero_box[2]
        y_s = self._zero_box[1]
        y_e = self._zero_box[3]
        zero_box = self._img[y_s:y_e, x_s:x_e]
        zero_box = np.reshape(zero_box, [1, 24, 24, 3])
        return zero_box

# first test one img change
# second test img set change
class ImgSet(object):

    def __init__(self, img_file, label_file, face_to_box_num=10):
        self._img_file = img_file
        self._label_file = label_file
        self._face_to_box_num = face_to_box_num

        with open(label_file, 'r') as file:
            self._annotations = file.readlines()

    def check_path(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def create_img_view(self, save_path, num):
        self.check_path(save_path)

        join_read = lambda pic: os.path.join(self._img_file, pic + '.jpg')
        join_save = lambda pic: os.path.join(save_path, pic + '.jpg')



        face_idx = 0
        while face_idx < num:
            idx_random_img = np.random.randint(0, len(self._annotations), 1)[0]
            img_anno = self._annotations[idx_random_img].strip().split(' ')
            img_anno_boxs = np.array(img_anno[1:]).astype(np.float32).astype(np.int32).reshape([-1, 4])
            idx_random_face = np.random.randint(0, img_anno_boxs.shape[0], 1)[0]

            pic = cv2.imread(join_read(img_anno[0]))

            img = ImgBase(pic, img_anno_boxs, idx_random_face)

            if img.check_face_size() is False:
                print('img face is smaller than 16')
                continue

            face_box = 0
            while face_box < self._face_to_box_num:
                # img_tmp = copy.copy(img)
                # # img_tmp.property_change()
                # img_tmp.zoom_change(24, random_scale=True)
                # while img_tmp.random_bounding_box() is not True:
                #     img_tmp = copy.copy(img)
                #     # img_tmp.property_change()
                #     img_tmp.zoom_change(24, random_scale=True)

                local_count = 0
                while True:
                    local_count += 1
                    if local_count > 100:
                        break
                    img_tmp = copy.copy(img)
                    img_tmp.zoom_change(24, random_scale=True)
                    if img_tmp.random_bounding_box() is True:
                        break

                face_ratio = img_tmp.get_face_probability()
                if face_ratio <= 0:
                    continue

                # img_tmp.random_zero_box()

                zoom_ratio = img_tmp.get_zoom_ratio()
                iou_ratio = img_tmp.get_iou_ratio()

                pos = img_tmp.get_position_ratio()

                # if face_ratio > 0.9:
                #     save_file = join_save(str(i) + '=====' + str(face_ratio))
                # else:
                save_file = join_save(str(face_idx) + '_' + str(face_box) + '_' + str(zoom_ratio) + '_' +
                                      str(iou_ratio) + '_' + str(face_ratio) + '+' + str(pos[0]) + '_' +
                                      str(pos[1]) + '_' + str(pos[2]) + '_' + str(pos[3]))

                img_tmp.save_bounding_box(save_file)



                if img_tmp.random_zero_box() is True:
                    zero_file = join_save('zs' + '_' + str(face_idx) + '_' + str(face_box) + '_' + 'ze')
                    img_tmp.save_zero_box(zero_file)

                face_box += 1

            face_idx += 1

    def create_tfrecord(self, save_path, num):
        pass

    def create_pickle(self, save_path, num):
        self.check_path(save_path)

        join_read = lambda pic: os.path.join(self._img_file, pic + '.jpg')
        join_save = lambda pic: os.path.join(save_path, pic + '.jpg')

        data = np.zeros([1, 24, 24, 3], np.int32)
        label = np.zeros([1, 6], np.float32)

        face_idx = 0
        while face_idx < num:
            idx_random_img = np.random.randint(0, len(self._annotations), 1)[0]
            img_anno = self._annotations[idx_random_img].strip().split(' ')
            img_anno_boxs = np.array(img_anno[1:]).astype(np.float32).astype(np.int32).reshape([-1, 4])
            idx_random_face = np.random.randint(0, img_anno_boxs.shape[0], 1)[0]

            pic = cv2.imread(join_read(img_anno[0]))

            img = ImgBase(pic, img_anno_boxs, idx_random_face)

            if img.check_face_size() is False:
                print('img face is smaller than 16')
                continue

            face_box = 0
            while face_box < self._face_to_box_num:
                # img_tmp = copy.copy(img)
                # # img_tmp.property_change()
                # img_tmp.zoom_change(24, random_scale=True)
                # while img_tmp.random_bounding_box() is not True:
                #     img_tmp = copy.copy(img)
                #     # img_tmp.property_change()
                #     img_tmp.zoom_change(24, random_scale=True)

                local_count = 0
                while True:
                    local_count += 1
                    if local_count > 100:
                        break
                    img_tmp = copy.copy(img)
                    img_tmp.zoom_change(24, random_scale=True)
                    if img_tmp.random_bounding_box() is True:
                        break

                face_ratio = img_tmp.get_face_probability()
                if face_ratio <= 0:
                    continue

                # img_tmp.random_zero_box()

                zoom_ratio = img_tmp.get_zoom_ratio()
                iou_ratio = img_tmp.get_iou_ratio()

                pos = img_tmp.get_position_ratio()

                # if face_ratio > 0.9:
                #     save_file = join_save(str(i) + '=====' + str(face_ratio))
                # else:
                save_file = join_save(str(face_idx) + '_' + str(face_box) + '_' + str(zoom_ratio) + '_' +
                                      str(iou_ratio) + '_' + str(face_ratio) + '+' + str(pos[0]) + '_' +
                                      str(pos[1]) + '_' + str(pos[2]) + '_' + str(pos[3]))

                # append data
                img_box = img_tmp.get_bounding_box_img()

                data = np.append(data, img_box, axis=0)

                # append label
                label_data = np.array([[zoom_ratio, iou_ratio, pos[0], pos[1], pos[2], pos[3]]], np.float32)
                label = np.append(label, label_data, axis=0)

                img_tmp.save_bounding_box(save_file)

                if img_tmp.random_zero_box() is True:
                    # append data
                    data = np.append(data, img_tmp.get_zero_box_img(), axis=0)
                    # append label
                    label_zero = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], np.float32)

                    label = np.append(label, label_zero, axis=0)

                    zero_file = join_save('zs' + '_' + str(face_idx) + '_' + str(face_box) + '_' + 'ze')
                    img_tmp.save_zero_box(zero_file)

                face_box += 1

            face_idx += 1

        return data, label


if __name__ == '__main__':
    # img_file = '/home/kevin/face/wider_face/WIDER_train/images'
    #
    # label_file = '../doc/wider_face_train.txt'
    #
    # view_save_path = '/home/kevin/face/face_data/view'
    #
    # tfrecord_save_path = '/home/kevin/face/face_data/tfrecord'
    #
    # # record_num = 1000
    #
    # view_num = 500
    #
    # process_img_set = ImgSet(img_file, label_file)
    #
    # # process_img_set.create_img_view(view_save_path, view_num)
    #
    # data, label = process_img_set.create_pickle(view_save_path, view_num)
    #
    # dic = {'data': data[1:], 'lable': label[1:]}
    #
    # j = pickle.dumps(dic)
    #
    # f = open('1000_pic_pickle', 'wb')  # 注意是w是写入str,wb是写入bytes,j是'bytes'
    # f.write(j)  # -------------------等价于pickle.dump(dic,f)
    #
    # f.close()
    # -------------------------反序列化

    f = open('1000_pic_pickle', 'rb')

    record = pickle.loads(f.read())  # 等价于data=pickle.load(f)

    data = record['data']
    label = record['lable']

    print(data.shape)
    print(label.shape)

    print(label[40])

    img = data[40].astype(np.uint8)
    img = img[:, :, (2, 1, 0)]

    plt.imshow(img)
    plt.show()


    # cv2.namedWindow('win1', flags=0)
    # print(data[1][0])
    # cv2.imshow('win1', data[1][0])
    # cv2.waitKey(0)

    # a = np.array([[[1, 2], [3, 4]]])
    # b = np.array([[[5, 6], [7, 8]]])
    #
    # a = np.append(a, b, axis=0)
    #


    # print(a)


    # d = np.array([1, 2, 3])
    # l = np.array([4, 5, 6])
    #
    # dic = {'data':d, 'lable':l}
    #
    # print(type(dic))  # <class 'dict'>
    #
    # j = pickle.dumps(dic)
    # print(type(j))  # <class 'bytes'>
    #
    # f = open('序列化对象_pickle', 'wb')  # 注意是w是写入str,wb是写入bytes,j是'bytes'
    # f.write(j)  # -------------------等价于pickle.dump(dic,f)
    #
    # f.close()
    # # -------------------------反序列化

    # f = open('序列化对象_pickle', 'rb')
    #
    # data = pickle.loads(f.read())  # 等价于data=pickle.load(f)
    #
    # print(type(data['data']))
