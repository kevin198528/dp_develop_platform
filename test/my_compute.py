import numpy as np

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

num = 1000

x_tmp = np.floor(np.array(np.random.randn(num, 1))*2)
y_tmp = np.floor(np.array(np.random.randn(num, 1))*2)


print(x_tmp)
print(y_tmp)

z = np.array([[0, 0, 16, 16]])

t = z.repeat(repeats=num, axis=0)

t[:, 0] = t[:, 0] + x_tmp[:, 0]

t[:, 2] = t[:, 2] + x_tmp[:, 0]

t[:, 1] = t[:, 1] + y_tmp[:, 0]

t[:, 3] = t[:, 3] + y_tmp[:, 0]

iou = IoU([0, 0, 16, 16], t)

# print(t)

print(iou)

print(min(iou))
print(max(iou))



a = 12.0
b = 16.0

print((a*a) /(b*b))

print(0.125*0.125*2)