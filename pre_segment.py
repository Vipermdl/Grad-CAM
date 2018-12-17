import cv2
import glob
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

class Drawcnts_and_cut(object):
    def __call__(self, original_img, box):
        filter = lambda x: 0 if x < 0 else x
        Xs = [filter(i[0]) for i in box]
        Ys = [filter(i[1]) for i in box]
        # Xs = [i[0] for i in box]
        # Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        crop_img = original_img[y1:y1 + hight, x1:x1 + width]
        return crop_img

class Crop_image(object):
    def __init__(self):
        # set color bound
        self.lower_grey = np.array([0, 0, 46]) # 灰色的hsv范围
        self.upper_grey = np.array([180, 43, 220])
        self.drawcnts_and_cut = Drawcnts_and_cut()

    def __call__(self, img_fname, source_path, target_path):
        image = cv2.imread(osp.join(source_path, img_fname))
        image = cv2.resize(image, None, fx=0.5, fy=0.5)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_grey, self.upper_grey)

        cv2.imshow("draw_img", image)
        cv2.waitKey(0)
        cv2.imshow("draw_img", hsv)
        cv2.waitKey(0)
        cv2.imshow("draw_img", mask)
        cv2.waitKey(0)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(img_grey, 127, 255, 0)
        # _, contours, hierarcy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        _, contours, hierarcy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))

        # draw a bounding box arounded the detected barcode and display the image
        # draw_img = cv2.drawContours(image.copy(), [box], -1, (0, 0, 255), 3)
        draw_img = image.copy()

        draw_img = self.drawcnts_and_cut(draw_img, box)

#        cv2.imwrite(osp.join(source_path, img_fname.split('.')[0]+'_.'+img_fname.split('.')[1]), draw_img)

        # return draw_img



if __name__ == '__main__':
    source_path = "guangdong_round2_test_a_20181011"
    target_path = "pre_image"
    image_list = os.listdir(source_path)
    image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))
    for img_fname in tqdm(image_list[111:112]):
        Crop_image()(img_fname, source_path, target_path)
