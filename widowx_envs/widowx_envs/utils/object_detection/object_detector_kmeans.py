import numpy as np
import cv2
from scipy.cluster.vq import kmeans
from widowx_envs.utils.object_detection.object_detector import ObjectDetector
from widowx_envs.utils.params import POINT_EXTRACTION_ROW_CUTOFF


class ObjectDetectorKmeans(ObjectDetector):
    def __init__(self, env, save_dir="", image_xyz=None, num_objects=1):
        super().__init__(env, save_dir, image_xyz=image_xyz, detector_type='kmeans')
        self.mask_diff_thresh = 15
        self.canvas_diff_thresh = 150 # 200
        self.max_byte_val = 255
        self.point_extraction_row_cutoff = POINT_EXTRACTION_ROW_CUTOFF
        self.extract_points_thresh = 20 # 20
        self.extract_points_stride = 1

        self.num_objects = num_objects
        self.background_image = None

    def get_centroids(self, image):
        img, canvas = self._background_subtraction(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self._downsample_average(img)
        points = self._extract_points(img, self.extract_points_stride, self.extract_points_thresh)
        centroids = self._k_means(points)
        centroids = {i: centroids[i] for i in range(len(centroids))}
        return centroids

    def try_make_background_image(self):
        user_in = input('Press t to get empty rgb image ')
        while user_in != 't':
            user_in = input('Press t to get empty rgb image ')
        self.background_image = self._get_image()

    def _background_subtraction(self, image):
        diff = cv2.absdiff(image, self.background_image)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        imask = mask > self.mask_diff_thresh

        canvas = np.zeros_like(image, np.uint8)
        canvas[imask] = image[imask]
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        # Canvas is grayscale. Convert to black (0) and white (255) values only.
        canvas = self.max_byte_val * (canvas > self.canvas_diff_thresh)

        if self.save_dir:
            cv2.imwrite(self.save_dir + "/image0.png", self.background_image)
            cv2.imwrite(self.save_dir + "/image1.png", image)
            cv2.imwrite(self.save_dir + "/rgbdiff.png", diff)
            cv2.imwrite(self.save_dir + "/diff.png", canvas)
        return diff, canvas

    def _downsample_average(self, image, num_pixels=4):
        new_image = np.copy(image)
        height, width = image.shape

        for i in range(num_pixels, height - num_pixels):
            for j in range(num_pixels, width - num_pixels):
                new_image[i][j] = np.mean(image[i - num_pixels:i+num_pixels, j-num_pixels:j+num_pixels])

        if self.save_dir:
            cv2.imwrite(self.save_dir + '/averaged.png', new_image)
        return new_image

    def _k_means(self, points):
        centroids, _ = kmeans(points, self.num_objects)
        return centroids

    def _extract_points(self, image, stride=4, threshold=50, save_dir=""):
        height, width = image.shape
        new_image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        points = []

        for i in range(0, height, stride):
            for j in range(0, width, stride):
                if image[i][j] > threshold and i > (self.point_extraction_row_cutoff * height):
                    new_image = cv2.circle(new_image, (j, i), radius=1, color=(0, 0, 255), thickness=1)
                    points.append([i, j])

        if save_dir != "":
            cv2.imwrite(save_dir + '/extracted.png', new_image)
        return np.array(points, dtype='float')
