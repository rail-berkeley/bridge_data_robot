
from widowx_envs.utils.grasp_utils import rgb_to_robot_coords, get_image
from widowx_envs.utils.params import (KMEANS_RGB_TO_ROBOT_TRANSMATRIX,
    DL_RGB_TO_ROBOT_TRANSMATRIX)
import numpy as np


class ObjectDetector:
    def __init__(self, env, save_dir, detector_type='base',
            image_xyz=None, skip_move_to_neutral=False):
        self.env = env
        self.save_dir = save_dir
        self.detector_type = detector_type
        self.image_xyz = image_xyz
        self.skip_move_to_neutral = skip_move_to_neutral

        if self.detector_type == 'ViLD':
            self.transmatrix = KMEANS_RGB_TO_ROBOT_TRANSMATRIX # need to change 
        elif self.detector_type == 'kmeans':
            self.transmatrix = KMEANS_RGB_TO_ROBOT_TRANSMATRIX
        elif self.detector_type == 'dl':
            self.transmatrix = DL_RGB_TO_ROBOT_TRANSMATRIX
        elif self.detector_type == 'manual':
            self.transmatrix = DL_RGB_TO_ROBOT_TRANSMATRIX
        else:
            raise NotImplementedError

    def get_centroids(self, img):
        raise NotImplementedError
    
    def get_all_centroids(self, img):
        raise NotImplementedError

    def go_neutral_and_get_random_center_query(self, skip_move_to_neutral=False):
        if not self.skip_move_to_neutral:
            self.env.move_to_neutral(1.5)
        else:
            self.env.reset()
        user_in = input('Press t to get empty rgb image ')
        while user_in != 't':
            user_in = input('Press t to get empty rgb image ')
        if hasattr(self, 'background_image'):
            self.background_image = self._get_image()
        loop_counter = 0
        goal = None
        while goal is None:
            self.env.move_to_neutral(1.5)
            user_in = input('Press c to continue ')
            while user_in != 'c':
                user_in = input('Press c to continue ')
            goal = self.go_neutral_and_get_random_center()
            loop_counter += 1
            if loop_counter >= 5:
                break

        return goal

    def get_all_centers(self, image, transform=True):
        # centers of the best estimates for each object class
        centroids = self.get_centroids(image)
        if len(centroids) == 0:
            return None
        centers = dict()
        for idx, center in centroids.items():
            if transform:
                centers[idx] = rgb_to_robot_coords(center, self.transmatrix)
            else:
                centers[idx] = center
        return centers
    
    def get_all_detected_centers(self, transform=True):
        # centers of all objects detected (can have the same class label)
        image = self._get_image()
        if image.ndim == 4:
            image = image[0]
        centroids = self.get_all_centroids(image)
        if len(centroids) == 0:
            return None
        centers = []
        for centroid in centroids:
            if transform:
                centers.append(rgb_to_robot_coords(centroid, self.transmatrix))
            else:
                centers.append(centroid)
        return centers

    def go_neutral_and_get_all_centers(self, return_image=False, transform=True):
        if hasattr(self, 'background_image') and self.background_image is None:
            return self.go_neutral_and_get_random_center_query()
        image = self._get_image()
        if image.ndim == 4:
            image = image[0]
        if return_image:
            original_image = image.copy()
        centers = self.get_all_centers(image, transform=transform)
        if return_image:
            return centers, original_image
        else:
            return centers

    def go_neutral_and_get_random_center(self, return_image=False, transform=True):
        if hasattr(self, 'background_image') and self.background_image is None:
            return self.go_neutral_and_get_random_center_query()
        image = self._get_image()
        if return_image:
            original_image = image.copy()
        centroids = self.get_all_centers(image, transform=transform)
        random_centroid = centroids[np.random.choice(list(centroids.keys()))]
        if return_image:
            return random_centroid, original_image
        else:
            return random_centroid

    def go_neutral_and_get_center_idx(self, idx):
        if hasattr(self, 'background_image') and self.background_image is None:
            return self.go_neutral_and_get_random_center_query()
        image = self._get_image()
        centroids = self.get_centroids(image)
        if len(centroids) == 0:
            return None
        if idx not in centroids.keys():
            return None
        centroid = centroids[idx]
        return rgb_to_robot_coords(centroid, self.transmatrix)

    def _get_image(self, transpose=True):
        return get_image(self.env, transpose, self.image_xyz, self.skip_move_to_neutral)

    def try_make_background_image(self):
        pass
