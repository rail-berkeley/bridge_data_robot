import torch
from widowx_envs.utils.object_detection import ObjectDetector
from widowx_envs.utils.params import *
from widowx_envs.utils.grasp_utils import get_image_obs
import os
from models.common import DetectMultiBackend, AutoShape
import numpy as np 


class ObjectDetectorDL(ObjectDetector):
    
    def __init__():
        pass 

    def get_bounding_box(self, img):
        # import pdb
        # pdb.set_trace()
        if not isinstance(img, np.ndarray):
            img = self.get_numpy(img)[0]
        
        results = self.model(img)
        if self.save_dir != "":
            results.save(os.path.join(self.save_dir, "{}_classifier".format(self.dl_detector_class)))
        df = results.pandas().xyxy[0]
        best_estimates = df.loc[df.groupby('class')['confidence'].idxmax()]
        bounding_boxes = dict()
        for i in range(len(best_estimates)):
            object_name = best_estimates.iloc[i]['name']
            bounding_boxes[object_name] = np.array([best_estimates.iloc[i]['ymin'],
                best_estimates.iloc[i]['ymax'], best_estimates.iloc[i]['xmin'], best_estimates.iloc[i]['xmax']]) \
                * self.scaling_factor

        return bounding_boxes
    
    def get_all_bounding_box(self, img):
        # get bounding boxes for all detected objects
        # import pdb
        # pdb.set_trace()
        if not isinstance(img, np.ndarray):
            img = self.get_numpy(img)[0]
        
        results = self.model(img)
        if self.save_dir != "":
            results.save(os.path.join(self.save_dir, "{}_classifier".format(self.dl_detector_class)))
        df = results.pandas().xyxy[0]
        bounding_boxes = []
        for i in range(len(df)):
            bounding_box = np.array([df.iloc[i]['ymin'], df.iloc[i]['ymax'], 
                                     df.iloc[i]['xmin'], df.iloc[i]['xmax']]) * self.scaling_factor
            bounding_boxes.append(bounding_box)

        return bounding_boxes

    def get_bounding_boxes_batch(self, imgs):
        results = self.model(imgs)
        if self.save_dir != "":
            results.save(os.path.join(self.save_dir), )
        ret_list = []
        dfs = results.pandas().xyxy
        for df in dfs:
            best_estimates = df.loc[df.groupby('class')['confidence'].idxmax()]
            bounding_boxes = dict()
            for i in range(len(best_estimates)):
                object_name = self.classes[best_estimates.iloc[i]['class']]
                bounding_boxes[object_name] = np.array([best_estimates.iloc[i]['ymin'],
                    best_estimates.iloc[i]['ymax'], best_estimates.iloc[i]['xmin'], best_estimates.iloc[i]['xmax']]) \
                    * self.scaling_factor
            ret_list.append(bounding_boxes)

        return ret_list

    @staticmethod
    def centroid_from_bounding_box(box):
        return np.array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2])

    @staticmethod
    def get_numpy(tensor):
        return tensor.to('cpu').detach().numpy()

    def get_centroids(self, img):
        bounding_boxes = self.get_bounding_box(img)
        centroids = dict()
        for key, box in bounding_boxes.items():
            centroids[key] = self.centroid_from_bounding_box(box)
        return centroids

    def get_centroids_batch(self, imgs):
        bounding_boxes_list = self.get_bounding_boxes_batch(imgs)
        ret = []
        for bounding_boxes in bounding_boxes_list:
            centroids = dict()
            for key, box in bounding_boxes.items():
                centroids[key] = self.centroid_from_bounding_box(box)
            ret.append(centroids)
        return ret
    
    def get_all_centroids(self, img):
        # get centroids for all detected objects
        bounding_boxes = self.get_all_bounding_box(img)
        centroids = []
        for box in bounding_boxes:
            centroids.append(self.centroid_from_bounding_box(box))
        return centroids

    def _get_image(self, transpose=False):
        obs = get_image_obs(self.env, self.image_xyz, skip_move_to_neutral=self.skip_move_to_neutral)
        if 'full_image' in obs:
            image = obs['full_image']
        else:
            if not self.size_warning_shown:
                print("Warning: 'full_image' key not present in obs. Parsing image with a DL model on 64x64x3 image.")
                self.size_warning_shown = True
            image = np.uint8(np.reshape(obs['image'] * 255, (3, 64, 64)))
        if transpose: image = np.transpose(image, (2, 0, 1))
        return image


if __name__ == '__main__':
    from widowx_envs.utils.multicam_server_rospkg.src.camera_recorder import CameraRecorder
    import rospy
    import cv2

    rospy.init_node('object_detector_dl', anonymous=True)
    recorder = CameraRecorder('/camera0/image_raw', shutdown_on_repeated_images=False)
    while not rospy.is_shutdown():
        img, _ = recorder.get_image_timestamp()
        detector = ObjectDetectorDL(save_dir='/tmp/')
        detected_objects_img = detector.draw_selected_objects(img)
        cv2.imshow("Image window", detected_objects_img)
        cv2.waitKey(3)
