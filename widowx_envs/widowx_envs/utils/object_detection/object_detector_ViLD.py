from widowx_envs.utils.object_detection.object_detector import ObjectDetector
from scripts.ViLD import ViLD
from PIL import Image


class ObjectDetectorViLD(ObjectDetector):
    def __init__(self, env, save_dir="", image_xyz=None): 
        super().__init__(env, save_dir, image_xyz=image_xyz, detector_type='ViLD')
        self.v = ViLD() 

    def get_results(self, img): 
        im = Image.fromarray(img)
        image_path = self.save_dir + "camera_obs.jpeg"
        im.save(image_path)
        return self.v.get_results(image_path, {"white circle": 'X', "black circle": "O", "green block": "Z"}, "square game board")

    def get_centroids(self, img): 
        return self.get_results(img)[2]
