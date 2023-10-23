from widowx_envs.utils.object_detection.object_detector import ObjectDetector
from scripts.ViLD import ViLD


class ObjectDetectorViLD(ObjectDetector):
    def __init__(self, env, save_dir="", image_xyz=None): 
        super().__init__(env, save_dir, image_xyz=image_xyz, detector_type='ViLD')

        self.game_board_str = "square game board"
        self.category_name_string = ';'.join(['black circle', 'white circle', self.game_board_str])
        self.category_names = ['background'] + [x.strip() for x in self.category_name_string.split(';')]
        self.v = ViLD() 

    def get_centroids(self, image_path): # need to change
        return self.v.get_centroids(image_path, self.category_names, self.game_board_str)[2]
        # return self.i.centroids

    def initialize_integrator(self, i):
        self.i = i
