from widowx_envs.utils.object_detection.object_detector import ObjectDetector
from scripts.ViLD import ViLD
from PIL import Image


class ObjectDetectorViLD(ObjectDetector):
    def __init__(self, env, save_dir="", image_xyz=None): 
        super().__init__(env, save_dir, image_xyz=image_xyz, detector_type='ViLD')

        self.game_board_str = "square game board"
        self.white_piece_str = "white circle"
        self.black_piece_str = "black circle"
        #self.calibration_str = "green block"
        #self.category_name_string = ';'.join([self.calibration_str, self.white_piece_str, self.black_piece_str, self.game_board_str])
        self.category_name_string = ';'.join([self.white_piece_str, self.black_piece_str, self.game_board_str])
        self.category_names = ['background'] + [x.strip() for x in self.category_name_string.split(';')]
        self.v = ViLD() 

    def get_results(self, img): 
        im = Image.fromarray(img)
        image_path = self.save_dir + "camera_obs.jpeg"
        im.save(image_path)
        return self.v.get_results(image_path, self.category_names, self.game_board_str)

    def get_centroids(self, img): 
        return self.get_results(img)[2]
