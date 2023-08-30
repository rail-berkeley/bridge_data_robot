from multicam_server.srv import Image
import rospy
import numpy as np
from widowx_envs.utils import AttrDict


class CamClient:
    def __init__(self):
        self.service_name = 'image_server'
        try:
            request_image_srv_func = rospy.ServiceProxy(
                self.service_name, Image)
            response = request_image_srv_func()
        except rospy.ServiceException as e:
            raise e
        self.img_height = response.image_height
        self.img_width = response.image_width

    def request_images(self):
        rospy.wait_for_service(self.service_name)
        try:
            request_image_srv_func = rospy.ServiceProxy(
                self.service_name, Image)
            response = request_image_srv_func()
            image0 = np.asarray(response.image0).reshape((response.image_height, response.image_width, 3))
            image1 = np.asarray(response.image1).reshape((response.image_height, response.image_width, 3))
            return [response.time_stamp0, response.time_stamp1], [image0, image1]
        except rospy.ServiceException as e:
            print(e)

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return AttrDict(img_height=self.img_height, img_width=self.img_width)
