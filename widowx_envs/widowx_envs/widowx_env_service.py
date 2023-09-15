## This is the highest level interface to interact with the widowx setup.

import time
import cv2
import argparse
import numpy as np
from typing import Optional, Tuple, Any

# install from: https://github.com/youliangtan/edgeml
from edgeml.interfaces import EdgeClient, EdgeServer, EdgeConfig

##############################################################################

DefaultEdgeConfig = EdgeConfig(
        port_number = 5566,
        action_keys = ["init", "move", "gripper", "reset"],
        observation_keys = ["image", "proprio"],
        broadcast_port= 5566 + 1,
    )

print_red = lambda x: print("\033[91m{}\033[00m".format(x))

##############################################################################

class WidowXEdgeServer():
    """
    This is the highest level abstraction of the widowx setup. We will run
    this as a server, and we can have multiple clients connect to it and
    reveives the observation and control the widowx robot.
    """
    def __init__(self, port: int = 5556, testing: bool = True):
        edgeml_config = DefaultEdgeConfig
        edgeml_config.port_number = port

        self.testing = testing
        self.bridge_env = None
        self.__server = EdgeServer(edgeml_config,
                                   obs_callback=self.__observe,
                                   act_callback=self.__action)

    def start(self, threaded: bool = False):
        """
        This starts the server. Default is blocking.
        """
        self.__server.start(threaded)

    def __action(self, type: str, req_payload: dict) -> dict:
        if type == "init":
            if self.testing:
                print_red("WARNING: Running in testing mode, \
                    no env will be initialized.")
                return {}

            from widowx_envs.widowx_env import BridgeDataRailRLPrivateWidowX
            from multicam_server.topic_utils import IMTopic
            from tf.transformations import quaternion_from_euler
            from tf.transformations import quaternion_matrix

            # brute force way to convert json to IMTopic
            env_params = None
            if req_payload:
                cam_imtopic = []
                for cam in req_payload["camera_topics"]:
                    imtopic_obj = IMTopic.model_validate(cam)
                    cam_imtopic.append(imtopic_obj)
                req_payload["camera_topics"] = cam_imtopic
                env_params = req_payload

            def get_tf_mat(pose):
                # convert pose to a 4x4 tf matrix, rpy to quat
                quat = quaternion_from_euler(pose[3], pose[4], pose[5])
                tf_mat = quaternion_matrix(quat)
                tf_mat[:3, 3] = pose[:3]
                return tf_mat
            
            self.get_tf_mat = get_tf_mat
            self.bridge_env = BridgeDataRailRLPrivateWidowX(
                env_params, fixed_image_size=128)
            print("Initialized bridge env.")

        elif type == "gripper":
            self.__gripper(req_payload["open"])
        elif type == "move":
            self.__move(req_payload["pose"], req_payload["duration"])
        elif type == "reset":
            self.__reset()
        return {}
    
    def __observe(self, types: list) -> dict:
        if self.bridge_env:
            # we will default return image and proprio only
            obs = self.bridge_env.current_obs()
            obs = {
                    "image": obs["image"],
                    "state": obs["state"],
                    "full_image": obs["full_image"][0]
                  }
        else:
            # use dummy img with random noise
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            obs = {"image": img, "state": {}, "full_image": img}
            print_red("WARNING: No bridge env not initialized.")
        return obs

    def __gripper(self, open: float):
        if self.bridge_env:
            if open > 0.5: # convert to bool, for future float support
                self.bridge_env.controller().open_gripper()
            else:
                self.bridge_env.controller().close_gripper()
        else:
            print_red("WARNING: No bridge env not initialized.")

    def __move(self, pose: np.ndarray, duration: float) -> bool:
        if self.bridge_env:
            # TODO: test this! Is better to use the controller
            # to move directly to the pose, instead of the gym api
            # self.bridge_env.step(pose)
            self.bridge_env.controller().move_to_eep(
                self.get_tf_mat(pose), blocking=False)
        else:
            print_red("WARNING: No bridge env not initialized.")

    def __reset(self):
        if self.bridge_env:
            self.bridge_env.controller().move_to_neutral(duration=1.0)
            self.bridge_env.controller().open_gripper()
            # self.bridge_env.reset()
            # self.bridge_env.start()
        else:
            print_red("WARNING: No bridge env not initialized.")

    def stop(self):
        """Stop the server."""
        self.__server.stop()

##############################################################################

class WidowXClient():
    def __init__(self,
                 host: str = "localhost",
                 port: int = 5556,
                 ):
        edgeml_config = DefaultEdgeConfig
        edgeml_config.port_number = port
        self.__client = EdgeClient(host, edgeml_config)
        print("Initialized widowx client.")

    def move(self, pose: np.ndarray, duration: float) -> bool:
        """
        Command the arm to move to a given pose in space.
            :param pose: dim of 6, [x, y, z, roll, pitch, yaw]
        """
        assert len(pose) == 6
        if self.__client.act("move", {"pose": pose, "duration": duration}) is None:
            return False
        return True

    def move_gripper(self, state: float) -> bool:
        """Open or close the gripper. 1.0 is open, 0.0 is closed."""
        if self.__client.act("gripper", {"open": state}) is None:
            return False
        return True

    def init(self, env_params: dict) -> bool:
        """Initialize the environment."""
        if self.__client.act("init", env_params) is None:
            return False
        return True

    def reset(self) -> bool:
        """Reset the arm to the neutral position."""
        return False if self.__client.act("reset", {}) is None else True

    def get_observation(self) -> Optional[dict]:
        """
        Get the current camera image and proprioceptive state.
            :return a dict of observations
        """
        res = self.__client.obs()
        return res if res else None

    def stop(self):
        """Stop the client."""
        self.__client.stop()

##############################################################################

def show_video(client, duration, full_image=True):
    """This shows the video from the camera for a given duration."""
    start = time.time()
    while (time.time() - start) < duration:
        res = client.get_observation()
        if res is None:
            print("No observation available... waiting")
            continue

        if full_image:
            img = res["full_image"]
        else:
            img = res["image"]       
            # if img.shape[0] != 3:  # sanity check to make sure it's not flattened
            img = (img.reshape(3, 128, 128).transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imshow("img", img)
        cv2.waitKey(100)  # 100 ms

def main():
    # NOTE: This is just for Testing
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--client', action='store_true')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dont_init', action='store_true')
    args = parser.parse_args()

    if args.server:
        widowx_server = WidowXEdgeServer(port=args.port, testing=args.test)
        widowx_server.start()
        widowx_server.stop()

    if args.client:
        env_params = {
            "fix_zangle": 0.1,
            "move_duration": 0.2,
            "adaptive_wait": True,
            "move_to_rand_start_freq": 1,
            "override_workspace_boundaries": [
                [0.1, -0.15, -0.1, -1.57, 0],
                [0.45, 0.25, 0.18, 1.57, 0],
            ],
            "action_clipping": "xyz",
            "catch_environment_except": False,
            "start_state": None,
            "return_full_image": False,
            "camera_topics": [{"name": "/D435/color/image_raw", "flip": True}],
        }
        widowx_client = WidowXClient(host=args.ip, port=args.port)

        if not args.dont_init:
            # NOTE: this normally takes 10 seconds to reset
            widowx_client.init(env_params)
        else:
            widowx_client.reset()

        # This ensures that the robot is ready to be controlled
        obs = None
        while obs is None:
            obs = widowx_client.get_observation()
            time.sleep(1)
            print("Waiting for robot to be ready...")

        # Coordinate Convention:
        #  - x: forward
        #  - y: left
        #  - z: up
        
        # move left up
        widowx_client.move(np.array([0.2, 0.1, 0.3, 0, 1.57, 0]), 0.2)
        show_video(widowx_client, 1.5)

        # close gripper
        print("Closing gripper...")
        widowx_client.move_gripper(0.0)
        show_video(widowx_client, 2.5)

        # move right down
        widowx_client.move(np.array([0.2, -0.1, 0.1, 0, 1.57, 0]), 0.2)
        show_video(widowx_client, 1.5)

        # open gripper
        print("Opening gripper...")
        widowx_client.move_gripper(1.0)
        show_video(widowx_client, 2.5)

        widowx_client.stop()
        print("Done all")

if __name__ == '__main__':
    main()
