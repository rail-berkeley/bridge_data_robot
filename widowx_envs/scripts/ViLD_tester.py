from widowx_envs.utils.params import WORKSPACE_BOUNDARIES
from widowx_envs.utils.object_detection.object_detector_ViLD import ObjectDetectorViLD 
from integration import Integrator
from widowx_envs.widowx.widowx_env import BridgeDataRailRLPrivateWidowX
from PIL import Image
import time 

i = Integrator() 

while True: 
    save_dir = "calcamtest/"

    env_params = { 
            'fix_zangle': 0.1,
            'move_duration': 0.2,
            'adaptive_wait': True,
            'move_to_rand_start_freq': 1,
            'override_workspace_boundaries': WORKSPACE_BOUNDARIES,
            'action_clipping': 'xyz',
            'catch_environment_except': False,
            'randomize_initpos': 'restricted_space',
            'skip_move_to_neutral': True,
            'return_full_image': True
    }
    env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=512)

    object_detector = ObjectDetectorViLD(env, save_dir) 

    #env.reset()

    img = object_detector._get_image()
    im = Image.fromarray(img)
    image_path = save_dir + "camera_obs.jpeg"
    im.save(image_path)

    board_state, bbox, centroids = object_detector.get_centroids(image_path)
    i.initialize_board_state(board_state, bbox, centroids)
    if i.game_over():
        break
    move = i.query_LLM()
    print("LLM Move:", move)
    #if move:
        #i.get_robot_coords(move)
        # start pick and place
    if i.game_over():
        break
    print()
    time.sleep(20) # we can definitely reduce this time if the robot does the pick and place instead of me