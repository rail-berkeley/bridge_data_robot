from widowx_envs.utils.params import WORKSPACE_BOUNDARIES
from widowx_envs.utils.object_detection.object_detector_ViLD import ObjectDetectorViLD 
from integration import Integrator
from widowx_envs.widowx.widowx_env import BridgeDataRailRLPrivateWidowX
from PIL import Image
import time 

i = Integrator() 

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

z = 0 
while z!=1: 

    #env.reset()

    img = object_detector._get_image()
    board_state, bbox, centroids = object_detector.get_results(img)

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
    time.sleep(10) # we can definitely reduce this time if the robot does the pick and place instead of me
    z+=1 