from widowx_envs.utils.params import *
from widowx_envs.utils.object_detection.object_detector_ViLD import ObjectDetectorViLD 
from integration import Integrator
from widowx_envs.widowx.widowx_env import BridgeDataRailRLPrivateWidowX
from PIL import Image
import time 
from widowx_envs.policies.scripted_pickplace import PickPlacePolicy
from widowx_envs.utils.exceptions import Environment_Exception
import numpy as np
import os
import argparse
import datetime
from PIL import Image
import logging
import pickle
import time

STEP_DURATION = 0.2
INITIAL_GRIPPER_CLOSE_PROB = 0.5

logging.getLogger('robot_logger').setLevel(logging.WARN)

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--object")
parser.add_argument("-d", "--data-save-directory", type=str, default="pickplacetest/")
parser.add_argument("-i", "--image_save_directory", type=str, default='calcamtest/')
parser.add_argument("--detector", choices=('kmeans', 'dl', 'manual', 'ViLD'), default='ViLD')
parser.add_argument("--tsteps", type=int, default=50)
parser.add_argument("--action-noise", type=float, default=0.005)


args = parser.parse_args()

args.data_save_directory = os.path.expanduser(args.data_save_directory)
directory_name = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
save_dir = os.path.join(args.data_save_directory, directory_name)
os.makedirs(save_dir)

i = Integrator() 

env_params = { 
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'adaptive_wait': True,
    'move_to_rand_start_freq': 1,
    'override_workspace_boundaries': TIC_TAC_TOE_WORKSPACE_BOUNDARIES,
    'action_clipping': 'xyz',
    'catch_environment_except': False,
    'randomize_initpos': 'restricted_space',
    'skip_move_to_neutral': True,    
}

env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=512)

if args.detector == 'ViLD':
    object_detector = ObjectDetectorViLD(env, save_dir=args.image_save_directory)
else:
    raise NotImplementedError

scripted_policy = PickPlacePolicy(env)

consecutive_detection_failures = 0

z = 0 
while z!=1: 
    img = object_detector._get_image()
    board_state, bbox, centroids = object_detector.get_results(img)
    if not board_state:
        break

    i.initialize_board_state(board_state, bbox, centroids)
    if i.game_over():
        break
    move = i.query_LLM()
    print("LLM Move:", move)
    if move:
        drop_point = i.get_robot_coords(move)
        
        print("Starting Trajectory")
        obs = env.reset()
        env.start()
        
        if np.random.rand() < INITIAL_GRIPPER_CLOSE_PROB:
            initial_gripper = 0.
            action = np.zeros(7)
            obs, _, _, _ = env.step(action, time.time() + 1, blocking=False)
        else:
            initial_gripper = 1.0

        if args.object:
            grasp_object_name = args.object
            centroids = object_detector.go_neutral_and_get_all_centers(transform=True) 
            print("DETECTED OBJECTS", centroids)
            keys = [x for x in list(centroids.keys()) if grasp_object_name in x]
            pick_point = centroids[keys[0]]
            print('PICKING UP ' + str(keys[0]))
        else:
            raise ValueError("must specify object name or set random object selection to True")
        
        #drop_point = np.array([0.50, 0.1])
        drop_point_z = PICK_POINT_Z
        pick_point_z = PICK_POINT_Z
        pick_point = np.append(pick_point, pick_point_z)
        drop_point = np.append(drop_point, drop_point_z)

        scripted_policy.reset(pick_point=pick_point, drop_point=drop_point, initial_gripper=initial_gripper)
        current_save_dir = os.path.join(save_dir, f"raw/traj_group0/traj{i}")
        image_save_dir = os.path.join(current_save_dir, "images0")
        os.makedirs(image_save_dir)

        full_image = obs['full_image'][0]

        observations = []
        actions = []
        images = []

        j = 0 
        last_tstep = time.time()
        #env.reset() 
        while j < args.tsteps:
            if time.time() > last_tstep + STEP_DURATION:
                last_tstep = time.time()
                action, agent_info = scripted_policy.get_action(obs=obs, curr_ts=j)
                action = np.random.normal(loc=action, scale=args.action_noise)
                action = np.clip(action, -1.0, 1.0)
                try:
                    next_obs, rew, done, info = env.step(action, last_tstep+STEP_DURATION, blocking=False)
                except Environment_Exception:
                    break

                full_image_next = next_obs['full_image'][0]

                im = Image.fromarray(full_image)
                imfilepath = os.path.join(image_save_dir, '{}.jpeg'.format(j))
                im.save(imfilepath)

                obs['full_obs'].pop('images')
                observations.append(obs['full_obs'])
                actions.append({"actions": action})

                obs = next_obs
                full_image = full_image_next
                
                j += 1

        # save final obs 
        im = Image.fromarray(full_image)
        imfilepath = os.path.join(current_save_dir, 'images0/{}.jpeg'.format(j))
        im.save(imfilepath)

        if j == args.tsteps:
            # save if traj is complete
            print("saving trajectory")
            with open(os.path.join(current_save_dir, "obs_dict.pkl"), 'wb') as handle:
                pickle.dump(observations, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(current_save_dir, "policy_out.pkl"), 'wb') as handle:
                pickle.dump(actions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if i.game_over():
        break
    print()
    time.sleep(10) # we can definitely reduce this time if the robot does the pick and place instead of me
    z+=1 