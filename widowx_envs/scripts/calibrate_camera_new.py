# from widowx_envs.widowx.widowx_grasp_env import GraspWidowXEnv
from widowx_envs.policies.scripted_reach import ReachPolicy
from widowx_envs.utils.grasp_utils import (
    compute_robot_transformation_matrix,
    rgb_to_robot_coords,
    execute_reach
)
from widowx_envs.utils.params import *
from widowx_envs.utils.object_detection.object_detector_ViLD import ObjectDetectorViLD 
from widowx_envs.widowx.widowx_env import BridgeDataRailRLPrivateWidowX
from tqdm import tqdm
import numpy as np
import argparse
import time
from sklearn.preprocessing import PolynomialFeatures
import torch
from widowx_envs.utils.params import *
from PIL import Image

BOUNDS = TIC_TAC_TOE_WORKSPACE_BOUNDARIES

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).to(device).float()

def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")

def generate_goals(env, retry=0, change_per_retry=0.03, starting_eps=0.03):
    reach_point_z = 0.03
    
    eps = starting_eps + retry * change_per_retry
    x_lo, x_hi = BOUNDS[0][0] + eps, BOUNDS[1][0] - eps
    y_lo, y_hi = BOUNDS[0][1] + eps, BOUNDS[1][1] - eps
    goals = [
            [x_lo, y_hi, reach_point_z],
            [0.5 * (x_lo + x_hi), y_hi, reach_point_z],
            [x_hi, y_hi, reach_point_z],
            [x_lo, 0.5 * (y_hi + y_lo), reach_point_z],
            [0.5 * (x_lo + x_hi), 0.5 * (y_hi + y_lo), reach_point_z],
            [x_hi, 0.5 * (y_hi + y_lo), reach_point_z],
            [x_lo, y_lo, reach_point_z],
            [0.5 * (x_lo + x_hi), y_lo, reach_point_z],
            [x_hi, y_lo, reach_point_z],
        ]
    return goals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_dir", type=str, default="calcamtest/")
    parser.add_argument("-d", "--detector", required=False, choices=('ViLD'), default='ViLD')
    parser.add_argument('--skip-move-to-neutral', action='store_true', default=False)
    args = parser.parse_args()
    
    set_gpu_mode(True)


    env_params = { 
        'fix_zangle': 0.1,
        'move_duration': 0.2,
        'adaptive_wait': True,
        'move_to_rand_start_freq': 1,
        'override_workspace_boundaries': BOUNDS,
        'action_clipping': 'xyz',
        'catch_environment_except': False,
        'randomize_initpos': 'restricted_space',
        'skip_move_to_neutral': True,
        'return_full_image': True
    }
    env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=512)

    if args.detector == 'ViLD':
        object_detector = ObjectDetectorViLD(env, args.save_dir) 
    else:
        raise NotImplementedError

    robot_coords = []
    rgb_coords = []

    reach_policy = ReachPolicy(env, reach_point=None)

    goals = generate_goals(env)

    print("GOALS:", goals)

    # env._controller.move_to_state([0.30, 0, 0.15],0, 3)

    for j in tqdm(range(len(goals))):
        retry = 0
        success = False
        while not success:
            # import ipdb; ipdb.set_trace()
            env.reset()
            # obs = execute_reach(env, reach_policy, goals[j])
            env._controller.move_to_state([0.30, 0, 0.15], 0, 3)
            env._controller.move_to_state(goals[j], 0, duration=3)
            obs = env.current_obs()
            input('Press ENTER to calibrate the point')
            before = time.time()
            if not args.skip_move_to_neutral:
                env.move_to_neutral()
            else:
                env.reset()
            if time.time() - before > 3:
                print("Movement to neutral position failed. Changing the goal position.")
                retry += 1
                goals[j] = generate_goals(env, retry=retry)[j]
                continue

            img = object_detector._get_image()

            print("Current Goal:", goals[j])
            centroids = object_detector.get_centroids(img)
            print("DETECTED OBJECTS:", centroids)
            ee_coord = obs['full_obs']['full_state'][:3]
            robot_coords.append(ee_coord)
            keys = list(centroids.keys())
            keys = [x for x in keys if "white circle" in x]
            rgb_coords.append(centroids[keys[0]])
            #print(rgb_coords)
            success = True

    print('Robot Coordinates: ')
    print(robot_coords)

    robot_coords = np.array(robot_coords)[:, :2]
    rgb_coords = np.array(rgb_coords)[:, :2]

    print('RGB Coordinates: ')
    print(rgb_coords)
    poly = PolynomialFeatures(2)
    temp = poly.fit_transform(rgb_coords)
    matrix = compute_robot_transformation_matrix(np.array(temp), np.array(robot_coords))
    print('RGB to Robot Coordinates Transformation Matrix: ')
    print(repr(matrix))
    residuals = rgb_to_robot_coords(np.array(rgb_coords), matrix) - np.array(robot_coords)
    residuals = [np.linalg.norm(i) for i in residuals]
    print('Residuals: ')
    print(residuals)
