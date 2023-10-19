from widowx_envs.widowx.widowx_grasp_env import GraspWidowXEnv
from widowx_envs.policies.scripted_reach import ReachPolicy
from widowx_envs.utils.grasp_utils import (
    compute_robot_transformation_matrix,
    rgb_to_robot_coords,
    execute_reach
)
from widowx_envs.utils.params import WORKSPACE_BOUNDARIES
from widowx_envs.utils.object_detection import ObjectDetectorKmeans # might need to fix the path for this, and add ViLD too
#from widowx_envs.widowx.env_wrappers import NormalizedBoxEnv
from tqdm import tqdm
import numpy as np
import argparse
import time
from sklearn.preprocessing import PolynomialFeatures
import torch
from widowx_envs.utils.params import *

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
    x_lo, x_hi = WORKSPACE_BOUNDARIES[0][0] + eps, WORKSPACE_BOUNDARIES[1][0] - eps
    y_lo, y_hi = WORKSPACE_BOUNDARIES[0][1] + eps, WORKSPACE_BOUNDARIES[1][1] - eps
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
    parser.add_argument("-s", "--save_dir", type=str, default="")
    parser.add_argument("-d", "--detector", required=True, choices=('ViLD', 'kmeans', 'dl'), default='ViLD')
    parser.add_argument("-i", "--id", help='object class ID for the DL detector', required=False, type=str, default=None)
    parser.add_argument('--skip-move-to-neutral', action='store_true', default=False)
    args = parser.parse_args()
    
    set_gpu_mode(True)


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
    env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=128)

    if args.detector == 'ViLD':
        object_detector = ObjectDetectorViLD(env, args.save_dir) 
    if args.detector == 'kmeans':
        object_detector = ObjectDetectorKmeans(env, args.save_dir)
        object_detector.try_make_background_image()
    elif args.detector == 'dl':
        object_detector = ObjectDetectorDL(
            env=env, weights=DL_TARGET_OBJECT_DETECTOR_CHECKPOINT, save_dir=args.save_dir, skip_move_to_neutral=args.skip_move_to_neutral)
    else:
        raise NotImplementedError

    robot_coords = []
    rgb_coords = []

    reach_policy = ReachPolicy(env, reach_point=None)

    goals = generate_goals(env)
    object_class_id = args.id

    for j in tqdm(range(len(goals))):
        retry = 0
        success = False
        while not success:
            # import ipdb; ipdb.set_trace()
            env.reset()
            # obs = execute_reach(env, reach_policy, goals[j])
            env._controller.move_to_state(goals[j], 0, duration=3)
            obs = env._get_obs()
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
            img = from_numpy(img.transpose(2, 0, 1))
            img = img[None]
            centroids = object_detector.get_centroids(img)
            print('centroids', centroids)
            if object_class_id is None:
                assert len(centroids) == 1, "There should be only one object found. If the tray is found, specify '--id' script argument."
                object_class_id = list(centroids.keys())[0]
            if object_class_id not in centroids:
                print("Centroid not detected. Changing the goal position.")
                retry += 1
                goals[j] = generate_goals(env, retry=retry)[j]
                continue
            robot_coords.append(obs['ee_coord'])
            rgb_coords.append(centroids[object_class_id])
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
