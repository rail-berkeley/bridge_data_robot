from widowx_envs.widowx_env import BridgeDataRailRLPrivateWidowX
from scripts.scripted_pickplace import PickPlacePolicy
from object_detector_ViLD import ObjectDetectorDL
from params import *
from widowx_envs.utils.grasp_utils import rgb_to_robot_coords
from widowx_envs.utils.exceptions import Environment_Exception
import random
import numpy as np
import os
import argparse
import datetime
from PIL import Image
import logging
import pickle
import time
import imp
from widowx_envs.trajectory_collector import TrajectoryCollector
import json

STEP_DURATION = 0.2
INITIAL_GRIPPER_CLOSE_PROB = 0.5

logging.getLogger('robot_logger').setLevel(logging.WARN)

def log_floor_height(save_dir, conf):
    hyperparams = imp.load_source('hyperparams', conf).config
    hyperparams['log_dir'] = "."
    meta_data_dict = json.load(open(hyperparams['collection_metadata'], 'r'))
    meta_data_dict['date_time'] = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    s = TrajectoryCollector(hyperparams)
    print('#################################################')
    print('#################################################')
    print("Move the gripper all the way to the lowest point of the workspace and end the trajectory.")
    _, obs_dict, _ = s.agent.sample(s.policies[0], 0)
    floor_height = np.min(obs_dict['full_state'][:, 2])
    meta_data_dict['floor_height'] = floor_height
    with open(os.path.join(save_dir, "collection_metadata.json"), 'w') as outfile:
        json.dump(meta_data_dict, outfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('teleop_conf', type=str, help='Path to teleop conf.')
    parser.add_argument("-o", "--object")
    parser.add_argument("-r", "--random-object-selection", default=True)
    parser.add_argument("-d", "--data-save-directory", type=str,
                        default="/home/robonet/trainingdata/scripted_demos/raw_data")
    parser.add_argument("-i", "--image_save_directory", type=str, default='')
    parser.add_argument("--detector", choices=('kmeans', 'dl', 'manual'), default='dl')
    parser.add_argument("--tsteps", type=int, default=50)
    parser.add_argument("--num-trajectories", type=int, default=50000)
    parser.add_argument("--action-noise", type=float, default=0.005)

    args = parser.parse_args()

    args.data_save_directory = os.path.expanduser(args.data_save_directory)
    directory_name = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    save_dir = os.path.join(args.data_save_directory, directory_name)
    os.makedirs(save_dir)
    log_floor_height(save_dir, args.teleop_conf)

    env_params = { 
        'fix_zangle': 0.1,
        'move_duration': 0.2,
        'adaptive_wait': True,
        'move_to_rand_start_freq': 1,
        'override_workspace_boundaries': WORKSPACE_BOUNDARIES,
        'action_clipping': 'xyz',
        'catch_environment_except': False,
        'randomize_initpos': 'restricted_space',
        # 'start_state': np.append(STARTPOS, [0, 0])
    }
    env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=128)

    #if args.detector == 'kmeans':
    #    object_detector = ObjectDetectorKmeans(env, args.image_save_directory, image_xyz=env.startpos)
    #    object_detector.try_make_background_image()
    #    transmatrix = KMEANS_RGB_TO_ROBOT_TRANSMATRIX
    #elif args.detector == 'dl':
    object_detector = ObjectDetectorDL(env=env, save_dir=args.image_save_directory, skip_move_to_neutral=True)
    transmatrix = DL_RGB_TO_ROBOT_TRANSMATRIX # regenerate using calibrate camera?? 
    #elif args.detector == 'manual':
    #    object_detector = ObjectDetectorManual(env=env, image_xyz=env.startpos)
    #    transmatrix = MANUAL_RGB_TO_ROBOT_TRANSMATRIX
    #    raise NotImplementedError

    scripted_policy = PickPlacePolicy(env)

    consecutive_detection_failures = 0

    for i in range(args.num_trajectories):
        if args.detector  == 'manual':
            input("Press Enter for Next Trajectory")
        print('traj #{}'.format(i))
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
            pick_point = centroids[grasp_object_name]
            print('PICKING UP ' + grasp_object_name)
        elif args.random_object_selection:
            # pick random object to pick and place
            grasp_object_name = 'random_object'
            centroids = object_detector.get_all_detected_centers(transform=True)
            centroids = np.array(centroids)
            # filter out unreasonable centroid that is out of workspace
            if len(centroids) == 0:
                print("detection fails")
                pick_point = np.random.uniform(WORKSPACE_BOUNDARIES[0][:2], WORKSPACE_BOUNDARIES[1][:2])
            else:
                centroids = centroids[(centroids[:, 0] > WORKSPACE_BOUNDARIES[0][0]) & 
                                    (centroids[:, 0] < WORKSPACE_BOUNDARIES[1][0]) &
                                    (centroids[:, 1] > WORKSPACE_BOUNDARIES[0][1]) & 
                                    (centroids[:, 1] < WORKSPACE_BOUNDARIES[1][1])]
                if len(centroids) == 0:
                    print("all detected objects are out of workspace")
                    consecutive_detection_failures += 1
                    if consecutive_detection_failures > 3:
                        print("process killed because all objects are out of workspace")
                        env.move_to_neutral()
                        env._controller.bot.arm.go_to_sleep_pose()
                        break
                    pick_point = np.random.uniform(WORKSPACE_BOUNDARIES[0][:2], WORKSPACE_BOUNDARIES[1][:2])
                else:
                    consecutive_detection_failures = 0
                    pick_point = random.choice(centroids)
        else:
            raise ValueError("must specify object name or set random object selection to True")
        drop_point = np.random.uniform(RESTRICTED_DROP_BOUNDARIES[0][:2], RESTRICTED_DROP_BOUNDARIES[1][:2])
        drop_point_z = np.random.uniform(DROP_POINT_Z_RANGE[0], DROP_POINT_Z_RANGE[1])
        pick_point_z = np.random.uniform(PICK_POINT_Z_RANGE[0], PICK_POINT_Z_RANGE[1])
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
                
                # save image for gif
                # image_obs = obs['image'][0]
                # current_image_formatted = np.transpose(np.uint8(np.reshape(image_obs * 255, (3, 128, 128))), (1, 2, 0))
                # images.append(Image.fromarray(current_image_formatted))

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
            
            # save gif
            # gif_name = os.path.join(current_save_dir, 'traj.gif')
            # images[0].save(gif_name,
            #                 format='GIF', append_images=images[1:],
            #                 save_all=True, duration=200, loop=0)