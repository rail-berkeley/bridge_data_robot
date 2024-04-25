#! /usr/bin/python3

import numpy as np
from threading import Lock
import logging
import os
import time

import rospy
from sensor_msgs.msg import JointState

from interbotix_xs_modules.arm import InterbotixArmXSInterface
from interbotix_xs_modules.core import InterbotixRobotXSCore
from interbotix_xs_modules.gripper import InterbotixGripperXSInterface

try:
    # older version of interbotix sdk
    from interbotix_xs_sdk.msg import JointGroupCommand
except:
    # newer version of interbotix sdk
    from interbotix_xs_msgs.msg import JointGroupCommand
    from interbotix_xs_msgs.srv import Reboot

from widowx_envs.utils.exceptions import Environment_Exception
import modern_robotics.core as mr
import widowx_envs.utils.transformation_utils as tr

from widowx_controller.custom_gripper_controller import GripperController
from widowx_controller.controller_base import RobotControllerBase

##############################################################################

# OUTLINE:
    # This is a low-level script intended to achieve a stable, compliant system with virtual Cartesian springs.
    # Most of the dynamic modeling is from Modern Robotics' implementations.
    # Transformation of actual desired trajectories/motion is trivial, so I will implement it elsewhere.

motor_constant = 9.999 # Each motor has its own motor-torque constant. It's relatively linear.

# ESTIMATE ACCELERATION: Because JointStates.msg only provides pos, vel, effort, we will have to estimate acceleration.

# velocity = self.JointState.velocity

num_joints = 5 # This actually needs to be read from the robot config file/robot info topic, based on the "arm" group.

velocity_samples = []

def estimate_acceleration(velocity, num_joints = num_joints, time = 0.01):    # A callback function that stores the last 3 velocity values

    global velocity_samples
    if len(velocity_samples) < 3:
        velocity_samples.append(velocity)
        return None
    else:
        velocity_samples.pop(0)
        velocity_samples.append(velocity)

        accelerations = []

        for joint_index in range(num_joints):
            acceleration = (velocity_samples[0][joint_index] - 2 * velocity_samples[1][joint_index] + velocity_samples[2][joint_index]) / time**2
            # Central difference approximation, selected because it's potentially more accurate than just (v1 - v0) / t
            accelerations.append(acceleration)    
        
        return accelerations

def balance_friction_torques(Kf, velocity, motor_constant):
    # FRICTION COMPENSATION: Experimental.
        # The theory goes, if we represent static motor friction as a torque, we can counteract this by applying a "helper" torque in the direction of the velocity to balance it.
        # This term is added to the final effort command 
    compensation_torque = Kf * velocity
    compensation_effort = compensation_torque / motor_constant
    return compensation_effort

grav = np.array([0, 0, 9.81])

def balance_gravity_torques(Slist, thetalist):
    # without a dynamic model of the robot, this will be hard to determine empirically.
    # Jacobian of each link * normed gravity vector in Cartesian space
    # TODO
    return 0

# Desired pose --> IKinSpace, publish thetalist (if none specified, then thetalist = current joint_state)
# --> subscribe to thetalist, find_arm + find_friction, gravity --> return torques

def find_err_pos(T, Slist, thetalist):
    eomg = eomg
    ev = ev
    M = M
    des_pos = mr.IKinSpace(Slist, M, T, thetalist, eomg, ev)
    err_pos = des_pos - thetalist
    return err_pos

def find_arm_torques(Ki, Kd, Kp, err_pos, err_accel=None, err_vel=None):
    # BASE ARM CONTROL LAW: Mostly just PID. Run at the desired control loop rate.
        # (Requires that all joints are initialized to take EFFORT commands. mode_configs must be in the launch file.)
        # Note: For impedance control (when encountering compliant objects):
        # -Ki (the inertial coeff.) should generally be raised. 
        # -Kd adjusts transience (settling time). Will cause instability if set too high.
        # -Kp (proportional gain); turn up for stiffness and more responsive acceleration. * Adjust carefully; use the minimum required for adequate motion.
        # All "err" values are matrices representing each joint in order

    torques = Ki(err_accel) + Kd(err_vel) + Kp(err_pos) # TODO: fix for vector multiplication
    return torques

def find_gripper_torque(Kp, Kd, err_gripper_vel, err_gripper_pos):
    # BASE GRIPPER CONTROL LAW: PD control law. Run at the desired control loop rate with the arm.
        # Ki is negligible, so it's being omitted.
    torques = Kd(err_gripper_vel) + Kp(err_gripper_pos)
    return torques

def publish_efforts(self, efforts):
        self.joint_commands = list(efforts)
        joint_commands = JointGroupCommand(self.group_name, self.joint_commands)
        self.core.pub_group.publish(joint_commands)

def publish_positions(self, positions):
        self.joint_commands = list(positions)
        joint_commands = JointGroupCommand(self.group_name, self.joint_commands)
        self.core.pub_group.publish(joint_commands)
        self.T_sb = FKinSpace(self.robot_des.M, self.robot_des.Slist, self.joint_commands)

# =====================================================================

class WidowXController(InterbotixArmXSInterface): # Needs overhauling, in progress
    def __init__(self, robot_name, print_debug, gripper_params,
                 enable_rotation='6dof',
                 gripper_attached='custom',
                 normal_base_angle=0,
                 Slist, M):
        
        self.Slist = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, -0.11065, 0.0, 0.0],
                                [0.0, 1.0, 0.0, -0.36065, 0.0, 0.04975],
                                [1.0, 0.0, 0.0, 0.0, 0.36065, 0.0],
                                [0.0, 1.0, 0.0, -0.36065, 0.0, 0.29975],
                                [1.0, 0.0, 0.0, 0.0, 0.36065, 0.0]])
        self.M = np.array([[1.0, 0.0, 0.0, 0.458325],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.36065],
                            [0.0, 0.0, 0.0, 1.0]])

    
        print('waiting for widowx_controller to be set up...')
        self.bot = InterbotixManipulatorXS(robot_model=robot_name)
        # TODO: Why was this call necessary in the visual_foresight? With the new SDK it messes the motor parameters
        # self.bot.dxl.robot_set_operating_modes("group", "arm", "position", profile_type="velocity", profile_velocity=131, profile_acceleration=15)

        if gripper_params is None:
            gripper_params = {}

        self._robot_name = robot_name
        rospy.on_shutdown(self.clean_shutdown)

        logger = logging.getLogger('robot_logger')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_level = logging.WARN
        if print_debug:
            log_level = logging.DEBUG
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        self._init_gripper(gripper_attached, gripper_params)

        self._joint_lock = Lock()
        self._angles, self._velocities, self._effort = {}, {}, {}
        rospy.Subscriber(f"/{robot_name}/joint_states", JointState, self._joint_callback)
        time.sleep(1)
        self._n_errors = 0

        self._upper_joint_limits = np.array(self.bot.arm.group_info.joint_upper_limits)
        self._lower_joint_limits = np.array(self.bot.arm.group_info.joint_lower_limits)
        self._qn = self.bot.arm.group_info.num_joints

        self.joint_names = self.bot.arm.group_info.joint_names
        self.default_rot = np.dot(tr.eulerAnglesToRotationMatrix([0, 0, normal_base_angle]), DEFAULT_ROTATION)

        self.neutral_joint_angles = np.zeros(self._qn)
        self.neutral_joint_angles[0] = normal_base_angle
        self.neutral_joint_angles[-2] = np.pi / 2
        self.neutral_joint_angles = NEUTRAL_JOINT_STATE
        self.enable_rotation = enable_rotation

        # This is all provided in the URDF, motor config, and documentation. This should be in a config file

    def reboot_motor(self, joint_name: str):
        """Experimental function to reboot the motor
        Supported joint names:
            - waist, shoulder, elbow, forearm_roll,
            - wrist_angle, wrist_rotate, gripper, left_finger, right_finger
        """
        rospy.wait_for_service('/wx250s/reboot_motors')
        try:
            reboot_motors = rospy.ServiceProxy('/wx250s/reboot_motors', Reboot)
            response = reboot_motors(cmd_type='single', name=joint_name,
                                     enable=True, smart_reboot=True)
            return response
        except rospy.ServiceException as e:
            print("Service call failed:", e)

    def clean_shutdown(self):
        pid = os.getpid()
        logging.getLogger('robot_logger').info('Exiting example w/ pid: {}'.format(pid))
        logging.shutdown()
        os.kill(pid, 9)

    def _joint_callback(self, msg):
        with self._joint_lock:
            for name, position, velocity, effort  in zip(msg.name, msg.position, msg.velocity, msg.effort):
                self._angles[name] = position
                self._velocities[name] = velocity
                self._effort[name] = effort

    def move_to_pose(self, target_xyz, target_zangle, duration=1.5):
        return None
        
    def set_moving_time(self, moving_time): # What does this do? Velocity for blocking control?
        return None
    
    def set_joint_angles(self, target_positions, duration=4):

    def check_motor_status_and_reboot(self):
        # print("checking motor status")
        status_codes = self.bot.dxl.robot_get_motor_registers("group", "all", "Hardware_Error_Status")
        # print(status_codes.values)
        # import ipdb; ipdb.set_trace()
        if len(status_codes.values) < 7:
            print("Some motor went wrong!")
            self.bot.dxl.robot_reboot_motors("group", "all", enable=True, smart_reboot=True)
            print("robot rebooted")
            self.move_to_neutral()
            raise Environment_Exception

    def move_to_neutral(self, duration=4):
        print('moving to neutral..')
        try:
            self.bot.arm.publish_positions(self.neutral_joint_angles, moving_time=duration)
            # print("Error in neutral position", np.linalg.norm(self.neutral_joint_angles - self.get_joint_angles()))
            # import ipdb; ipdb.set_trace()
            if np.linalg.norm(self.neutral_joint_angles - self.get_joint_angles()) > 0.1:
                print("moving to neutral failed!")
                self.check_motor_status_and_reboot()
        except rospy.service.ServiceException:
            print('stuck during reset')
            import pdb; pdb.set_trace()

    
    # def get_joint_angles(self):
    #     '''
    #     Returns current joint angles
    #     '''
    #     with self._joint_lock:
    #         try:
    #             return np.array([self._angles[k] for k in self.joint_names])
    #         except KeyError:
    #             return None

    # def get_joint_effort(self):
    #     '''
    #     Returns current joint angles
    #     '''
    #     with self._joint_lock:
    #         try:
    #             return np.array([self._effort[k] for k in self.joint_names])
    #         except KeyError:
    #             return None

    # def get_joint_angles_velocity(self):
    #     '''
    #     Returns velocities for joints
    #     '''
    #     with self._joint_lock:
    #         try:
    #             return np.array([self._velocities[k] for k in self.joint_names])
    #         except KeyError:
    #             return None

    def get_cartesian_pose(self, matrix=False):
        #FkinSpace
        return None

    def _init_gripper(self, gripper_attached, gripper_params):
        if gripper_attached == 'custom':
            self._gripper = GripperController(robot_name=self.bot.dxl.robot_name, des_pos_max=gripper_params.des_pos_max, des_pos_min=gripper_params.des_pos_min)
            self.custom_gripper_controller = True
        elif gripper_attached == 'custom_narrow':
            self._gripper = GripperController(robot_name=self.bot.dxl.robot_name, upper_limit=0.022, des_pos_max=gripper_params.des_pos_max, des_pos_min=gripper_params.des_pos_min)
            self.custom_gripper_controller = True
        elif gripper_attached == 'default':
            self.custom_gripper_controller = False
        else:
            raise ValueError("gripper_attached value has to be either 'custom', 'custom_narrow' or 'default'")

    def get_gripper_desired_position(self):
        # Previously determined by position commands; rewrite to get an err_gripper_pos for PD loop

    # def set_continuous_gripper_position(self, target):
        # Publish a gripper_command topic instead of doing this

    # def get_gripper_position(self):
        # Subscribe to gripper_state instead of doing this

    def open_gripper(self, wait=False):
        # Publish gripper_max_pos to gripper_command 

    def close_gripper(self, wait=False):
        # Publish gripper_min_pos to gripper command