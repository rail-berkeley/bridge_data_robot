import numpy as np

### run_scripted_pickplace_callibration
CONTAINER_POSITION = np.array([-0.12, -0.28])
X_MARGIN = 0.03
X_THRESH = -0.05
Y_THRESH = np.inf
PICK_POINTS = [[0.19, X_MARGIN + X_THRESH, 0.032],
                [0.36, 0.12, 0.032]]
WORKSPACE_BOUNDARIES = [[0.19, -0.16, 0.029, -0.75, 0],
                        [0.36, 0.12, 0.17, 0.75, 0]]

TIC_TAC_TOE_WORKSPACE_BOUNDARIES = [[0.19, -0.16, 0.029, -0.75, 0],
                        [0.60, 0.12, 0.17, 0.75, 0]]

PERIMETER_SWEEP_WORKSPACE_BOUNDARIES = [[0.16, -0.17, 0.029, -1.57, 0],
                        				[0.38, 0.12, 0.17,  1.57, 0]]
STARTPOS = [0.0, -0.25, 0.15]
PICK_POINT_Z = 0.03
DROP_POINT_Z = 0.05
FAR_REACH_Z_THRESH = np.inf
LIFT_AFTER_GRASP = True
PICK_Z_MULTIPLIER = 1.0
WRIST_ANGLE_INDEX = 4
WRIST_ANGLE_RANGE = (-0.021, 0.021)
WRIST_TARGET_THRESH = 0.2
PERIMETER_SWEEP_FREQ = 2
OBJECT_OFFSETS = {
    'tray': {
        'Frog': [0, 0.03, 0],
        'Elephant': [0, 0.03, 0],
        'BlueRat': [-0.03, 0.015, 0],
        'Star': [-0.03, 0.015, 0],
    },
    'container': {
        'Frog': [0, 0, 0],
        'Elephant': [0, 0, 0],
        'BlueRat': [0, 0, 0],
        'Star': [0.01, 0.015, 0],
    }
}

CONTAINER_HEIGHT = {
    'LowPot': 0.0325
}

CONTAINER_PROXIMITY_HIGHER_GRASP = {
    'LowPot': {'proximity': 0.1, 'offset': -0.02}
}

VILD_RGB_TO_ROBOT_TRANSMATRIX = [[1.86458763e-01, -2.65330020e-01], 
    [-2.53418603e-04, 1.18623271e-03], 
    [1.1491619e-03, 1.50029106e-04], 
    [9.14574675e-07, 2.45525262e-08], 
    [-1.94619032e-06, -4.27330438e-07], 
    [9.51640060e-07, -3.52438397e-07]]

### calibrate_camera.py, object_detector_kmeans.py
KMEANS_RGB_TO_ROBOT_TRANSMATRIX = [[ 7.59416873e-01, -3.89187872e-01],
    [-1.36140236e-02,  3.30508616e-03],
    [-1.61799085e-02, -5.98485796e-03],
    [ 5.71792010e-05,  5.14624241e-05],
    [ 1.45330565e-04, -1.25233647e-05],
    [ 8.86058630e-06,  5.11294279e-05]]
DL_RGB_TO_ROBOT_TRANSMATRIX = [[    0.38081,    -0.15001],
        [  -0.027763,     0.16143],
        [   -0.08254,    0.036088],
        [ -0.0037169,   -0.011731],
        [   0.015174,  -0.0015721],
        [  -0.010473,   -0.005279]]
OVERRIDE_GOALS = None

### object_detector_dl.py
DL_OBJECT_DETECTOR_DIRECTORY = '/home/robonetv2/widowx_envs/widowx_envs/utils/object_detection/ultralytics_yolov5_master'
DL_OBJECT_DETECTOR_CHECKPOINT = '/home/robonetv2/widowx_envs/widowx_envs/utils/object_detection/best.pt'
OBJECT_DETECTOR_CLASSES = ['Lego', 'Bird', 'Chicken', 'Rabbit', 'Bear', 'Puppet', 'Elephant', 'Frog', 'Monkey', 'Fourpetal',
  'Trapezoid', 'Panda', 'Mouse', '2x2lego', 'ShortCylinder', 'BlueRat', 'Star', 'Rhombus', 'PolarBear',
  'LowPotLid', 'Battery', 'Heart', 'SmallCup', 'Pickle', 'Dog', 'SmallOrange', 'Apool', 'Mentos', 'Turtle',
  'SoySauce']

DL_CONTAINER_DETECTOR_CHECKPOINT = '/home/robonetv2/widowx_envs/widowx_envs/utils/object_detection/best_reward.pt'
CONTAINER_DETECTOR_CLASSES = ['Teddy Bear', 'Puppet', 'Bird', 'Lego', 'Tray', 'LowPot', 'PrintedDrawer', 'WoodenDrawer']
CONTAINER_CLASS = 'Tray'

###object_detector_manual.py
IMAGE_PATH = "object_detector_manual_image.png"
CENTROID_PATH = "object_detector_manual_centroid.pkl"

# Post rotation coordinates:
POINT_EXTRACTION_ROW_CUTOFF = 0.45

### For run_scripted_grasping
TWO_POSITIONS = [np.asarray([0.20, 0, 0.032]), np.asarray([0.30, 0, 0.032])]
POSITION = np.array([-0.1711, -0.35, 0.032])

#widowx/src/custom_gripper_controller
MAX_GRIPPER_PWM = 350

try:
    from widowx_envs.utils.params_private import *
    test = 1
except ImportError:
    print("""
    Consider copying params.py to params_private.py, i.e.

    cp widowx_envs/utils/params.py widowx_envs/utils/params_private.py
    """)
