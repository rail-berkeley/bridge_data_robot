from widowx_envs.widowx_env import BridgeDataRailRLPrivateWidowX
from multicam_server.topic_utils import IMTopic
import numpy as np
import time

WORKSPACE_BOUNDS = np.array([[0.1, -0.15, -0.1, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]])
CAMERA_TOPICS = [IMTopic("/D435/color/image_raw")]
STEP_DURATION = 0.2
BLOCKING = True

env_params = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": WORKSPACE_BOUNDS,
        "action_clipping": "xyz",
        "catch_environment_except": False,
        "start_state": np.array([0.3, 0, 0.15, 0, 0, 0, 1]),
        "return_full_image": False,
        "camera_topics": CAMERA_TOPICS,
    }
env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=128)


# create policy class and query it for action 
action = np.array([0.01, 0, 0, 0, 0, 0, 1])

env.reset()
env.start()

last_tstep = time.time()
t = 0
while t < 1:
    if time.time() > last_tstep + STEP_DURATION or BLOCKING:
        last_tstep = time.time()
        
        

        env.step(action, last_tstep + STEP_DURATION, blocking=BLOCKING)
        t += 1