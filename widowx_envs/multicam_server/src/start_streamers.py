#! /usr/bin/env python3

from streamer import Streamer
import os
import yaml
import re
import shutil
import subprocess
import rospy

##############################################################################

def get_param(parameter_name):
    if not rospy.has_param(parameter_name):
        rospy.logerr('Parameter %s not provided. Exiting.', parameter_name)
        exit(1)
    return rospy.get_param(parameter_name)


def load_connector_chart():
    config_path = get_param("~camera_connector_chart")
    if not os.path.exists(config_path):
        rospy.logerr(f"The usb connector chart in path {config_path} does not exist. You can use the example usb_connector_chart_example.yml as a template.")
        rospy.logerr("you can find the usb outlets of the webcams by running $v4l2-ctl --list-devices")
        exit(1)
    return yaml.load(open(config_path, 'r'), Loader=yaml.CLoader)


def get_dev(output_string, usb_id):
    lines = output_string.decode().split('\n')
    for i, line in enumerate(lines):
        if usb_id in line:
            return re.search('video(\d+)', lines[i + 1]).group(1)
    raise ValueError('usb_id {} not found!'.format(usb_id))


def reset_usb(reset_names):
    if shutil.which('usbreset') is None:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        res = subprocess.call(f'gcc {current_dir}/usbreset.c -o /usr/local/bin/usbreset')
        if not res == 0:
            rospy.logerr(f'usbreset install exit code: {res}')
            raise ValueError('could not install usbreset !')
    res = subprocess.run(['lsusb'], stdout=subprocess.PIPE)
    lines = res.stdout.decode().split('\n')
    for line in lines:
        for name in reset_names:
            if name in line:
                numbers = re.findall(r'(\d\d\d)', line)[:2]
                rospy.loginfo('resetting usb with lsusb line: {}'.format(line))
                cmd = 'sudo usbreset /dev/bus/usb/{}/{}'.format(*numbers)
                res = subprocess.call(cmd.split())
                if not res == 0:
                    rospy.logerr(f'exit code: {res}')
                    raise ValueError('could not reset !')


def process_camera_connector_chart():
    connector_chart_dict = load_connector_chart()

    res = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE)
    output_string = res.stdout

    # reset_names = set()
    # for key, value in connector_chart_dict.items():
    #     reset_names.add(value[1])
    # rospy.loginfo(f"reseting lsusb names {reset_names}")
    # reset_usb(reset_names)

    providers = []
    topic_names = []
    for topic_name, usb_id in connector_chart_dict.items():
        dev_number = get_dev(output_string, usb_id)
        providers.append(dev_number)
        topic_names.append(topic_name)
    return providers, topic_names


def populate_params():
    params = {}
    params['fps'] = get_param("~fps")
    params['frame_id'] = get_param("~frame_id")
    params['retry_on_fail'] = get_param("~retry_on_fail")
    params['buffer_queue_size'] = get_param("~buffer_queue_size")
    params['python_node'] = get_param("~python_node")
    return params

##############################################################################

def main():
    base_call = "roslaunch multicam_server streamer.launch"
    rospy.init_node('start_streamers', anonymous=True)  # node name is provided in the roslaunch file
    topic_names = []
    if get_param("~camera_connector_chart"):
        video_stream_providers, topic_names = process_camera_connector_chart()
    else:
        video_stream_provider = get_param("~video_stream_provider")
        parsed_video_stream_provider = eval(video_stream_provider)
        if isinstance(parsed_video_stream_provider, list):
            video_stream_providers = parsed_video_stream_provider
        elif isinstance(parsed_video_stream_provider, int):
            video_stream_providers = [parsed_video_stream_provider]
        else:
            rospy.logerr("Pass either list or an integer as video_stream_provider to video_stream_opencv node.")
            rospy.loginfo(f"Arguments provided: {video_stream_provider}")
        for i in range(len(video_stream_providers)):
            topic_names.append(f'camera{i}')

    processes = []
    for index, [video_stream_provider, topic_name] in enumerate(zip(video_stream_providers, topic_names)):
        full_params = {'video_stream_provider': video_stream_provider, 'camera_name': topic_name, 'node_name': f'streamer_{index}'}
        full_params.update(populate_params())
        appended_string = ''
        for key, val in full_params.items():
            appended_string += key + ':=' + str(val) + ' '
        proc = subprocess.Popen((base_call + ' ' + appended_string).split())
        processes.append(proc)
    while not rospy.is_shutdown():
        rospy.sleep(0.1)
    for proc in processes:
        proc.kill()
        proc.communicate()

if __name__ == '__main__':
    main()
