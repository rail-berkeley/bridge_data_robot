#!/usr/bin/env python
import argparse
import os
import time
import yaml


def get_dev(output_string, usb_id):
    import re
    lines = output_string.decode().split('\n')
    for i, line in enumerate(lines):
        if usb_id in line:
            return re.search('video(\d+)', lines[i + 1]).group(1)
    raise ValueError('usb_id {} not found!'.format(usb_id))


""" example connect_chart file
connector_chart = {
    "cam0": 'usb-0000:00:14.0-9.3',
    "cam1": 'usb-0000:00:14.0-9.4',
    "cam2": 'usb-0000:00:14.0-10.3',
    "cam3": 'usb-0000:00:14.0-10.4',
    "cam4": 'usb-0000:08:00.0-2',
}
"""
""" results in
HD Web Camera: HD Web Camera (usb-0000:00:14.0-10.4):
	/dev/video4
	/dev/video5

HD Web Camera: HD Web Camera (usb-0000:00:14.0-14):
	/dev/video0
	/dev/video1

HD Web Camera: HD Web Camera (usb-0000:00:14.0-9.4):
	/dev/video2
	/dev/video3

HD Web Camera: HD Web Camera (usb-0000:08:00.0-1):
	/dev/video8
	/dev/video9

HD Web Camera: HD Web Camera (usb-0000:08:00.0-2):
	/dev/video6
	/dev/video7
"""


def main(args):
    assert len(args.cam_order) == len(args.topic_names), "Number of providers should equal number of topics"

    base_call = "roslaunch foresight_rospkg camera.launch video_stream_provider:={} camera_name:={} visualize:={} fps:=20 buffer_queue_size:=2 &"
    visualize_str = "false"
    if args.visualize:
        visualize_str = "true"

    connector_chart = yaml.load(open(args.connector_chart_path, 'r'), Loader=yaml.CLoader)

    if args.use_connector_chart or args.chart_names[0] != '':
        if args.chart_names[0] != '':
            camera_connector_dict = {}
            for topic in args.chart_names:
                if topic not in connector_chart:
                    raise ValueError("topic {} not in connector chart!".format(topic))
                camera_connector_dict[topic] = connector_chart[topic]
        else:
            camera_connector_dict = connector_chart

        import subprocess
        res = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE)
        output_string = res.stdout

        for topic_name, usb_id in camera_connector_dict.items():
            dev_number = get_dev(output_string, usb_id)
            provider = '/dev/video{}'.format(dev_number)
            os.system(base_call.format(provider, topic_name, visualize_str))
            print(base_call.format(provider, topic_name, visualize_str))
            time.sleep(5)
        return

    if args.ncam == -1:
        for provider, cam_name in zip(args.cam_order, args.topic_names):
            provider = '/dev/video{}'.format(provider)
            os.system(base_call.format(provider, cam_name, visualize_str))
            time.sleep(2)
    else:
        for i in range(0, args.ncam):
            provider = '/dev/video{}'.format(i*2)
            cam_name = 'cam{}'.format(i*2)
            print(base_call.format(provider, cam_name, visualize_str))
            os.system(base_call.format(provider, cam_name, visualize_str))
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="orders and launches cameras")
    parser.add_argument("--ncam", type=int, default=-1,
                        help="")
    parser.add_argument("--use_connector_chart", action='store_true',
                        help="whether to use fixed usage of usb ports, based on output ports coming from v4l2-ctl --list-devices")
    parser.add_argument("--chart_names", type=str, nargs='+',
                        default=[''],
                        help="chart topic names")
    parser.add_argument("--cam_order", type=int, nargs='+', default=[2, 0],
                        help="list of camera video stream providers")
    parser.add_argument("--topic_names", type=str, nargs='+',
                        default=['cam0', 'cam1'],
                        help="list of camera topic names")
    parser.add_argument('--visualize', action='store_true', default=True, help="if flag supplied image_view will show")
    parser.add_argument('--connector_chart_path', type='str', help="path to the connector chart yaml file - an example file is provided in the script")
    args = parser.parse_args()
    main(args)


"""
output of $v4l2-ctl --list-devices

HP TrueVision HD Camera: TAKMLY (usb-0000:00:14.0-10.3):
	/dev/video0
	/dev/video1

HD Web Camera: HD Web Camera (usb-0000:00:14.0-10.4):
	/dev/video2
	/dev/video3

HD Web Camera: HD Web Camera (usb-0000:00:14.0-5):
	/dev/video10
	/dev/video11

HD Web Camera: HD Web Camera (usb-0000:00:14.0-6):
	/dev/video6
	/dev/video7

HD Web Camera: HD Web Camera (usb-0000:05:00.0-1):
	/dev/video4
	/dev/video5

HD Web Camera: HD Web Camera (usb-0000:05:00.0-2):
	/dev/video8
	/dev/video9

"""