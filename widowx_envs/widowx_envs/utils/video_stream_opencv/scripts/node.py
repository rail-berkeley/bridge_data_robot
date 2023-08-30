#! /usr/bin/env python3
"""
@author: Jedrzej Orbik
"""

import cv2
import rospy
import threading
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Streamer:
    def __init__(self, resource, node_id=0):
        self.full_resource_path = "/dev/video" + str(resource)
        self.running = False
        self.parse_rosparam()
        success = self.setup_capture_device()
        if not success:
            return
        self.publisher = rospy.Publisher(self._camera_name + str(node_id) + '/image_raw', Image, queue_size=1)
        self.buffer = []
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self.start_capture()
        self.start_publishing()

    def __del__(self):
        if hasattr(self, '_capture_thread'):
            self._capture_thread.join()
        if hasattr(self, '_publishing_thread'):
            self._publishing_thread.join()

    def setup_capture_device(self):
        success = False
        if not os.path.exists(self.full_resource_path):
            rospy.logerr("Device %s does not exist.", self.full_resource_path)
            return success
        rospy.loginfo("Trying to open resource: %s", self.full_resource_path)
        self.cap = cv2.VideoCapture(self.full_resource_path)
        if not self.cap.isOpened():
            rospy.logerr(f"Error opening resource: {self.full_resource_path}")
            rospy.loginfo("opencv VideoCapture can't open it.")
            rospy.loginfo("The device %s is possibly in use. You could try reconnecting the camera.", self.full_resource_path)
        if self.cap.isOpened():
            rospy.loginfo(f"Correctly opened resource {self.full_resource_path}.")
            success = True
        return success

    @staticmethod
    def get_param(parameter_name):
        if not rospy.has_param(parameter_name):
            rospy.logerr('Parameter %s not provided. Exiting.', parameter_name)
            exit(1)
        return rospy.get_param(parameter_name)

    def parse_rosparam(self):
        self._camera_name = self.get_param("~camera_name")
        self._buffer_queue_size = self.get_param("~buffer_queue_size")
        self._fps = self.get_param("~fps")
        self._frame_id = self.get_param("~frame_id")
        self._retry_on_fail = self.get_param("~retry_on_fail")

    def start_capture(self):
        self._capture_thread = threading.Thread(target=self.capture)
        self._capture_thread.start()

    def capture(self):
        # running at full speed the camera allows
        while not rospy.is_shutdown():
            rval, frame = self.cap.read()
            self.running = rval
            if not rval:
                rospy.logwarn(f"The frame has not been captured. You could try reconnecting the camera resource {self.full_resource_path}.")
                rospy.sleep(3)
                if self._retry_on_fail:
                    rospy.loginfo(f"Searching for the device {self.full_resource_path}...")
                    self.setup_capture_device(exit_on_error=False)
            else:
                with self._lock:
                    while len(self.buffer) >= self._buffer_queue_size:
                        self.buffer.pop(0)
                    self.buffer.append(frame)

    def publish_image(self, image):
        imgmsg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        imgmsg.header.frame_id = self._frame_id
        self.publisher.publish(imgmsg)

    def start_publishing(self):
        self._publishing_thread = threading.Thread(target=self.publishing)
        self._publishing_thread.start()

    def publishing(self):
        rate = rospy.Rate(self._fps)
        while not rospy.is_shutdown():
            image = None
            if self.running:
                with self._lock:
                    if self.buffer:
                        image = self.buffer[-1].copy()
                if image is not None:
                    self.publish_image(image)
            rate.sleep()

def main():
    rospy.init_node('dummy_name', anonymous=True)
    video_stream_provider = Streamer.get_param("~video_stream_provider")
    parsed_video_stream_provider = eval(video_stream_provider)

    node_id = 0
    streamers = []
    if isinstance(parsed_video_stream_provider, list):
        for resource in parsed_video_stream_provider:
            streamers.append(Streamer(resource, node_id))
            node_id += 1
    elif isinstance(parsed_video_stream_provider, int):
        streamers.append(Streamer(parsed_video_stream_provider))
    else:
        rospy.logerr("Pass either list or an integer as video_stream_provider to video_stream_opencv node.")
        rospy.loginfo(f"Arguments provided: {video_stream_provider}")
        exit(1)
    running = True
    while running and not rospy.is_shutdown:
        running = False
        for streamer in streamers:
            if streamer.running:
                running = True
                break
        rospy.sleep(0.1)

if __name__ == '__main__':
    main()
