#! /usr/bin/env python3

import cv2
import rospy
import threading
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import copy


class Streamer:
    def __init__(self):
        self.parse_rosparam()
        self.full_resource_path = "/dev/video" + str(self._video_stream_provider)
        success = self.setup_capture_device()
        if not success:
            return
        self.publisher = rospy.Publisher(self._topic_name + '/image_raw', Image, queue_size=1)
        self._buffer = []
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self.start_capture()
        self.start_publishing()

    def setup_capture_device(self):
        success = False
        if not os.path.exists(self.full_resource_path):
            rospy.logerr("Device '%s' does not exist.", self.full_resource_path)
            return success
        rospy.loginfo("Trying to open resource: '%s' for topic '%s'", self.full_resource_path, self._topic_name)
        self.cap = cv2.VideoCapture(self.full_resource_path)
        if not self.cap.isOpened():
            rospy.logerr(f"Error opening resource: {self.full_resource_path}")
            rospy.loginfo("opencv VideoCapture can't open it.")
            rospy.loginfo("The device '%s' is possibly in use. You could try reconnecting the camera.", self.full_resource_path)
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
        self._fps = self.get_param("~fps")
        self._frame_id = self.get_param("~frame_id")
        self._retry_on_fail = self.get_param("~retry_on_fail")
        self._buffer_queue_size = self.get_param("~buffer_queue_size")
        self._topic_name = self.get_param("~camera_name")
        self._video_stream_provider = self.get_param("~video_stream_provider")

    def start_capture(self):
        self._capture_thread = threading.Thread(target=self.capture)
        self._capture_thread.start()

    def capture(self):
        # running at full speed the camera allows
        while not rospy.is_shutdown():
            rval, frame = self.cap.read()
            if not rval:
                rospy.logwarn(f"The frame has not been captured. You could try reconnecting the camera resource {self.full_resource_path}.")
                rospy.sleep(3)
                if self._retry_on_fail:
                    rospy.loginfo(f"Searching for the device {self.full_resource_path}...")
                    self.setup_capture_device(exit_on_error=False)
            else:
                reading = [frame, rospy.Time()]
                with self._lock:
                    while(len(self._buffer) > self._buffer_queue_size):
                        self._buffer.pop(0)
                    self._buffer.append(reading)

    def publish_image(self, reading):
        img = reading[0]
        time = reading[1]
        imgmsg = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        imgmsg.header.frame_id = self._frame_id
        imgmsg.header.stamp = time
        self.publisher.publish(imgmsg)

    def start_publishing(self):
        self._publishing_thread = threading.Thread(target=self.publishing)
        self._publishing_thread.start()

    def publishing(self):
        rate = rospy.Rate(self._fps)
        while not rospy.is_shutdown():
            reading = None
            with self._lock:
                if self._buffer:
                    reading = self._buffer[0]
                    self._buffer.pop(0)
            if reading is not None:
                self.publish_image(reading)
            rate.sleep()

def main():
    rospy.init_node('streamer', anonymous=True)
    streamer = Streamer()
    rospy.spin()


if __name__ == '__main__':
    main()