#!/usr/bin/env python3

import cv2
from pydantic import BaseModel

class IMTopic(BaseModel):
    """
    Configuration for an image topic. User can also use json to create this object:
        obj = IMTopic.model_validate_json(json_data)
    """
    name: str
    width: int = 640
    height: int = 480
    top: int = 0
    bot: int = 0
    right: int = 0
    left: int = 0
    dtype: str = "bgr8"
    flip: bool = False
    info_name: str = None

    def __init__(self, name, **data):
        """For backwards compatibility, we allow user to pass in a string for name."""
        super().__init__(name=name, **data)

    def process_image(self, img):
        # Check for overcrop conditions
        assert self.bot + self.top <= img.shape[0], "Overcrop! bot + top crop >= image height!"
        assert self.right + self.left <= img.shape[1], "Overcrop! right + left crop >= image width!"

        # If bot or right is negative, set to value that crops the entire image
        bot = self.bot if self.bot > 0 else -(img.shape[0] + 10)
        right = self.right if self.right > 0 else -(img.shape[1] + 10)

        # Crop image
        img = img[self.top:-bot, self.left:-right]

        # Flip image if necessary
        if self.flip:
            img = img[::-1, ::-1]

        # Resize image if necessary
        if (self.height, self.width) != img.shape[:2]:
            return cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return img
