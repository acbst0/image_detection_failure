import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from PIL import Image

class take_model:
    def __init__(self, link, purpose, image):
        self.link = link
        self.purpose = purpose
        self.image = image
    
    def take(self):
        # Apply image detector on a single image.
        detecter = hub.load(self.link)
        num_detec = detecter(num_detections)
        return num_detec

image_path = "test.png"
image = Image.open(r"C:\Users\user1\Desktop\GIT-Commands-TR\WhatsApp Görsel 2023-12-19 saat 22.34.32_7b38e3b6.jpg")

image_np = np.array(image)
detection_model = take_model("https://www.kaggle.com/models/tensorflow/faster-rcnn-inception-resnet-v2/frameworks/TensorFlow2/variations/640x640/versions/1", "Image detect", image_np)
nums = detection_model.take()