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
        detector = hub.load(self.link)
        detector_output = detector({"image": self.image})  # Girdi olarak bir sözlük bekliyoruz
        return detector_output

image_path = "test.png"
image = Image.open(r"C:\Users\user1\Desktop\GIT-Commands-TR\WhatsApp Görsel 2023-12-19 saat 22.34.32_7b38e3b6.jpg")

image_np = np.array(image)
detection_model = take_model("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1", "Image detect", image_np)
nums = detection_model.take()
print(nums)
