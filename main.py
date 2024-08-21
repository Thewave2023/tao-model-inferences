import onnxruntime as ort
import numpy as np
import cv2

WIDTH = 1280
HEIGHT = 736


# 加载 ONNX 模型
session = ort.InferenceSession('./model_cut.onnx')

# 读取图像
image = cv2.imread('./000003.jpg')
original_image = image.copy()
image_height, image_width = image.shape[:2]

# 预处理图像
image = cv2.resize(image, (WIDTH, HEIGHT))
image = image.astype(np.float32)
image /= 255.0
image = np.transpose(image, [2, 0, 1])
image = np.expand_dims(image, axis=0)

# 推理
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: image})

print(outputs)
