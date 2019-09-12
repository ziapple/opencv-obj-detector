#!/usr/bin/python3
#!--*-- coding:utf-8 --*--
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


model_path = "ssd_mobilenet_v1_coco_2018_01_28"
frozen_pb_file = os.path.join(model_path, 'frozen_inference_graph.pb')

score_threshold = 0.3

img_file = 'test.jpg'

# Read the graph.
with tf.gfile.FastGFile(frozen_pb_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    img_cv2 = cv2.imread(img_file)
    img_height, img_width, _ = img_cv2.shape

    img_in = cv2.resize(img_cv2, (300, 300))
    img_in = img_in[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    outputs = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={
                       'image_tensor:0': img_in.reshape(1,
                                                        img_in.shape[0],
                                                        img_in.shape[1],
                                                        3)})

    # Visualize detected bounding boxes.
    num_detections = int(outputs[0][0])
    for i in range(num_detections):
        classId = int(outputs[3][0][i])
        score = float(outputs[1][0][i])
        bbox = [float(v) for v in outputs[2][0][i]]
        if score > score_threshold:
            x = bbox[1] * img_width
            y = bbox[0] * img_height
            right = bbox[3] * img_width
            bottom = bbox[2] * img_height
            cv2.rectangle(img_cv2,
                          (int(x), int(y)),
                          (int(right), int(bottom)),
                          (125, 255, 51),
                          thickness=2)

plt.figure(figsize=(10, 8))
plt.imshow(img_cv2[:, :, ::-1])
plt.title("TensorFlow MobileNetV2-SSD")
plt.axis("off")
plt.show()