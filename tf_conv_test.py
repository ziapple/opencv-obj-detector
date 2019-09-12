import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# 展示numpy格式图片
def show_from_cv(img, title=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# 展示tensor格式图片
def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# tensor转化为numpy
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


def normalize(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))


# 输入图片
input_data = cv2.imread('test.jpg').astype(np.float32)
img_height, img_width, _ = input_data.shape
img_input = np.zeros((1, img_height, img_width, 3))
img_input[0] = input_data
img_input = img_input.astype(np.float32)
print(img_input.shape)
x = tf.Variable(img_input, dtype=np.float32)
y1 = tf.Variable(np.random.rand(3, 3, 3, 1), dtype=np.float32)
y2 = tf.Variable(np.random.rand(3, 3, 1, 1), dtype=np.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 模拟卷积层1
    x1 = tf.nn.conv2d(x, y1, strides=[1, 1, 1, 1], padding='SAME')
    # 模拟激活函数
    z = tf.nn.relu(x1)
    # 模拟dropout
    z = tf.nn.dropout(z, 0.8)
    # z = tf.nn.conv2d(x1, y2, strides=[1, 1, 1, 1], padding='SAME')
    r = sess.run(z)
    print(r)
    r = r[0].flatten()/np.max(r[0])
    r = r.reshape(img_height, img_width)
    plt.figure()
    plt.imshow(r)
    plt.pause(5)
