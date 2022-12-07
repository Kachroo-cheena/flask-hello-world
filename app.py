from flask import Flask, jsonify, request, render_template, redirect
# Importing all necessary libraries
import cv2
import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import pandas as pd
import numpy as np
import sys
from scipy.spatial import distance
import json
import pickle
import os


metric = 'cosine'

model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

IMAGE_SHAPE = (224, 224)

layer = hub.KerasLayer(model_url)
model = tf.keras.Sequential([layer])

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = os.getcwd()+"/uploads"


def get_video_metadata(image_to_find):
    threshold = 0.5
    image_to_find = extract(image_to_find)
    filehandler = open('vectors.obj', 'rb')
    temp = pickle.load(filehandler)
    # prin?t(temp)
    return (find_minimum_distance_with_image(temp, image_to_find, threshold))


def find_minimum_distance_with_image(video_dic, image_to_find, threshold):
    dist_dic = {}
    for vd in video_dic:
        video_1 = video_dic[vd]
        temp_video_1 = []
        for i in video_1:
            temp_video_1.append(distance.cdist(
                [i], [image_to_find], metric)[0][0])
        video_1_min = min(temp_video_1)
        dist_dic[vd] = video_1_min
    for i in dist_dic:
        if dist_dic[i] <= threshold:
            return i


def convert_to_vectors(filename):
    filename = filename.split(".")[0]
    video_1_temp = os.listdir('data/{}/'.format(filename))
    video_1 = []
    for i in video_1_temp:
        video_1.append(extract('data/{}/'.format(filename)+i))
    return video_1


def extract(file):
    file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
    # display(file)
    file = np.stack((file,)*3, axis=-1)
    file = np.array(file)/255.0
    embedding = model.predict(file[np.newaxis, ...])
    vgg16_feature_np = np.array(embedding)
    flattended_feature = vgg16_feature_np.flatten()
    return flattended_feature


def convert_to_frames(filename):
    os.system("ffmpeg -i {0} -filter:v fps=10 {1}".format(filename,
              filename.split('.')[0]+"_changed.mp4"))
    filename = filename.split(".")[0]
    cam = cv2.VideoCapture("{}_changed.mp4".format(filename))
    cam.set(cv2.CAP_PROP_FPS, 10)
    try:

        # creating a folder named data
        if not os.path.exists('data/{}'.format(filename)):
            os.makedirs('data/{}'.format(filename))

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0

    while (True):

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = './data/{}/frame'.format(filename) + \
                str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


@app.route('/', methods=['POST'])
def hello_world():
    if request.method == "POST":
        print("JDSHGJHGJHDSGJDHS")
        if request.files:
            print("dfjhbjhbj")
            image = request.files["image"]
            print(image)
            image.save(os.path.join(
                app.config["IMAGE_UPLOADS"], image.filename))
            temp = (get_video_metadata(os.path.join(
                app.config["IMAGE_UPLOADS"], image.filename)))
            if temp == None:
                return "No Videos Found"
            else:
                return temp
