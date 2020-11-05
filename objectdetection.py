# file used for query consumers
# commented codes can be used for report generation

import queue
import threading

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from classification import CarClassifier
from feature_extraction import ColorClassifier
from keras_yolo3.yolo import YOLO

yolo_obj = YOLO()
car_classifier = CarClassifier()

#queue for each query
q1 = queue.Queue()
q2 = queue.Queue()
q3 = queue.Queue()

#for acquiring results
results = []
# Q1_time=[]
# Q2_time=[]
# Q3_time=[]

#Yolo object detection
class Q1Consumer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.yolo_obj = YOLO()

    def run(self):
        global q1
        global q2
        global q3
        global results
        # global Q1_time
        # global Q2_time
        # global Q3_time
        yolo_obj = self.yolo_obj
        while True:
            data = q1.get()
            frame_num = data['frame_num']
            if frame_num == 'exit':
                q2.put({'frame_num': 'exit'})
                q3.put({'frame_num': 'exit'})
                break
            img = data['img']
            query = data['query']
            # start_time = time.time()
            cars = yolo_obj.detect_image(img)
            # exec_time = time.time() - start_time
            # Q1_time.append()
            result = {"frame_num": frame_num, "num_cars": len(cars)}
            print('Query 1:')
            print(result)
            results.append(result)
            if query > 1:
                self.send(cars, frame_num, img, query)

    def send(self, cars, frame_num, img, query):
        global q2
        global q3
        data = {'cars': cars, 'frame_num': frame_num,
                'query': query, 'img': img}
        q2.put(data)

# Crop and car model classification
class Q2Consumer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.car_classifier = CarClassifier()

    def run(self):
        global q2
        global results
        # global Q1_time
        # global Q2_time
        # global Q3_time
        while True:
            data = q2.get()
            frame_num = data['frame_num']
            if frame_num == 'exit':
                # q2.put({'frame_num':'exit'})
                break
            img = data['img']
            cars = data['cars']
            query = data['query']
            labels = []
            for car in cars:
                cropped = img.crop(
                    (car['left'], car['top'], car['right'], car['bottom']))
                cropped = np.array(cropped)
                predictions = self.car_classifier.predict(cropped)
                label = self.get_label(predictions)
                labels.append(label)
            # start_time = time.time()

            predictions = self.car_classifier.predict(img)
            # exec_time = time.time() - start_time
            # Q2_time.append()
            result = {"frame_num": frame_num, 'type': labels}
            results.append(result)
            print('Query 2:')
            print(result)
            if query > 2:
                self.send(cars, frame_num, img, labels, query)

    def get_label(self, predictions):
        label = np.argmax(predictions)
        if label == 0:
            label = 'Hatchback'
        else:
            label = 'Sedan'
        return label

    def send(self, cars, frame_num, img, labels, query):
        global q3
        data = {'cars': cars, 'frame_num': frame_num,
                'query': query, 'img': img, 'labels': labels}
        q3.put(data)

    def get_predictions(self):
        pass

# Crop and Color extraction consumer
class Q3Consumer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.color_classifier = ColorClassifier()
        self.video_writer = video = cv2.VideoWriter(
            'final.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (360, 288))

    def run(self):
        global q3
        global results
        # global Q1_time
        # global Q2_time
        # global Q3_time
        while True:
            data = q3.get()
            frame_num = data['frame_num']
            if frame_num == 'exit':
                break
            if frame_num == 1:
                frame_num
            cars = data['cars']
            img = data['img']
            query = data['query']
            labels = data['labels']
            # start_time = time.time()
            for idx, car in enumerate(cars):
                cropped = img.crop(
                    (car['left'], car['top'], car['right'], car['bottom']))
                cropped = np.array(cropped)
                color = self.color_classifier.get_color(cropped)
                labels[idx] += ' '+color

            # exec_time = time.time() - start_time
            # Q3_time.append()
            result = {"frame_num": frame_num, 'labels': labels}
            results.append(result)
            print('Query 3:')
            print(result)
            self.show_video(img, labels, cars)

    #drawing labels and boxes in images write ouput file
    def show_video(self, img, labels, cars):
        font = ImageFont.truetype(font='./FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
        thickness = (img.size[0] + img.size[1]) // 300
        for idx, car in enumerate(cars):
            predicted_class = labels[idx]
            left = car['left']
            top = car['top']
            right = car['right']
            bottom = car['bottom']
            label = predicted_class
            draw = ImageDraw.Draw(img)
            label_size = draw.textsize(label, font)
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline='red')
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill='red')
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        img = np.array(img)
        self.video_writer.write(img)
        # cv2.imshow('f',img)
        # cv2.waitKey(1)


def detect_objects(img, frame_num, query):
    global q1
    if frame_num == 'exit':
        img = []
    else:
        img = Image.fromarray(img)
    data = {"frame_num": frame_num, "img": img, 'query': query}
    q1.put(data)


def start_queries(query):
    jobs = []
    if query > 0:
        jobs.append(Q1Consumer())
        if query > 1:
            jobs.append(Q2Consumer())
            if query > 2:
                jobs.append(Q3Consumer())
    for task in jobs:
        task.start()
    for task in jobs:
        task.join()
    task


def get_results():
    global results
    return results
    # global Q1_time
    # global Q2_time
    # global Q3_time
    # return Q1_time
    # return Q2_time
    # return Q3_time


if __name__ == "__main__":
    pass
