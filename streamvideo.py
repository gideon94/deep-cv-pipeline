# Main file in the pipeline run this file to get results
# used for streaming the given video and consuming the video

import csv
import queue
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from objectdetection import detect_objects, get_results, start_queries

QUERY = 3
q = queue.Queue()
EXIT = False

# Video consumer calls the queries
class Consumer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while(not EXIT):
            global q
            global QUERY
            frame, frame_num = q.get()
            detect_objects(frame, frame_num, QUERY)
            if frame_num == 'exit':
                break

# video streamer
class Producer(threading.Thread):
    def __init__(self, path):
        threading.Thread.__init__(self)
        self.frame_num = 0
        self.vid = cv2.VideoCapture(path)

    def run(self):
        global q
        frame_num = self.frame_num
        vid = self.vid
        while(vid.isOpened()):
            ret, frame = vid.read()
            if ret:
                frame_num += 1
                data = (frame, frame_num)
                q.put(data)
            else:
                data = (frame, 'exit')
                q.put(data)
                break
            time.sleep(1/30)
        vid.release()

# number of queries to execute
def set_query(query):
    global EXIT
    global QUERY
    if query == 0:
        EXIT = True
    QUERY = query

# get query number from user
def get_input():
    query = int(input('Enter query number:'))
    set_query(query)
    return query

# csv report not in format given
def write_to_csv():
    results = get_results()
    df = pd.DataFrame(results)
    df.to_csv('report.csv')


def main():
    jobs = []
    query = get_input()
    # append query consumers
    jobs.append(threading.Thread(target=start_queries, args=(query,)))
    jobs.append(Producer('./video.mp4'))
    jobs.append(Consumer())

    # start the tasks
    for task in jobs:
        task.start()
    for task in jobs:
        task.join()
    write_to_csv()


if __name__ == "__main__":
    main()
