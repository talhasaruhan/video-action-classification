import pickle
import numpy as np
import cv2
import os
import random

bound = 10

vdir = "UCF101"
optical_flow_dir = "optical-flow"
res_dir = "resized"

def mkdir(p):
    if(not os.path.exists(p)):
        os.makedirs(p)
    return

mkdir(vdir+"/"+optical_flow_dir)
mkdir(vdir+"/"+res_dir)

flist = [os.path.join(vdir, file) for file in os.listdir(vdir)]

random.shuffle(flist)

for i, file in enumerate(flist):
    if not os.path.isfile(file):
        continue

    cap = cv2.VideoCapture(file)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fl_name=vdir+"/"+optical_flow_dir+"/"+os.path.basename(file)
    res_name=vdir+"/"+res_dir+"/"+os.path.basename(file)
    if os.path.exists(fl_name) and os.path.exists(res_name):
        print("Passing", file)
        continue

    flow_video = cv2.VideoWriter(fl_name, cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224), isColor=False)
    resized_video = cv2.VideoWriter(res_name, cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224), isColor=True)

    L = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (224, 224))
    resized_video.write(frame1)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    while cap.isOpened() :
        # read frame
        ret, frame2 = cap.read()
        if not ret :
            break

        # resize and save frame in spatial stream
        frame2 = cv2.resize(frame2, (224, 224))
        resized_video.write(frame2)

        # compute optical flow
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # map flow from [-bound, bound] to [0, 255] w/clipping and save optical flow into optical flow stream
        flow = np.round((flow + bound) / (2. * bound) * 255.)
        flow[flow < 0] = 0
        flow[flow > 255] = 255

        flow_video.write(flow[..., 0].astype('u1'))
        flow_video.write(flow[..., 1].astype('u1'))

        # set last frame to next
        frame1 = frame2

    flow_video.release()
    resized_video.release()
    print("Done {}/{}".format(i, len(flist)), file)

    