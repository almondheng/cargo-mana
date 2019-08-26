#!/usr/bin/env python

import sys
import string
from random import randint, choice
import numpy as np
import cv2
import json
from azure.storage.blob import BlockBlobService, ContentSettings

service = BlockBlobService(account_name='', account_key='')

first_frame = None
last_hit = False

def geofence(frame, mask_img):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    global first_frame, last_hit
    if first_frame is None:
        first_frame = gray
        return frame

    frame_delta = cv2.absdiff(first_frame, gray)
    ret, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)

    height, width, depth = frame.shape

    # mask_img = cv2.resize(mask_img, thresh.shape)
    masked = cv2.bitwise_and(thresh, mask_img)

    thresh = cv2.dilate(masked, None, iterations=2)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if any(cv2.contourArea(contour) > 10 for contour in contours):
        if not last_hit:
            cv2.rectangle(frame, (0, 0), (width, height), (0,0,255), 3)
        last_hit = not last_hit

    #cv2.imshow('debug', thresh)

    colored_gray = cv2.bitwise_and(gray, mask_img)
    colored_frame = cv2.cvtColor(colored_gray, cv2.COLOR_GRAY2RGB)
    colored_frame[:, :, 2][mask_img == 255] = 255
    frame = frame + colored_frame
    return frame

def main():

    # capture video from webcam
    cap = cv2.VideoCapture(0)

    # set video size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    # read first frame
    success, frame = cap.read()
    
    if not success:
        print('Failed to read video.')
        sys.exit(1)

    # create multi_tracker object
    multi_tracker = cv2.multi_tracker_create()

    # load mask image
    mask_img = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    # width, height, channels = frame.shape
    # mask_img = cv2.resize(mask_img, (height, width))

    print('press SPACE to register cargo')

    colors = []
    labels = []

    first_frame = None
    
    while cap.isOpened():
        success, original_frame = cap.read()
        frame = original_frame.copy()
        if not success:
            print('Failed to read video')
            sys.exit(1)

        # get updated location of objects in subsequent frames
        success, boxes = multi_tracker.update(frame)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            x, y = newbox[0], newbox[1]
            w, h = newbox[2], newbox[3]
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            cv2.putText(frame, labels[i], (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, int(w)*int(h)/100000+0.5,(0,0,255),2)

        # draw geofence
        frame = geofence(frame, mask_img)

        # show frame
        #cv2.namedWindow('Cargo Mana', cv2.WINDOW_NORMAL)
        cv2.imshow('Cargo Mana', frame)
        k = cv2.waitKey(1) & 0xFF
				
        # select object to track on SPACE button
        if k == ord(' '):
            bbox = cv2.selectROI('multi_tracker', frame)
            colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
            labels.append(f'{randint(0, 1000):03}-{randint(0, 1000):02} {randint(0, 1000):04}')
            # create CSRT algorithm tracker and add to multi_tracker
            multi_tracker.add(cv2.TrackerCSRT_create(), frame, bbox)
            success, boxes = multi_tracker.update(frame)
						
				# send data on S button
        if k == ord('s'):
            payload = {label: dict(zip(('x', 'y', 'w', 'h'), box))
            for label, box in zip(labels, boxes)}
            ok, original_image = cv2.imencode('.jpg', original_frame)
            content_settings = ContentSettings('image/jpeg')
            service.create_blob_from_bytes('box', 'frame.jpg', original_image.tobytes(), content_settings=content_settings)
            content_settings = ContentSettings('application/json')
            service.create_blob_from_text('box', 'box.json', json.dumps(payload), content_settings=content_settings)
            print(payload)
						
        # reset geofence on R button
        if k == ord('r'):
            first_frame = None

        # quit on ESC button
        if k == 27: # Esc pressed
            break

if __name__ == '__main__':
    main()
