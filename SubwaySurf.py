import numpy as np

import cv2
import pyautogui
from time import time
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt
import math as m


# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for images.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# Setup the Pose function for videos.
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# Initialize mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils




def detectPose(image, pose, draw=False, display=False):

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Check if any landmarks are detected and are specified to be drawn.
    if results.pose_landmarks and draw:
        # Draw Pose Landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237),
                                                                                 thickness=2, circle_radius=2))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');
        plt.show()
    # Otherwise
    else:

        # Return the output image and the results of pose landmarks detection.
        return output_image, results

def tryLandmarks():
    IMAGE_PATH = 'sample.jpeg'
    img = cv2.imread(IMAGE_PATH)
    detectPose(img, pose_image, draw=True, display=True)

def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

def checkHandsJoined(image, results, draw=False, display=False):

    lm = results.pose_landmarks
    h, w = image.shape[:2]
    lmPose = mp_pose.PoseLandmark

    output_image = image.copy()

    l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
    l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)

    r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
    r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)

    dist = findDistance(l_wrist_x,l_wrist_y,r_wrist_x,r_wrist_y)

    if dist < 140:
        handStatus = 'Hands Joined'
        color  = (0,255,0)
    else:
        handStatus = 'Hands not joined'
        color = (0,0,255)

    if draw:
        cv2.putText(output_image, handStatus, (10,30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        cv2.putText(output_image,f'Distance between hands: {dist}',(10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    else:
        return output_image, handStatus

def tryHands():
    cameraVideo = cv2.VideoCapture(0)
    cameraVideo.set(3, 1280)
    cameraVideo.set(4, 960)

    cv2.namedWindow('Hands Joined?', cv2.WINDOW_NORMAL)

    while cameraVideo.isOpened():
        ok, frame = cameraVideo.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        # Get the height and width of the frame of the webcam video.
        frame_height, frame_width, _ = frame.shape

        # Perform the pose detection on the frame.
        frame, results = detectPose(frame, pose_video, draw=True)

        # Check if the pose landmarks in the frame are detected.
        if results.pose_landmarks:
            # Check if the left and right hands are joined.
            frame, _ = checkHandsJoined(frame, results, draw=True)

        # Display the frame.
        cv2.imshow('Hands Joined?', frame)

        # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed and break the loop.
        if (k == 27):
            break

    # Release the VideoCapture Object and close the windows.
    cameraVideo.release()
    cv2.destroyAllWindows()


def checkLeftRight(image, results, draw=False, display=False):

    lm = results.pose_landmarks
    h, w = image.shape[:2]
    lmPose = mp_pose.PoseLandmark

    output_image = image.copy()

    horizontalPosition = None

    shr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    shr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

    shl_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    shl_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

    distance_shoulders = shr_x - shl_x

    midx = w//2

    if shr_x > midx and shl_x > midx:
        horizontalPosition = 'Right'

    elif shl_x<midx and shr_x < midx:
        horizontalPosition = 'Left'

    else:
        horizontalPosition = 'Center'

    if draw:
        cv2.putText(output_image, horizontalPosition, (5, h - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

        cv2.line(output_image, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)

    if display:

        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

        # Otherwise
    else:
        return output_image, horizontalPosition


def tryLeftRight():
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)

    cv2.namedWindow('Horizontal Movements', cv2.WINDOW_NORMAL)

    # Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():

        # Read a frame.
        ok, frame = camera_video.read()

        # Check if frame is not read properly then continue to the next iteration to read the next frame.
        if not ok:
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        # Get the height and width of the frame of the webcam video.
        frame_height, frame_width, _ = frame.shape

        # Perform the pose detection on the frame.
        frame, results = detectPose(frame, pose_video, draw=True)

        # Check if the pose landmarks in the frame are detected.
        if results.pose_landmarks:
            # Check the horizontal position of the person in the frame.
            frame, _ = checkLeftRight(frame, results, draw=True)

        # Display the frame.
        cv2.imshow('Horizontal Movements', frame)

        # Wait for 1ms. If a a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed and break the loop.
        if (k == 27):
            break

    # Release the VideoCapture Object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()

def checkJumpCrouching(image, results, midy = 250, draw=False, display=False):
    lm = results.pose_landmarks
    h, w = image.shape[:2]
    lmPose = mp_pose.PoseLandmark

    output_image = image.copy()

    verticalPosition = None

    shr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    shr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

    shl_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    shl_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

    currrmid = abs(shl_y+shr_y)/2

    upper_bound = midy+100
    lower_bound = midy-15

    if currrmid>upper_bound:
        verticalPosition = 'Crouching'
    elif currrmid < lower_bound:
        verticalPosition = 'Jumping'
    else:
        verticalPosition = 'Stand'

    if draw:
        cv2.putText(output_image, verticalPosition, (5, h - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

        cv2.line(output_image, (0, midy),(w, midy),(255, 255, 255), 2)

    if display:

        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

        # Otherwise
    else:
        return output_image, verticalPosition


def tryJumpCrouching():
    # Initialize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)

    # Create named window for resizing purposes.
    cv2.namedWindow('Verticial Movements', cv2.WINDOW_NORMAL)

    # Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():

        # Read a frame.
        ok, frame = camera_video.read()

        # Check if frame is not read properly then continue to the next iteration to read the next frame.
        if not ok:
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        # Get the height and width of the frame of the webcam video.
        frame_height, frame_width, _ = frame.shape

        # Perform the pose detection on the frame.
        frame, results = detectPose(frame, pose_video, draw=True)

        # Check if the pose landmarks in the frame are detected.
        if results.pose_landmarks:
            # Check the posture (jumping, crouching or standing) of the person in the frame.
            frame, _ = checkJumpCrouching(frame, results, draw=True)

        # Display the frame.
        cv2.imshow('Verticial Movements', frame)

        # Wait for 1ms. If a a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed and break the loop.
        if (k == 27):
            break

    # Release the VideoCapture Object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()

def game():
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)

    time1 = 0

    # Create named window for resizing purposes.
    cv2.namedWindow('Play Game', cv2.WINDOW_NORMAL)
    pos_x = 0
    pos_y = 0
    gameStart = False
    num_frames = 10
    counter = 0
    midy = 0

    while camera_video.isOpened():
        ok, frame = camera_video.read()

        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        # Get the height and width of the frame of the webcam video.
        frame_height, frame_width, _ = frame.shape

        # Perform the pose detection on the frame.
        frame, results = detectPose(frame, pose_video, draw=True)

        if results.pose_landmarks:
            if gameStart:
                frame, horizontalPosition = checkLeftRight(frame,results,draw=True)

                if (horizontalPosition == 'Left' and pos_x!=-1) or (horizontalPosition == 'Center' and pos_x == 1):
                    pyautogui.press('left')
                    pos_x -=1

                elif (horizontalPosition == 'Right' and pos_x!=1) or (horizontalPosition== 'Center' and pos_x == -1):
                    pyautogui.press('right')
                    pos_x+=1


                frame, verticalPosition = checkJumpCrouching(frame,results,midy,draw = True)

                if (verticalPosition == 'Jumping' and pos_y==0):
                     pyautogui.press('up')
                     pos_y = 1
                elif verticalPosition == 'Crouching' and pos_y==0:
                    pyautogui.press('down')
                    pos_y = -1

                elif verticalPosition == 'Stand' and pos_y != 0:

                    # Update the veritcal position index of the character.
                    pos_y = 0
            else:
                cv2.putText(frame, 'JOIN BOTH HANDS TO START THE GAME.', (5, frame_height - 10), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 3)

            if checkHandsJoined(frame,results,draw = True)[1] == 'Hands Joined':

                counter += 1

                if counter >= num_frames:
                    if not gameStart:
                        gameStart = True
                        pos_y  = 0
                        shr_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)
                        shl_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)
                        midy = abs(shr_y+shl_y)//2
                        print(midy)
                        #start game

                        pyautogui.click(x=1300, y=800, button='left')

                    else:
                        # Press the space key.
                        pyautogui.press('space')

                    counter  = 0


        else:
            counter = 0
            gameStart = False

        time2 = time()

        # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
        if (time2 - time1) > 0:
            # Calculate the number of frames per second.
            frames_per_second = 1.0 / (time2 - time1)

            # Write the calculated number of frames per second on the frame.
            cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 3)

        # Update the previous frame time to this frame time.
        # As this frame will become previous frame in next iteration.
        time1 = time2

        # ----------------------------------------------------------------------------------------------------------------------

        # Display the frame.
        cv2.imshow('Subway Surfers with Pose Detection', frame)

        # Wait for 1ms. If a a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed and break the loop.
        if (k == 27):
            break

    camera_video.release()
    cv2.destroyAllWindows()

game()



