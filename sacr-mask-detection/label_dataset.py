from curses import window
import cv2
import numpy as np

from utilities import get_video_frames

# Constants
WIDTH = 1000
HEIGHT = 562
TARGET_SIZE = (WIDTH, HEIGHT)

# Create an empty list of points for the coordinates
list_points = list()
 
# Define the callback function that we are going to use to get our coordinates
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
        list_points.append([x,y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
        list_points.append([x,y])




# Create a black image and a window
windowName = 'Scene Settings'
cv2.namedWindow(windowName)

# bind the callback function to window
cv2.setMouseCallback(windowName, CallBackFunc)


if __name__ == "__main__":
    print("Extracting frames.")
    frames = get_video_frames("video/room6-short.mp4", target_size = TARGET_SIZE)[180:]
    print("Frames: ", len(frames))

    #Start inference on video frames
    for i, frame in enumerate(frames):
        j = 0
        while (True):
            cv2.imshow(windowName, frame)
            if len(list_points) == 2:
                pts = np.array(list_points)
                print(pts)
                list_points = list()
                #cv2.imwrite("faces3/"+str(i)+str(j)+".png", frame[])
                with open("labels/"+str(i)+"-"+str(j)+".txt", "w") as f:
                    f.write(str(pts[0])+"\n"+str(pts[1]))
                
                j = j + 1
            
            #Finish
            key = cv2.waitKey(20)
            if key == 27:
                break
