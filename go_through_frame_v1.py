"""
For connecting to the AirSim drone environment and testing API functionality
"""
import setup_path 
import airsim

import os
import tempfile
import pprint
# import the necessary packages
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from random import randrange

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)


# airsim.wait_key('Manual mode is setup. Press any key to send RC data to takeoff')

# client.moveByRC(rcdata = airsim.RCData(pitch = 0.0, throttle = 30.0, is_initialized = True, is_valid = True))
landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()
else:
    print("already flying...")
    client.hoverAsync().join()

client.moveByManualAsync(vx_max = 1E6, vy_max = 1E6, z_min = -1E6, duration = 9E10)
# airsim.wait_key('Set Yaw and pitch to 0.5')

# client.moveByRC(rcdata = airsim.RCData(roll = 0.5, throttle = 0.0, yaw = 0.5, is_initialized = True, is_valid = True))
import sys, termios, tty, os, time
 
#get rectangle center point
def get_rectangle_center_point(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = 255-gray
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    kernel = np.ones((5,5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    kernel = np.ones((3,3), np.uint8)
    edged = cv2.erode(edged, kernel, iterations=1)
    
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    #get center point of the rectangle
    sum_x = 0
    sum_y = 0
    # loop over the contours individually
    for c in cnts:
        # This is to ignore that small hair countour which is not big enough
        if cv2.contourArea(c) < 20000 or cv2.contourArea(c) > 150000:
            continue
        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        #check that is a square
        percentage = abs(box[1][0]/ box[1][1])
        if(percentage<0.8 or percentage>1.2):
            continue

        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        box = perspective.order_points(box)
        # draw the contours on the image
        orig = image.copy()
        #cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 5)

        #get center point of the rectangle
        sum_x = 0
        sum_y = 0
        
        # loop over the original points
        for (xA, yA) in list(box):
            # draw circles corresponding to the current points and
            #cv2.circle(orig, (int(xA), int(yA)), 9, (0,0,255), -1)
            #cv2.putText(orig, "({},{})".format(xA, yA), (int(xA - 50), int(yA - 10) - 20),
            #    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,0,0), 5)
            sum_x += xA
            sum_y += yA
        #draw center point of the rectangle    
        #cv2.circle(orig, (int(sum_x/4), int(sum_y/4)), 9, (0,0,255), -1)
    return int(sum_x/4), int(sum_y/4)
index_=0
vehicleControl = client.moveByVelocityAsync(0, 0, -1, 2)
vehicleControl.join()
count=0               
while True:
    
    time.sleep(1.2)
    # get camera images from the car
    responses = client.simGetImages([
        airsim.ImageRequest("2", airsim.ImageType.Scene,False,False), 
        airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
        ]) 
    response_left = responses[0]
    response_right = responses[1]
    
    # get numpy array
    left_image = np.fromstring(response_left.image_data_uint8, dtype=np.uint8) 
    # reshape array to 3 channel image array H X W X 3
    img_rgb = left_image.reshape(response_left.height, response_left.width, 3)

    # original image is fliped vertically
    #img_rgb = np.flipud(img_rgb)
    locate="/home/peterparker/Airsim/AirSim/PythonClient/multirotor/202012drone_go_frame/test_img/"+str(index_)+".png"
    cv2.imwrite(locate, img_rgb)
    index_+=1
    frame_center_point_X , frame_center_point_Y= get_rectangle_center_point(img_rgb)
    #airsim.write_file(os.path.normpath(left_filename + '.png'), response_left.image_data_uint8)
    #airsim.write_file(os.path.normpath(right_filename + '.png'), response_right.image_data_uint8)
  
    

    ##########################
    
    point = Point(frame_center_point_X, frame_center_point_Y)
    polygon = Polygon([(930, 510), (990, 510), (990, 570), (930, 570)])
 
    print(frame_center_point_X,frame_center_point_Y)
    if(frame_center_point_X==0 and frame_center_point_Y ==0):
        count+=1
        if(count>3):
            vehicleControl = client.moveByVelocityAsync(0, 1, 0, 0.1)
            vehicleControl.join()
            print("You pressed w")
            print(randrange(2))
            count=0
        # with open("instruction.txt", "a") as myfile:
        #     myfile.write("w\n")
        # print("do nothing")
    elif(polygon.contains(point)==True):
        print("You pressed q")
        break
    else:   
        if(frame_center_point_X > 990):
            vehicleControl = client.moveByVelocityAsync(-1, 0, 0, 0.5)
            vehicleControl.join()
            print("You pressed d")
            with open("instruction.txt", "a") as myfile:
                myfile.write("d\n")
        elif(frame_center_point_X < 930):
            vehicleControl = client.moveByVelocityAsync(1, 0, 0, 0.5)
            vehicleControl.join()
            print("You pressed a")
            with open("instruction.txt", "a") as myfile:
                myfile.write("a\n")
        else:
            if(frame_center_point_Y > 570):
                vehicleControl = client.moveByVelocityAsync(0, 0, 1, 0.5)
                vehicleControl.join()
                print("You pressed p")
                with open("instruction.txt", "a") as myfile:
                    myfile.write("p\n")
            elif(frame_center_point_X < 510):                
                vehicleControl = client.moveByVelocityAsync(0, 0, -1, 0.5)
                vehicleControl.join()
                print("You pressed o")
                with open("instruction.txt", "a") as myfile:
                    myfile.write("o\n")

landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("already landed...")
else:
    print("landing...")
    client.landAsync().join()

client.armDisarm(False)
# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)