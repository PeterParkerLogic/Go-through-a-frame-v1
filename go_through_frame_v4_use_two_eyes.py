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
#======================
#psmnet import
#from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import math
#import from another folder
import sys
sys.path.insert(1, '/home/peterparker/Airsim/AirSim/PythonClient/multirotor/psmnet')
from models import *
from PIL import Image
from RT_stereov4 import HRstereoNet
import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import glob
import pickle

#psmnet load model
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--leftimg', default= 'left_63.png',
                    help='load model')
parser.add_argument('--rightimg', default= 'right_63.png',
                    help='load model')                                      
parser.add_argument('--model', default='RT_stereov4',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
#parser.add_argument('--leftdir', required=True)
#parser.add_argument('--rightdir', required=True)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
elif args.model == 'RT_stereov4':
    model = HRstereoNet(args.maxdisp)   
else:
    print('no model')

# print('model: ', model)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    # model.load_state_dict(state_dict['state_dict'])

    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict['state_dict'].items():
        if 'module' not in k:
            k = 'module.' + k

        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
#===========================


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)


landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()
else:
    print("already flying...")
    client.hoverAsync().join()

client.moveByManualAsync(vx_max = 1E6, vy_max = 1E6, z_min = -1E6, duration = 9E10)

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
    max_x = 0
    min_x =10000
    max_y = 0
    min_y =10000
    # loop over the contours individually
    for c in cnts:
        # This is to ignore that small hair countour which is not big enough
        if cv2.contourArea(c) < 7000 or cv2.contourArea(c) > 450000:
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
        max_x = 0
        min_x =10000
        max_y = 0
        min_y =10000
        # loop over the original points
        for (xA, yA) in list(box):
            # draw circles corresponding to the current points and
            #cv2.circle(orig, (int(xA), int(yA)), 9, (0,0,255), -1)
            #cv2.putText(orig, "({},{})".format(xA, yA), (int(xA - 50), int(yA - 10) - 20),
            #    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,0,0), 5)
            max_x = max(max_x,xA)
            min_x = min(min_x,xA)
            max_y = max(max_y,yA)
            min_y = min(min_y,yA)
            sum_x += xA
            sum_y += yA
        #draw center point of the rectangle    
        #cv2.circle(orig, (int(sum_x/4), int(sum_y/4)), 9, (0,0,255), -1)
    return int(sum_x/4), int(sum_y/4),int(max_x) ,int(min_x) ,int(max_y) , int(min_y) 


def dronemove(vx,vy,vz,duration,movement):
    vehicleControl = client.moveByVelocityBodyFrameAsync(vx, vy, vz, duration)
    vehicleControl.join()
    print("You pressed ",movement)
    with open("instruction.txt", "a") as myfile:
        myfile.write(movement+"\n")

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = imgL.cuda()
           imgR = imgR.cuda() 
           #print(imgR.shape)   

        with torch.no_grad():
            disp = model(imgL,imgR)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        ### convert to depth
        baseline = 0.25
        #1920*1080
        focal_length = 1395.9733

        pred_depth = baseline * focal_length / pred_disp
     

    
    
        return pred_disp, pred_depth

def psmnet_predict_depth(left_img,right_img, max_x, min_x, max_y, min_y):
    if(max_x == 0 and min_x ==10000 and max_y == 0 and min_y ==10000):
        return 0
    else:
        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]}
        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**normal_mean_var)])    

        #img_list = glob.glob('/home/peterparker/Airsim/AirSim/PythonClient/multirotor/airsim_drone_left/*.png')
        #concatImg_list = []
        # predDepth_list = []


            
        imgL_o = left_img
        imgR_o = right_img

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o) 

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        #start_time = time.time()
        pred_disp, pred_depth = test(imgL,imgR)
        #process_time= time.time() - start_time
        #print(str(index+1)+' / '+str(len(img_list))+' '+'time = %.2f' %(time.time() - start_time)+'\n')

        if top_pad !=0 or right_pad != 0:
            img = pred_disp[top_pad:,:]
            img_depth = pred_depth[top_pad:,:]
        else:
            img = pred_disp
            img_depth = pred_depth
        
        
        #search img_depth < 2 
        red = np.where(img_depth[min_y-25:max_y+25, min_x-25:max_x+25] < 2)
        #returm how many red points in the polygan
        return len(red[0])
    


index_=0

vehicleControl = client.moveByVelocityAsync(0, 0, -1, 2)
vehicleControl.join()


#count for moving a littile to find the frame
count=0   
#left_right_count ,left right movement will cause a little go up so we need to let up down movement to run
left_right_count = 0          
num_red_in_frame=50
#check same instruction count
intstruction_count=0
intstruction_movement=""
find_2m_three_time_or_not = 0
left_right_move_time = 2
up_down_move_time = 1
sleep_time = 0

while True:
    # if(count<=3):
    #time.sleep(4)
    # get camera images from the car
    responses = client.simGetImages([
        airsim.ImageRequest("2", airsim.ImageType.Scene,False,False), 
        airsim.ImageRequest("1", airsim.ImageType.Scene,False,False), #scene vision image in png format
        ]) 
    response_left = responses[0]
    response_right = responses[1]
    
    # get numpy array
    left_image = np.fromstring(response_left.image_data_uint8, dtype=np.uint8) 
    right_image = np.fromstring(response_right.image_data_uint8, dtype=np.uint8) 
    # reshape array to 3 channel image array H X W X 3
    Limg_rgb = left_image.reshape(response_left.height, response_left.width, 3)
    Rimg_rgb = right_image.reshape(response_right.height, response_right.width, 3)
    # original image is fliped vertically
    #img_rgb = np.flipud(img_rgb)
    locate="/home/peterparker/Airsim/AirSim/PythonClient/multirotor/202012drone_go_frame/test_img/"+str(index_)+".png"
    cv2.imwrite(locate, Limg_rgb)
    index_+=1
    frame_center_point_X, frame_center_point_Y, max_x, min_x, max_y, min_y = get_rectangle_center_point(Limg_rgb)
    #airsim.write_file(os.path.normpath(left_filename + '.png'), response_left.image_data_uint8)
    #airsim.write_file(os.path.normpath(right_filename + '.png'), response_right.image_data_uint8)
    if(find_2m_three_time_or_not < 4):
        red_points_in_polygan = psmnet_predict_depth(Limg_rgb, Rimg_rgb, max_x, min_x, max_y, min_y)
        if(red_points_in_polygan > num_red_in_frame):
            find_2m_three_time_or_not += 1
       
    else:
        left_right_move_time = 0.3
        up_down_move_time = 0.3
        sleep_time = 2

    print("red_points_in_polygan : ",red_points_in_polygan)
    # print("max_x : ",max_x)
    # print("min_x : ",min_x)
    # print("max_y : ",max_y)
    # print("min_y : ",min_y)
    ##########################
    
    point = Point(frame_center_point_X, frame_center_point_Y)
    polygon = Polygon([(935, 515), (985, 515), (985, 565), (935, 565)])
 
    print(frame_center_point_X,frame_center_point_Y)
    #if can not detect any frame 
    if(frame_center_point_X==0 and frame_center_point_Y ==0):
        if(find_2m_three_time_or_not >= 4):
            #just can move left and right
            with open('instruction.txt', 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                if(intstruction_movement==last_line):
                    intstruction_count+=1
                intstruction_movement = last_line
            if(intstruction_count>1):
                if(intstruction_movement=="d"):
                    #turn left 
                    dronemove(0, -0.5, 0, 2,"a")
                    time.sleep(sleep_time)
                else:
                    #turn right
                    dronemove(0, 0.5, 0, 2,"d")
                    time.sleep(sleep_time)
                intstruction_count = 0

            else:
                #turn left or right
                if(intstruction_movement=="a"):
                    #turn left 
                    dronemove(0, -0.5, 0, 2,intstruction_movement)
                    time.sleep(sleep_time)
                else:
                    #turn right
                    dronemove(0, 0.5, 0, 2,intstruction_movement)
                    time.sleep(sleep_time)
            

        count+=1
        print("find_2m_three_time_or_not:",find_2m_three_time_or_not)
        if(find_2m_three_time_or_not < 4 and count>3):
            Denominator = randrange(1,10,1)
            vehicleControl = client.moveByVelocityBodyFrameAsync(1, 0, 0, 0.2)
            vehicleControl.join()
            print("You pressed w")
            print("randrange:",Denominator)
            count=0      
    elif(find_2m_three_time_or_not >= 4 and polygon.contains(point)==True):
        dronemove(0, -0.125, 0, 4,"a")
        time.sleep(sleep_time)
        vehicleControl = client.moveByVelocityBodyFrameAsync(1, 0, 0, 8)
        vehicleControl.join()
        print("You pressed q")
        break
    else:   
       
        if(frame_center_point_Y > 565):
            #go down
            dronemove(0, 0, 0.5, up_down_move_time,"p")
            time.sleep(sleep_time)
            print("570")
            count=0
        elif(frame_center_point_Y < 515): 
            #go up               
            dronemove(0, 0, -0.5, up_down_move_time,"o")
            time.sleep(sleep_time)
            print("510")
            count=0
            left_right_count=0
        if(frame_center_point_X > 985):
            #turn right
            dronemove(0, 0.5, 0, left_right_move_time,"d")
            time.sleep(sleep_time)
            print("990")
            count=0
            left_right_count += 1
        elif(frame_center_point_X < 935):          
            #turn left
            dronemove(0, -0.5, 0, left_right_move_time,"a")
            time.sleep(sleep_time)
            print("930")
            count=0
            left_right_count += 1
    
    
    
            



landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("already landed...")
else:
    print("landing...")
    client.landAsync().join()

client.armDisarm(False)
# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)