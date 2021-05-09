"""
Created on Tue Apr 20 03:08:46 2021
GUI Server
@author: Qureshi
"""

######Libraries for Gui###################### 
from tkinter import *
from tkinter import scrolledtext
import PIL
import time
from PIL import Image,ImageTk
######Libraries for the overall server#######
import cv2
######library for  sending over socket########

import imagezmq
image_hub = imagezmq.ImageHub()

#####Librarry for decoding of images for faster sending over socket### 
import simplejpeg
import numpy as np
#from matplotlib import pyplot as plt
#import keras
import tensorflow.keras as keras
from keras.models import load_model
from keras.preprocessing import image
import face_recognition
import os
#from PIL import Image



#########the server and whole appratus code#####
###declaring Global Variable####
frameL = np.zeros([300,300,3],dtype= np.uint8)
name = 'no name'
img_disp = np.zeros([520,520,3],dtype= np.uint8)
disparity = np.zeros([520,520,3],dtype= np.uint8)
filteredImg = np.zeros([300,300,3],dtype= np.uint8)
filtered= np.zeros([520,520,3],dtype= np.uint8)
color = (255, 255, 255)
my_model = load_model('Inferance_face_antispoofing-1618217329_____inferance_face_antispoof_classifier.h5')

def face_detector_croper_valuer(img):
    global memory
    frame = img
    small_frame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25)
    faces = face_recognition.face_locations(small_frame,
                            number_of_times_to_upsample=2, model="hog")
    for index, faces_found in enumerate(faces):
            
        top, right, bottom, left = faces_found
        left = left * 4
        top = top * 4
        right = right * 4
        bottom = bottom * 4
        frame = cv2.rectangle(frame, 
                        pt1 = (left, top), pt2 =(right, bottom) ,
                        color = (255,0,0))
        
        memory = [top, bottom, left,right]
    return frame, memory[0], memory[1], memory[2], memory[3]


def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color



num_disparities = 64  # number of disparities to check
block = 9  # block size to match

KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.45
print('Loading known faces...')
known_face_encoding = []
known_face_names = []

# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):
    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]
                
        # Append encodings and name
        known_face_encoding.append(encoding)
        known_face_names.append(name)
all_face_locations = []
all_face_encodings = []
all_face_name = []   


value = 0
def depth_map(imgL, imgR):
    global x,y
    global depth_of_pixel
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size =3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=4*16,#5*16  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=50,  #default was 10 and after 15
        speckleWindowSize=200,  #default was 50
        speckleRange=32,  #default was 32
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000#80000
    sigma = 1.3#1.3
    visual_multiplier = 6#6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR).astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL).astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    #############Finding Distance of objective pixel##########
    #####using formula z = (F*b)/d
    #####b is found by takin sqrt(of all three terms of translational matrix T )
    baseline = 3.027848 # in cm
    focal_length = 772.46502013
    #filteredImg = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET )
    #filteredImg = filteredImg[20:500, 80:490]

    return filteredImg

def rectification(lFrame, rFrame):

    w, h = lFrame.shape[:2] # both frames should be of same shape
    frames = [lFrame, rFrame]
    ###CAmera parameters##############


    cameraMatrix1 =  np.array( [[438.3834058,    0.0,         237.55659974],
 [  0.0,         783.64808014, 273.71711951],
 [  0.0,           0.0,           1.0,        ]] )

    cameraMatrix2 = np.array( [[425.29925777,   0.0,         226.01267985],
 [  0.0,         761.28196013, 275.71170695],
 [  0.0,           0.0,           1.0        ]]  )

    camMats = [cameraMatrix1, cameraMatrix2]
    #camMats = [mtxL, mtxR]

    distCoeffs1 =  np.array([[ 4.09491645e-01, -2.58001795e+00,  5.03661908e-03,  2.16913214e-03,
   8.11592997e+00]] )
    distCoeffs2 = np.array([[ 5.05836475e-01, -3.98317770e+00,  1.44028937e-02, -1.74233101e-02,
   1.48314464e+01]])

    #R and P parameters of cameras

    R1 = np.array([[ 0.92103433, -0.04157285,  0.38725633],
 [ 0.05081355,  0.99861488, -0.01364927],
 [-0.38615249,  0.03224932,  0.92187105]] )

    R2 = np.array( [[ 0.92743215, -0.02023811,  0.37344347],
 [ 0.01626866,  0.99977272,  0.01377834],
 [-0.37363744, -0.00670305,  0.92755061]] )


    P1 = np.array([[ 772.46502013,    0.0,         -115.88994122,    0.0        ],
 [   0.0,          772.46502013,  260.84034729,    0.0        ],
 [   0.0,            0.0,            1.0,            0.0        ]])

    P2 = np.array([[ 772.46502013,    0.0,         -115.88994122,  -25.21916459],
 [   0.0,          772.46502013,  260.84034729,    0.0        ],
 [   0.0,            0.0,            1.0,            0.0        ]] )


    leftMapX, leftMapY = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1)

    left_rectified = cv2.remap(lFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)


    rightMapX, rightMapY = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_32FC1)

    right_rectified = cv2.remap(rFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)



    return left_rectified, right_rectified

def face_detector(img, disparity):
    global img_disp, filtered, color
    frame = img
    filtered = disparity
    
    small_frame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25)
    faces = face_recognition.face_locations(small_frame,
                            number_of_times_to_upsample=2, model="hog")
        
    #####finding face encodings of frame grabbed
    all_face_encodings = face_recognition.face_encodings(small_frame, faces)
 
    #looping through the face locations and the face embeddings

    for current_face_location,current_face_encoding in zip(faces, all_face_encodings):
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
    
        #global known_face_encoding
        #find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encoding, current_face_encoding, TOLERANCE)
       
        #string to hold the label
        
        name_of_person = 'Unknown Registery'
        name = 'no_Depth'    
        img_disp = disparity
        img_disp = img_disp[top_pos:bottom_pos, left_pos:right_pos]
        
        cv2.imwrite('./1img_disp.jpg', img_disp)
        # resize the array (image) then PIL image
            
        test_image = cv2.imread('./1img_disp.jpg')
        test_image = cv2.resize(test_image, (100, 100))
        test_image = (test_image[...,::-1].astype(np.float32)) 
            
            
        test_image = np.expand_dims(test_image, axis = 0)
        test_image = test_image /255
            
            
        prediction1 = my_model.predict(test_image)
        result1 =  np.argmax(prediction1,axis=1)
        result1 = str(result1)
        result1 = result1.replace('[', '')
        result1 = result1.replace(']', '')
        result = int(result1)
        #print("\n\n\n class is \n\n", str(result))
            
        if(result <= 0):
            name = 'FACE'
            #print("\n\t\t\tThis is Face\n")
            #console_text_box.delete(1.0,END)
            #console_text_box.insert(INSERT,'\n'+ str(name_of_person))
            #console_text_box.insert(INSERT,'\n\nThis is Face\n')
        elif(result <= 1):
            name = 'Anti_face'
            #print("\n\t\t\tThis is Anti Face\n")
            #console_text_box.delete(1.0,END)
            #console_text_box.insert(INSERT,'\n'+ str(name_of_person))
            #console_text_box.insert(INSERT,'\n\nThis is Anti Face\n')
        
        
        
        #check if the all_matches have at least one item
        #if yes, get the index number of face that is located in the first(because of accurate first result) index of all_matches
        #get the name corresponding to the index number and save it in name_of_person
        if (True in all_matches):
            global color, filteredImg    
            first_match_index = all_matches.index(True)
            #print("\n\t\t\tI am true\t", first_match_index)
            name_of_person = known_face_names[first_match_index]
            color = name_to_color(name_of_person)
            img_disp = disparity
            img_disp = img_disp[top_pos:bottom_pos, left_pos:right_pos]
            cv2.imwrite('./1img_disp.jpg', img_disp)
            # resize the array (image) then PIL image
            
            test_image = cv2.imread('./1img_disp.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
            #img1 = image.img_to_array(im_resized)
            #img1 = np.expand_dims(img1, axis = 0)
            #img1 = img1 /255
            
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction1 = my_model.predict(test_image)
            result1 =  np.argmax(prediction1,axis=1)
            result1 = str(result1)
            result1 = result1.replace('[', '')
            result1 = result1.replace(']', '')
            result = int(result1)
            #print("\n\n\n class is \n\n", str(result))
            
            name = 'no name'
            if(result <= 0):
                name = 'FACE'
                #print("\n\t\t\tThis is Face\n")
                #console_text_box.delete(1.0,END)
                #console_text_box.insert(INSERT,'\n'+ str(name_of_person))
                #console_text_box.insert(INSERT,'\n\nThis is Face\n')
                #cv2.imshow(name,img)
            elif(result <= 1):
                name = 'Anti_face'
                #print("\n\t\t\tThis is Anti Face\n")
                #console_text_box.delete(1.0,END)
                #console_text_box.insert(INSERT,'\n'+ str(name_of_person))
                #console_text_box.insert(INSERT,'\n\nThis is Anti Face\n')
        
     
        
        ###display  name 
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.rectangle(filtered,(left_pos,top_pos),(right_pos,bottom_pos),color,2)
        cv2.putText(filtered, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
        cv2.putText(filtered, name, (right_pos,top_pos), font, 0.5, (255,255,255),1)
        
    return filtered



###################################################
   
#######################################




#######The GUI Code#####

window = Tk()

window.title("DSM app")
window.configure(background = '#23272a')
window.geometry('900x900')
window.bind('<Escape>', lambda e: show_frame())
def close_gate():
    frame_gateCl = cv2.imread('./closed_gate.jpg')
    cv2imageGate = cv2.cvtColor(frame_gateCl, cv2.COLOR_BGR2RGB)
    cv2imageGate = cv2.resize(cv2imageGate, (300,300))
    imgG = PIL.Image.fromarray(cv2imageGate)
    Gimgtk = ImageTk.PhotoImage(image=imgG)
    gate_image_lbl.Gimgtk = Gimgtk
    gate_image_lbl.configure(image=Gimgtk)
    console_text_box.delete(1.0,END)
    console_text_box.insert(INSERT,'\nclosing Gate in 3..2..1')
    console_text_box.insert(INSERT,'\n\nGate Closing Successfull\n')
    welcome_msg_lbl.configure(text = 'Welcome To Server')                        
    _, _ = image_hub.recv_jpg()
    image_hub.send_reply(b'close')            
    
 
def verify_and_open_gate():
    #print('\nVerified')
    console_text_box.delete(1.0,END)
    sent_from, jpg_buffer = image_hub.recv_jpg()
    image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
    image_hub.send_reply(b'OK')            
            
    frameL = image[:, :500]
    frameR = image[:, 500:]
                
                
    left_rec, right_rec = rectification(frameL, frameR)
    lft_rec_for_face = left_rec
    frame = lft_rec_for_face
    small_frame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25)
    faces = face_recognition.face_locations(small_frame,
                            number_of_times_to_upsample=2, model="hog")
        
    #####finding face encodings of frame grabbed
    all_face_encodings = face_recognition.face_encodings(small_frame, faces)
 
    #looping through the face locations and the face embeddings
    
    for current_face_location,current_face_encoding in zip(faces, all_face_encodings):
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
    
        #global known_face_encoding
        #find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encoding, current_face_encoding, TOLERANCE)
       
        #string to hold the label
        
        name_of_person = 'Unknown Registery'
        name = 'Not Regitered'    
        #print("\nYes I am in Verify\n")    
        #check if the all_matches have at least one item
        #if yes, get the index number of face that is located in the first(because of accurate first result) index of all_matches
        #get the name corresponding to the index number and save it in name_of_person
        if (True in all_matches):
            global color, filteredImg    
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
            ############starting model for detecting antispoof attaxck####
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
            #print("\nYes I am in Verify if(True)\n")    
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp1.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp1.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction1 = my_model.predict(test_image)
            result1 =  np.argmax(prediction1,axis=1)
            result1 = str(result1)
            result1 = result1.replace('[', '')
            result1 = result1.replace(']', '')
            
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp2.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp2.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction2 = my_model.predict(test_image)
            result2 =  np.argmax(prediction2,axis=1)
            result2 = str(result2)
            result2 = result2.replace('[', '')
            result2 = result2.replace(']', '')
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp3.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp3.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction3 = my_model.predict(test_image)
            result3 =  np.argmax(prediction3,axis=1)
            result3 = str(result3)
            result3 = result3.replace('[', '')
            result3 = result3.replace(']', '')
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp4.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp4.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction4 = my_model.predict(test_image)
            result4 =  np.argmax(prediction4,axis=1)
            result4 = str(result4)
            result4 = result4.replace('[', '')
            result4 = result4.replace(']', '')
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp5.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp5.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction5 = my_model.predict(test_image)
            result5 =  np.argmax(prediction5,axis=1)
            result5 = str(result5)
            result5 = result1.replace('[', '')
            result5 = result1.replace(']', '')
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp6.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp6.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction6 = my_model.predict(test_image)
            result6 =  np.argmax(prediction6,axis=1)
            result6 = str(result6)
            result6 = result6.replace('[', '')
            result6 = result6.replace(']', '')
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp7.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp7.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction7 = my_model.predict(test_image)
            result7 =  np.argmax(prediction7,axis=1)
            result7 = str(result7)
            result7 = result7.replace('[', '')
            result7 = result7.replace(']', '')
            
            
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp7.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp7.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction8 = my_model.predict(test_image)
            result8 =  np.argmax(prediction8,axis=1)
            result8 = str(result8)
            result8 = result8.replace('[', '')
            result8 = result7.replace(']', '')
            
            
            
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp7.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp7.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction9 = my_model.predict(test_image)
            result9 =  np.argmax(prediction9,axis=1)
            result9 = str(result9)
            result9 = result9.replace('[', '')
            result9 = result9.replace(']', '')
            
            
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp7.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp7.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction10 = my_model.predict(test_image)
            result10 =  np.argmax(prediction10,axis=1)
            result10 = str(result10)
            result10 = result10.replace('[', '')
            result10 = result10.replace(']', '')
            
            
            
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp7.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp7.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction11 = my_model.predict(test_image)
            result11 =  np.argmax(prediction11,axis=1)
            result11 = str(result11)
            result11 = result11.replace('[', '')
            result11 = result11.replace(']', '')
            
            
            
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp7.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp7.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction12 = my_model.predict(test_image)
            result12 =  np.argmax(prediction12,axis=1)
            result12 = str(result12)
            result12 = result12.replace('[', '')
            result12 = result12.replace(']', '')
            
            
            
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp7.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp7.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction13 = my_model.predict(test_image)
            result13 =  np.argmax(prediction13,axis=1)
            result13 = str(result13)
            result13 = result13.replace('[', '')
            result13 = result13.replace(']', '')
            
            
            
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp7.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp7.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction14 = my_model.predict(test_image)
            result14 =  np.argmax(prediction14,axis=1)
            result14 = str(result14)
            result14 = result14.replace('[', '')
            result14 = result14.replace(']', '')
            
            
            
            
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
            image_hub.send_reply(b'OK')            
            
            frameL = image[:, :500]
            frameR = image[:, 500:]
                
                
            left_rec, right_rec = rectification(frameL, frameR)
            lft_rec_for_face = left_rec
            _, a, b, c, d = face_detector_croper_valuer(lft_rec_for_face)
            
            Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
            Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(Lgray,Rgray )
            
            img_disp = disparity
            img_disp = img_disp[a:b, c:d]
            cv2.imwrite('./img_disp7.jpg', img_disp)
            
            test_image = cv2.imread('./img_disp7.jpg')
            test_image = cv2.resize(test_image, (100, 100))
            test_image = (test_image[...,::-1].astype(np.float32)) 
            
        
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image /255
            
            
            prediction15 = my_model.predict(test_image)
            result15 =  np.argmax(prediction15,axis=1)
            result15 = str(result15)
            result15 = result15.replace('[', '')
            result15 = result15.replace(']', '')
            
            
            
            result = int(result1) +int(result2) +int(result3) +int(result4) +int(result5) +int(result6) +int(result7)+int(result8)+int(result9)+int(result10)+int(result11)+int(result12)+int(result13)+int(result14)+int(result15) 
            result = result / 15
            name = 'no name'
            print("\n\n\n class is \n\n", str(result))
            if(result <= 0.5):
                name = 'FACE'
                msseg = 'Welcome  ' + str(name_of_person)
                welcome_msg_lbl.configure(text = msseg)
                console_text_box.delete(1.0,END)
                console_text_box.insert(INSERT,'\nWelcome '+ str(name_of_person))
                console_text_box.insert(INSERT,'\nThis is Face')
                console_text_box.insert(INSERT,'\nVerification Successfull')
                console_text_box.insert(INSERT,'\nOpenning Door for you\n')
                console_text_box.insert(INSERT,'\nPlease close the Door after entering\n')
                frame_gateOp = cv2.imread('./opened_gate.jpg')
                cv2imageGate = cv2.cvtColor(frame_gateOp, cv2.COLOR_BGR2RGB)
                cv2imageGate = cv2.resize(cv2imageGate, (300,300))
                imgG = PIL.Image.fromarray(cv2imageGate)
                Gimgtk = ImageTk.PhotoImage(image=imgG)
                gate_image_lbl.Gimgtk = Gimgtk
                gate_image_lbl.configure(image=Gimgtk)
                _, _ = image_hub.recv_jpg()
                image_hub.send_reply(b'open')            
    
             
                
            elif(result >= 0.5):
                name = 'Anti_face'
                console_text_box.delete(1.0,END)
                console_text_box.insert(INSERT,'\n'+ str(name_of_person))
                console_text_box.insert(INSERT,'\nThis is Anti Face')
                console_text_box.insert(INSERT,'\n\nSPOOF ATTACK...ALERTING SYSTEM!!!')
                console_text_box.insert(INSERT,'\n\nInitiating ALARM\n')
                
                        
        else:
            print("\n\tUn registerd try again\n")
            console_text_box.delete(1.0,END)
            console_text_box.insert(INSERT,'\n'+ str(name_of_person))
            console_text_box.insert(INSERT,'\nCoudnt Find match\n')
            console_text_box.insert(INSERT,'\nTry Again If you are registered!!!')
                
    
    

main_name_lbl = Label(window, text = 'Depth Sensing Module       ', font=("Arial Bold", 20), bg = '#23272a', fg = 'white')
main_name_lbl.grid(column = 1, row  = 0)

welcome_msg_lbl  = Label(window, text = 'Welcome To Server', font=("Arial Bold", 15), bg = '#23272a', fg = 'white')
welcome_msg_lbl.grid(column = 2, row = 1)

'''
txt = Entry(window, width = 20, bg = '#2c2f33', fg = 'white')#, state = 'disabled')
txt.grid(column = 2, row = 7)
txt.focus()
'''
close_gate_btn = Button(window, text = 'Click to Close Gate', bg = '#2c2f33', fg = 'white', command = close_gate)
close_gate_btn.grid(column = 2, row = 7)

msg_name_label = Label(window, text = 'Waiting for client..', font=("Arial Bold", 10), bg = '#23272a', fg = 'white')
msg_name_label.grid(column = 0, row =2)

####Defining main frames Labels
framel_lbl = Label(window, bg = '#2c2f33')
framel_lbl.grid(column = 0 , row = 4)

live_depth_lbl = Label(window,  bg = '#2c2f33')
live_depth_lbl.grid(column = 1, row = 4)

live_depth_color_tag_lbl = Label(window, bg = '#2c2f33')
live_depth_color_tag_lbl.grid(column = 1, row = 5)

tag_guider_lbl = Label(window, bg = '#2c2f33')
tag_guider_lbl.grid(column = 1, row = 6)
tag_guider_lbl.configure(text ='Near<--------------------------------------------------->Far')

open_door_button = Button(window, text = 'Click to Verify and Open Gate', bg = '#2c2f33', fg = 'white', command = verify_and_open_gate)
open_door_button.grid(column = 2, row = 6)


console_text_box = scrolledtext.ScrolledText(window, width = 40, height = 20, bg = '#2c2f33', fg = 'white')
console_text_box.grid(column = 0, row = 8)
####methods to use it####
#To insert --->  txt.insert(INSERT,'You text goes here')
#To Delete ---> txt.delete(1.0,END)

gate_image_lbl = Label(window, bg = '#2c2f33')
gate_image_lbl.grid(column = 1, row = 8)
##############Displaying panell####





tag = cv2.imread('./Capture.jpg')
frame_gateCl = cv2.imread('./closed_gate.jpg')
OframeL = frameL
#Oframe_filtered = cv2.flip(filteredImg, 1)
Oframe_filtered = filteredImg
    
cv2imageL = cv2.cvtColor(OframeL, cv2.COLOR_BGR2RGBA)
cv2image_filtered = cv2.cvtColor(Oframe_filtered, cv2.COLOR_BGR2RGB)
cv2imageGate = cv2.cvtColor(frame_gateCl, cv2.COLOR_BGR2RGB)
cv2imageT = tag #cv2.cvtColor(tag, cv2.COLOR_BGR2RGB)
    
cv2imageL = cv2.resize(cv2imageL, (300,300))
cv2image_filtered = cv2.resize(cv2image_filtered, (300,300))
cv2imageGate = cv2.resize(cv2imageGate, (300,300))
cv2imageT = cv2.resize(cv2imageT, (300,20))
    
imgL = PIL.Image.fromarray(cv2imageL)
imgFiltered = PIL.Image.fromarray(cv2image_filtered)
imgG = PIL.Image.fromarray(cv2imageGate)
imgT = PIL.Image.fromarray(cv2imageT)
    
Limgtk = ImageTk.PhotoImage(image=imgL)
Filteredimgtk = ImageTk.PhotoImage(image=imgFiltered)
Gimgtk = ImageTk.PhotoImage(image=imgG)
Timgtk = ImageTk.PhotoImage(image=imgT)
    
    
    
framel_lbl.Limgtk = Limgtk
live_depth_lbl.Filteredimgtk = Filteredimgtk
live_depth_color_tag_lbl.Timgtk = Timgtk
gate_image_lbl.Gimgtk = Gimgtk
    
framel_lbl.configure(image=Limgtk)
live_depth_lbl.configure(image=Filteredimgtk)
live_depth_color_tag_lbl.configure(image=Timgtk)
gate_image_lbl.configure(image=Gimgtk)












########All the tkinter things are defined above
#######################################################################
#######################################################################
#######################################################################
#######################################################################
def show_frame():
        
    sent_from, jpg_buffer = image_hub.recv_jpg()
    image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                colorspace='BGR')
    image_hub.send_reply(b'OK') 
    mseg = 'Reciving form Client '+ str(sent_from)        
    msg_name_label.configure(text = mseg)
    frameL = image[:, :500]
    frameR = image[:, 500:]
        
        
    left_rec, right_rec = rectification(frameL, frameR)
    lft_rec_for_face = left_rec
    Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
    Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)
    disparity = depth_map(Lgray,Rgray )
            
                 
    Depth_Img = face_detector(lft_rec_for_face, disparity)
    filtered_depth = Depth_Img 
    filteredImg = cv2.applyColorMap(filtered_depth, cv2.COLORMAP_JET )
    filteredImg = filteredImg[20:500, 80:490]
    
    
    tag = cv2.imread('./Capture.jpg')
    frame_gateOp = cv2.imread('./opened_gate.jpg')
    frame_gateCl = cv2.imread('./closed_gate.jpg')
    OframeL = frameL
    #Oframe_filtered = cv2.flip(filteredImg, 1)
    Oframe_filtered = filteredImg
    
    cv2imageL = cv2.cvtColor(OframeL, cv2.COLOR_BGR2RGBA)
    cv2image_filtered = cv2.cvtColor(Oframe_filtered, cv2.COLOR_BGR2RGB)
    #cv2imageGate = frame_gateCl
    cv2imageT = tag #cv2.cvtColor(tag, cv2.COLOR_BGR2RGB)
    
    cv2imageL = cv2.resize(cv2imageL, (300,300))
    cv2image_filtered = cv2.resize(cv2image_filtered, (300,300))
    #cv2imageGate = cv2.resize(cv2imageGate, (300,300))
    cv2imageT = cv2.resize(cv2imageT, (300,20))
    
    imgL = PIL.Image.fromarray(cv2imageL)
    imgFiltered = PIL.Image.fromarray(cv2image_filtered)
    #imgG = PIL.Image.fromarray(cv2imageGate)
    imgT = PIL.Image.fromarray(cv2imageT)
    
    Limgtk = ImageTk.PhotoImage(image=imgL)
    Filteredimgtk = ImageTk.PhotoImage(image=imgFiltered)
    #Gimgtk = ImageTk.PhotoImage(image=imgG)
    Timgtk = ImageTk.PhotoImage(image=imgT)
    
    
    
    framel_lbl.Limgtk = Limgtk
    live_depth_lbl.Filteredimgtk = Filteredimgtk
    live_depth_color_tag_lbl.Timgtk = Timgtk
    #gate_image_lbl.Gimgtk = Gimgtk
    
    framel_lbl.configure(image=Limgtk)
    live_depth_lbl.configure(image=Filteredimgtk)
    live_depth_color_tag_lbl.configure(image=Timgtk)
    #gate_image_lbl.configure(image=Gimgtk)
   
    framel_lbl.after(2, show_frame)
    
window.mainloop()
window.quit()
show_frame()