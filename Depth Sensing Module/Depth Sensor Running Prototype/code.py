
import cv2
import numpy as np
#from matplotlib import pyplot as plt
#import keras
#from keras.models import load_model
#from keras.preprocessing import image


import imagezmq
image_hub = imagezmq.ImageHub()

#####Librarry for decoding of images for faster sending over socket### 
import simplejpeg



import time
name = 'no name'
img_disp = np.zeros([520,520,3],dtype= np.uint8)

memory = [None, None, None, None]
nullifier = (None, None, None, None)


#my_model = load_model('classifier.h5')



#import keyboard
#from webcam_code import Camera

num_disparities = 64  # number of disparities to check
block = 9  # block size to match



#camie = Camera(srcl = 0, srcr = 1)

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
    filteredImg = filteredImg[20:500, 80:490]

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


    #Fliping upside down the images
    #left_rectified = cv2.flip(left_rectified, -1)
    #right_rectified =  cv2.flip(right_rectified, -1)


    return left_rectified, right_rectified



def face_detector(img):
   face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   faces=face_cascade.detectMultiScale(img,scaleFactor= 3, minNeighbors =5)
   global  memory
    
   for(x,y,w,h) in faces :
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      print("\n\t\tHere are values  : ", x, y, w, h)
      memory = [x, y, w, h]
    
   return img, memory[0], memory[1], memory[2], memory[3]



def capture(event,x,y,flags,param):
        global value
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            #imgDisp = disparity
            imgCDisp = filteredImg 
            imgDisp = disparity
            #_,a,b,c,d = face_detector(left_rectified)
            #imgDisp = imgDisp[a:a+c, b:b+d]
            #a, b, c, d = nullifier
            #cv2.imwrite('./Live_anti_spoof/left_image '+str(value)+' .jpg',framL )
            #cv2.imwrite('./Live_anti_spoof/right_image '+str(value)+' .jpg',framR )
            cv2.imwrite('./Live_Depth_anti_spoof/L_with_color_Anti_face_D_Wout_Crop_image '+str(value)+' .jpg',imgCDisp )
            cv2.imwrite('./Live_Depth_anti_spoof/L_without_color_Anti_face_D_Wout_Crop_image '+str(value)+' .jpg',imgDisp )
            
            value = value + 1
            print("\n\n\t\t\tset of image pair "+str(value)+" is being captured")

        '''
        
        '''
        elif event == cv2.Event_RBUTTONDOWN:
            img_disp = disparity
            img = cv2.resize(disparity, (100,100))
            img = image.img_to_array(img)
            print("\n\n\nShape of image is \n\n", img.shape)
            img = np.expand_dims(img, axis = 0)
            img = img /255
            result = my_model.predict_classes(img)
            prediction = my_model.predict(img)
            print("\n\n\n class is \n\n", str(result))
            print("\n\n\n Prediction are \n\n", prediction)
            if(str(result) == '[[0]]'):
                name = 'FACE'
                #cv2.imshow(name,img)
            elif(str(result) == '[[1]]'):
                name = 'Anti_face'
            '''

cv2.namedWindow("image window")
cv2.setMouseCallback("image window",capture)

while True:
        sent_from, jpg_buffer = image_hub.recv_jpg()
        image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
        image_hub.send_reply(b'OK')            
            
        frameL = image[:, :500]
        frameR = image[:, 500:]
        
        #frameL, frameR = camie.capture_frame()
        left_rectified, right_rectified = rectification(frameL, frameR)

        #left_rectified = left_rectified[0:500,125:299]
        #right_rectified = right_rectified[0:500,125:299]
        left_rectified = cv2.resize(left_rectified, (500, 500))
        right_rectified = cv2.resize(right_rectified, (500, 500))

        #time.sleep(0.3)
        #grid For left & right  frame

        pt1= (0,int((frameL.shape[0])/2))
        pt2= (int(frameL.shape[1]),int((frameL.shape[0])/2))
        cv2.line(frameL,pt1 ,pt2 ,color=(0,0,255), thickness= 1 )
        pt1= (int((frameL.shape[1])/2),0)
        pt2= (int((frameL.shape[1])/2),int(frameL.shape[0]))
        cv2.line(frameL,pt1 ,pt2 ,color= (0,0,255) ,thickness= 1 )
        #grid for right frame
        pt1= (0,int((frameR.shape[0])/2))
        pt2= (int(frameR.shape[1]),int((frameR.shape[0])/2))
        cv2.line(frameR,pt1 ,pt2 ,color=(0,0,255), thickness= 1 )
        pt1= (int((frameR.shape[1])/2),0)
        pt2= (int((frameR.shape[1])/2),int(frameR.shape[0]))
        cv2.line(frameR,pt1 ,pt2 ,color= (0,0,255) ,thickness= 1 )

        #Combinig images
        #frameL = cv2.resize(frameL, (400,400))
        #frameR = cv2.resize(frameR, (400,400))

        new_img = np.hstack((frameL, frameR))


        ####For Rectified Images


        pt1= (0,int((left_rectified.shape[0])/2))
        pt2= (int(left_rectified.shape[1]),int((left_rectified.shape[0])/2))
        cv2.line(left_rectified,pt1 ,pt2 ,color=(0,0,255), thickness= 1 )
        pt1= (int((left_rectified.shape[1])/2),0)
        pt2= (int((left_rectified.shape[1])/2),int(left_rectified.shape[0]))
        cv2.line(left_rectified,pt1 ,pt2 ,color= (0,0,255) ,thickness= 1 )
        #grid for right frame
        pt1= (0,int((right_rectified.shape[0])/2))
        pt2= (int(right_rectified.shape[1]),int((right_rectified.shape[0])/2))
        cv2.line(right_rectified,pt1 ,pt2 ,color=(0,0,255), thickness= 1 )
        pt1= (int((right_rectified.shape[1])/2),0)
        pt2= (int((right_rectified.shape[1])/2),int(right_rectified.shape[0]))
        cv2.line(right_rectified,pt1 ,pt2 ,color= (0,0,255) ,thickness= 1 )
        new_img2 = np.hstack((left_rectified, right_rectified))

        #framL, framR = camie.capture_frame()
        ##############################
        sent_from, jpg_buffer = image_hub.recv_jpg()
        image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                                        colorspace='BGR')
        image_hub.send_reply(b'OK')            
            
        framL = image[:, :500]
        framR = image[:, 500:]
        
        left_rec, right_rec = rectification(framL, framR)

        Lgray = cv2.cvtColor(left_rec, cv2.COLOR_BGR2GRAY)
        Rgray = cv2.cvtColor(right_rec, cv2.COLOR_BGR2GRAY)

        '''
        #####MY OWN DISPARTIY######
        rows, cols = Lgray.shape[1], Lgray.shape[0]

        kernel = np.ones([block, block]) / block

        disparity_maps = np.zeros(
        [Lgray.shape[0], Lgray.shape[1], num_disparities])
        for d in range(0, num_disparities):
            # shift image
            translation_matrix = np.float32([[1, 0, d], [0, 1, 0]])
            shifted_image = cv2.warpAffine(
            Rgray, translation_matrix,
            (Rgray.shape[1], Rgray.shape[0]))
            # calculate squared differences
            SAD = abs(np.float32(Lgray) - np.float32(shifted_image))
            # convolve with kernel and find SAD at each point
            filtered_image = cv2.filter2D(SAD, -1, kernel)
            disparity_maps[:, :, d] = filtered_image

        disparity = np.argmin(disparity_maps, axis=2)
        disparity = np.uint8(disparity * 255 / num_disparities)
        disparity = cv2.equalizeHist(disparity)
        '''

        disparity = depth_map(Lgray,Rgray )
        filteredImg = cv2.applyColorMap(disparity, cv2.COLORMAP_JET )
        imgDisp = disparity
        imgCDisp = filteredImg
        #cv2.imwrite('./babloo_live_Anti_depth/babloo_L_with_color_Anti_D_Wout_Crop_image '+str(value)+' .jpg',imgCDisp )
        #cv2.imwrite('./babloo_live_Anti_depth/babloo_L_without_color_Anti_D_Wout_Crop_image '+str(value)+' .jpg',imgDisp )
        value = value + 1    
        #disp = np.hstack((disparity, filteredImg))
        #cv2.imshow("left window", frameL)
        #cv2.imshow("right window", frameR)
        cv2.imshow("image window", new_img)
        cv2.imshow("Rectified window", new_img2)
        cv2.imshow("Disparity Window",disparity)
        cv2.imshow("Colored Depth Window",filteredImg)
        cv2.imshow(name, img_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                #webiel.stop()
                #webier.stop()
                break
