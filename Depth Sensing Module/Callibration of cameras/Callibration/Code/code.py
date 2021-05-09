import numpy as np
import cv2
import glob
import argparse
import time
import matplotlib.pyplot as plt
my_gray = np.zeros((512,512,3), dtype = np.uint8)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpointsL = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane.
imgpointsR = []  # 2d points in image plane.
objpointsR = []  # 3d point in real world space



def calibrate(file_name, square_size, width, height):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    
    global my_gray
    #The data available in calibration data is below
    #images_selected = (4, 11, 16, 17, 18, 19, 23, 24, 27, 30, 32, 34, 39)
    #new data set in folder new calibration test is below
    #images_selected = (0,1,4,5,8,13,14,17,18,20,21,26,29,34,35,37,38,40,42,43,46,56,65,70,76,
    #                   77,94,95,98,104,106,110,121,122,123,130,132,134,136,137,143,145,148)
    #images_selected = (0,4,13,14,18,29,34,40,42,46,70,76,94,98,106,110,130,134)
    #images_selected = (0, 25)
    for value in range(0, 25):
        img = cv2.imread("/home/pi/Desktop/ Final prototype/The Callibration Test/Image Taker/data/"+str(file_name)+str(value)+" .jpg")
        #cv2.imshow("image "+str(file_name)+str(value), img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        my_gray = gray
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        print("\n\t\treturn value is(i.e. corner found?): ", ret)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            cv2.imwrite("Chessboard "+str(file_name)+str(value)+".jpg",img)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (my_gray.shape[1], my_gray.shape[0]), None, None)
    
    return [ret, mtx, dist,  rvecs, tvecs, objpoints, imgpoints]



def main():
    
    ####THESE IS FOR LEFT CAMERA###########
    retL, mtxL, distL, rvecsL, tvecsL, objpointsL, imgpointsL  = calibrate(file_name = "left_image ",square_size = 0.016, width=9, height=6)
    #print("\n\t\tLeft camera mtx = ",mtxL, "\t\t\t dist = ",distL, "\t\t\tObject ponts = ",objpointsL ,"\t\t\t\timagepoints = ",imgpointsL)
    #save_coefficients(mtx, dist, path = "/home/pi/Desktop/Prototype2/the calibration test/calibrations_results/left camera")
    #camera_matrix, dist_matrix = load_coefficients(path = "/home/pi/Desktop/Prototype  2/the calibration test/calibrations_results/left camera")
    #print("\n\t\t###########THIS ARE LEFT CAMERA RESULTS################\n")
    #print("\n\t\t\tCamera matrix: ",camera_matrix ,"\n\t\t\ttdist_matrix : ",dist_matrix)
    
    ########THIS IS FOR RIGHT CAMERA########
    retR, mtxR, distR, rvecsR, tvecsR, objpointsR, imgpointsR= calibrate(file_name = "right_image ", square_size = 0.016, width=9, height=6)
    #print("\n\t\tright camera mtx = ",mtxR, "\t\t\t dist = ",distR, "\t\t\tObject ponts = ",objpointsR ,"\t\t\t\timagepoints = ",imgpointsR)
    #camera_matrix, dist_matrix = load_coefficients(path = "/home/pi/Desktop/Prototype  2/the calibration test/calibrations_results/right camera")
    #save_coefficients(mtx, dist, path = "/home/pi/Desktop/Prototype2/the calibration test/calibrations_results/right camera")
    #print("\n\t\t###########THIS ARE RIGHT CAMERA RESULTS################\n")
    #print("\n\t\t\tCamera matrix: ",camera_matrix ,"\n\t\t\ttdist_matrix : ",dist_matrix)
    
    ###stereo calibration#####
    print((my_gray.shape[1], my_gray.shape[0]))
    stereocalibration_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    stereocalibration_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpointsL,imgpointsL,imgpointsR,mtxL,distL,mtxR,distR,(my_gray.shape[1], my_gray.shape[0]),criteria = stereocalibration_criteria, flags = stereocalibration_flags)
    print("\n\n\t\t\tthe value of T is : \t", T,"\n\n\t\t\tthe value of R is : \t", R, "\n\n\n\t\t\t the value of E is : \t", E, "\n\n\n the value of F is : \t",F)
    
    print("\n\tstereocalibration_retval = :\n\t\t", stereocalibration_retval,"\n\tcameraMatrix1 = :\n\t\t",cameraMatrix1,"\n\tdistCoeffs1 = :\n\t\t",distCoeffs1,"\n\tcameraMatrix2 :\n\t\t", cameraMatrix2, "\n\tdistCoeffs2 = : \n\t\t",distCoeffs2)
    #save_coefficients(cameraMatrix1, distCoeffs1, path ="/home/pi/Desktop/Prototype2/the calibration test/calibrations_results/left camera" )
    #save_coefficients(cameraMatrix2, distCoeffs2, path = "/home/pi/Desktop/Prototype2/the calibration test/calibrations_results/right camera")
    
    ########all the calibration is above

   
    lFrame = cv2.imread("/home/pi/Desktop/ Final prototype/The Callibration Test/Image Taker/left_image 5 .jpg")
    rFrame = cv2.imread("/home/pi/Desktop/ Final prototype/The Callibration Test/Image Taker/right_image 5 .jpg")
    w, h = lFrame.shape[:2] # both frames should be of same shape


    image_size = (w,h)
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
    print("\n\n\nThe value of Q is : ", Q,"\n")
    print("\n\n\nThe value of roi_left is : ", roi_left,"\n")
    print("\n\n\nThe value of roi_right is : ", roi_right,"\n")
    
    print("\n\n\n\t\t\t cameraMatrix1, distCoeffs1, R1, P1  ", cameraMatrix1,"\n", distCoeffs1,"\n\n\tR1 \n", R1, "\n\n\tP1 \n",P1)
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1)
    
    left_rectified = cv2.remap(lFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    print("\n\n\n\t\t\t cameraMatrix2, distCoeffs2, R2, P2  ", cameraMatrix2,"\n", distCoeffs2,"\n\n\tR2 \n", R2, "\n\n\tP2 \n",P2)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_32FC1)
    
    right_rectified = cv2.remap(rFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    #Fliping upside down the images
    left_rectified = cv2.flip(left_rectified, 0)
    right_rectified =  cv2.flip(right_rectified, 0)

    
    view = np.hstack([lFrame, rFrame])    
    rectView = np.hstack([left_rectified, right_rectified])
    #disparity = depth_map(left_rectified, right_rectified)
    cv2.imwrite("/home/pi/Desktop/Prototype2/Rectified_images/Rectified_image_left.jpg",left_rectified)
    cv2.imwrite("/home/pi/Desktop/Prototype2/Rectified_images/Rectified_image_right.jpg",right_rectified)
    
    cv2.imshow('view', view)
    cv2.imshow('rectView', rectView)
    #cv2.imshow('gray',disparity)
        
    # Wait indefinitely for any keypress
    cv2.waitKey(0)
    


    '''
    frames = [lFrame, rFrame]

    # Params from camera calibration
    '''
    '''
    cameraMatrix1 =  np.array([[739.23706917,   0.0       ,  276.63473707],
                      [  0.0         ,719.25183276, 217.08851191],
                      [  0.0          , 0.0,           1.0        ]]) 

    cameraMatrix2 = np.array([[826.14827551  , 0.0    ,     359.32819164],
                     [  0.0        , 827.99528183 ,218.69706181],
                     [  0.0         ,  0.0    ,       1.0        ]]) 
    
    camMats = [cameraMatrix1, cameraMatrix2]
    '''
    '''
    camMats = [mtxL, mtxR]
    '''
    '''
    distCoeffs1 =  np.array([[ 0.46352624, -2.99438371 ,-0.01122744 ,-0.02780513 ,8.45173922]])
    distCoeffs2 = np.array([[ 0.61305993 ,-2.87155974 ,-0.02722687 ,-0.01089135 , 6.36760082]])
    
    distCoeffs = [distCoeffs1, distCoeffs2]
    '''
    '''
    distCoeffs = [distL, distR]
    camSources = [0,1]
    for src in camSources:
        distCoeffs[src][0][4] = 0.0 # use only the first 2 values in distCoeffs

    # The rectification process
    newCams = [0,0]
    roi = [0,0]
    for src in camSources:
        newCams[src], roi[src] = cv2.getOptimalNewCameraMatrix(cameraMatrix = camMats[src], 
                                                               distCoeffs = distCoeffs[src], 
                                                               imageSize = (w,h), 
                                                               alpha = 0)



    rectFrames = [0,0]
    for src in camSources:
        rectFrames[src] = cv2.undistort(frames[src], 
                                        camMats[src], 
                                        distCoeffs[src])
    # See the results
    view = np.hstack([frames[0], frames[1]])    
    rectView = np.hstack([rectFrames[0], rectFrames[1]])

    
    cv2.imwrite("/home/pi/Desktop/Prototype2/Rectified_images/Rectified_image_left.jpg",rectFrames[0])
    cv2.imwrite("/home/pi/Desktop/Prototype2/Rectified_images/Rectified_image_right.jpg",rectFrames[1])
    #imgL = cv2.imread("/home/pi/Desktop/Prototype2/Rectified_images/Rectified_image_left.jpg",0)   
    #imgR = cv2.imread("/home/pi/Desktop/Prototype2/Rectified_images/Rectified_image_right.jpg",0)

   
    #cv2.imwrite('/home/pi/Desktop/Prototype1/Disparity Test neww 5.jpg', disparity)
    
    cv2.imshow('view', view)
    cv2.imshow('rectView', rectView)
    
        
    # Wait indefinitely for any keypress
    cv2.waitKey(0)
    '''
main()    
    
