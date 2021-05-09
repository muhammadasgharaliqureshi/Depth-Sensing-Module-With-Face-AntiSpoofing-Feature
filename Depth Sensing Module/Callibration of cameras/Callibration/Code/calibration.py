import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpointsL = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane.
imgpointsR = []  # 2d points in image plane.
objpointsR = []  # 3d point in real world space


def calibrate(file_name, no_of_imgs , square_size, width, height):
                """ Apply camera calibration operation for images in the given directory path. """
                # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
                objp = np.zeros((height*width, 3), np.float32)
                objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

                objp = objp * square_size

                # Arrays to store object points and image points from all the images.
                objpoints = []  # 3d point in real world space
                imgpoints = []  # 2d points in image plane.

                global my_gray
                for value in range(0, no_of_imgs):
                        img = cv2.imread(file_name+str(value)+' .jpg')
                        #cv2.imshow("image "+str(file_name)+str(value), img)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        my_gray = gray
                        # Find the chess board corners
                        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
                        print("\n\t\treturn value is(i.e. corner found?): ", ret, "   for test frame:  ", value)

                        # If found, add object points, image points (after refining them)
                        if ret:
                            objpoints.append(objp)

                            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                            imgpoints.append(corners2)

                            # Draw and display the corners
                            #img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
                            #cv2.imwrite("Chessboard "+str(file_name)+str(value)+".jpg",img)
                           
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (my_gray.shape[1], my_gray.shape[0]), None, None)

                return [ret, mtx, dist,  rvecs, tvecs, objpoints, imgpoints]


class Calibration:
        def __init__(self, left_image_path = '/home/pi/Desktop/ Final prototype/The Callibration Test/Image Taker/data/left_image ' ,
                             right_image_path = '/home/pi/Desktop/ Final prototype/The Callibration Test/Image Taker/data/right_image ',
                             no_of_images = 24):
               
               self.Lpath = left_image_path
               self.Rpath = right_image_path
               self.images = no_of_images + 1

        


        def start(self):
                
                ####THESE IS FOR LEFT CAMERA###########
                retL, mtxL, distL, rvecsL, tvecsL, objpointsL, imgpointsL  = calibrate(self.Lpath, self.images,square_size = 0.016, width=9, height=6)
              
                ########THIS IS FOR RIGHT CAMERA########
                retR, mtxR, distR, rvecsR, tvecsR, objpointsR, imgpointsR= calibrate(self.Rpath, self.images, square_size = 0.016, width=9, height=6)
              
                ###stereo calibration#####
                stereocalibration_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
                stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
                stereocalibration_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpointsL,imgpointsL,imgpointsR,mtxL,distL,mtxR,distR,(my_gray.shape[1], my_gray.shape[0]),criteria = stereocalibration_criteria, flags = stereocalibration_flags)
                return cameraMatrix1, distCoeffs1,cameraMatrix2, distCoeffs2, R, T, (my_gray.shape[1], my_gray.shape[0])
