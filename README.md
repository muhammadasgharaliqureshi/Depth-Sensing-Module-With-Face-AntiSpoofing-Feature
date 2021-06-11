# Depth-Sensing-Module-With-Face-AntiSpoofing-Feature
This prototype uses Depth as a feature in Neural network and then train itselfs to do Classification between Depth face(i.e. Real Face) and Spoof Attack (i.e. Image, video). The concept used by it is,  first create a depth sensing stereo camera based module. Stereo Camera module can be designed by using two low cost cameras after that calibrate them in stereo configuration. After calibration and rectification Depth can be precepted by calculating Disparity between the two R.G.B images obtained( By stereo cameras). The obtained depth was tested on real faces and fake faces(i.e. image & video of a person face) and Data set was collected. after collecting Data set I trained a model (which I named Face Anti spoofeer) and used that model with a pretrained face Recognition model to validate a recognized persona as a real person or a spoof attack. Further more this project also includes controlling a security Door wirelessly via Arduino with the support of GUI that is running on Server and of course Raspberry pi was used as a client because it itself is a stereo module( in my case. You can make your stereo cameras by using your type of hardware). 
Following is the procedure of using this prototype......
First make sure you are registered for door security system, for this you would need to get your self recognized from face recognition. And for getting recognized by the face Recognizer, you will need to capture your face images with face centered in each image and save all those images in a folder with your name. Also make sure you have saved all your Images as a "JPEG", this is important guys because in JPEG image faces are easily detected due to higher quality of images.




Now!!!!  We will assemble our system by installing all the required Libraries.



following are the libraries that this project is using


==> python ------->(for linux use version >= 3.5, for windows I recommend you to use 3.5 as it supports dlib in windows)


==> tkinter


==> OpenCv-----> (any version of opencv-contrib-python)


==> numpy


==> tensor flow -----> (version >2.2.0, I recommend to use 2.4.0 since all the training is done using that version.)


==> keras -------> (I used 2.4.3, you can use any version that supports tensorflow 2.2.0 or > 2.2.0)


==> dlib ---------> (use latest version)


==> face-recognition -----------> (use latest version)


==> imagezmq


==> simplejpeg



Before installing all these libraries we have to make suere that all the pre requisites files are being installed first.
You can find all the prerequisites on google.


################################################################################################################################################################################################################################################################################
                                                
                                                ###################
                                                HOW TO USE THE CODE?
                                                ####################
Here you will be seeing many folders but don't worry I will guide you step by step how to use each folder.

first make sure you have setup your client (that is your stereo module).
for callibrating your cameras you can use chessboard image that is inside Depth Sensing Module/Callibration of cameras/Callibration/


after setting up your stereo camera client module.
Open GUI folder, the main code is there that is main_Gui_code.py.
you can change you app graphics in tkinter_test.py and then paste your required new gui code inside the main_Gui_code.py

for training your own model you can use my pretrained model that is already using another pretrained model Resnet50.
The model is located in   Depth-Sensing-Module-With-Face-AntiSpoofing-Feature/GUI/Inferance_face_antispoofing-1618217329_____inferance_face_antispoof_classifier.h5


The model code can be found in   Depth-Sensing-Module-With-Face-AntiSpoofing-Feature/Face Anti Spoofing Network/Code and Dataset/

The accuracy of model that I have trained using a pretrained ResNet50 is located in  Depth-Sensing-Module-With-Face-AntiSpoofing-Feature/model (FACE AntiSpoofer) Accuracy.jpg


The Arduino code for opening and closing gate is located in   Depth-Sensing-Module-With-Face-AntiSpoofing-Feature/arduino gate/bluetooth_test_code/.
you would require HC05 bluetooth sensor to use with arduino.

                                               ###############
                                               HOW TO RUN CODE
                                               ###############
                                               
(Step1)   First place you image foler in known_faces folder that is located in    Depth-Sensing-Module-With-Face-AntiSpoofing-Feature/GUI/known_faces/



(Step 2)  After that run your server code That is main_Gui_code.py


(Step 3)  After that run client code that is located in   Depth-Sensing-Module-With-Face-AntiSpoofing-Feature/Client Code/

           If you want to use Arduino automated gate system then run faster_client.py
           
           other wise you can use new_faster_client_for_separate_discription.py for using it without arduino gate system

                          
                          
                          




                                            #######################
                                              Video Guide and link
                                            #######################
                               https://www.youtube.com/watch?v=vzztgHNkhVk 
                  
