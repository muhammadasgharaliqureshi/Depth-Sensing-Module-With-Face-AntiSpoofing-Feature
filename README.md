# Depth-Sensing-Module-With-Face-AntiSpoofing-Feature
This prototype uses Depth as a feature in Neural network and then train itselfs to do Classification between Depth face(i.e. Real Face) and Spoof Attack (i.e. Image, video). The concept used by it is,  first create a depth sensing stereo camera based module. Stereo Camera module can be designed by using two low cost cameras after that calibrate them in stereo configuration. After calibration and rectification Depth can be precepted by calculating Disparity between the two R.G.B images obtained( By stereo cameras). The obtained depth was tested on real faces and fake faces(i.e. image & video of a person face) and Data set was collected. after collecting Data set I trained a model (which I named Face Anti spoofeer) and used that model with a pretrained face Recognition model to validate a recognized persona as a real person or a spoof attack. Further more this project also includes controlling a security Door wirelessly via Arduino with the support of GUI that is running on Server and of course Raspberry pi was used as a client because it itself is a stereo module( in my case. You can make your stereo cameras by using your type of hardware). 
Following is the procedure of using this prototype......
First make sure you are registered for door security system, for this you would need to get your self recognized from face recognition. And for getting recognized by the face Recognizer, you will need to capture your face images with face centered in each image and save all those images in a folder with your name. Also make sure you have saved all your Images as a "JPEG", this is important guys because in JPEG image faces are easily detected due to higher quality of images.




Now!!!!  We will assemble our system by installing all the required Libraries.



following are the libraries that this project is using
==> python ------->(for linux use version >= 3.5, for windows I recommend you to use 3.5 as it supports ddlib in windows)
==> tkinter
==> OpenCv-----> (any version of opencv-contrib-python)
==> numpy
==> tensor flow -----> (version >2.2.0, I recommend to use 2.4.0 since all the trainig is done using that version.)
==> keras -------> (I used 2.4.3, you can use any version that supports tensorflow 2.2.0 or > 2.2.0)
==> dlib ---------> (use latest version)
==> face-recognition -----------> (use latest version)
==> imagezmq
==> simplejpeg
Before installing all these libraries we have to make suere that all the pre requisites files are being installed first.
so follow me step by step for installing all the pre requisites of libraries
sudo.....
..........
....
...


The Libraries required are all wrritten in requirenments.txt file. all you had to do is write following command in youe desired enviroment that you are using for python.
pip/pip3 install -r requirnments.txt

here you will use pip if you have pyhon2 and pip3 if you have python3 in linux and in windows it depends on no of versions that are installed in pyrhon. so make sure what version of python pip or pip3 is representing and use that w.r.t your taste.

