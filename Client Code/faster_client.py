#######for bluetooth#######
import serial
port = serial.Serial("/dev/rfcomm0", baudrate = 9600)


#######For Image sending and message reciving
import sys
import socket
import traceback
import imagezmq
import simplejpeg
import numpy as np
from webcam_code import Camera

rpi_name     = socket.gethostname() 
camie = Camera(srcl = 0, srcr = 2)
jpeg_quality = 80                   # 0 to 100, higher is better quality
server_name = 'Write Hostname OF Your Server Name'
try:
    with imagezmq.ImageSender(connect_to='tcp://'+str(server_name)+':5555') as sender:
        while True:
            frameL, frameR = camie.capture_frame()
            image = np.hstack((frameL, frameR))
            
            jpg_buffer     = simplejpeg.encode_jpeg(image, quality=jpeg_quality, 
                                                    colorspace='BGR')
            reply = sender.send_jpg(rpi_name, jpg_buffer)
            #print('\n', reply)
            if reply == b'open':
                print("\nopened")
                port.write(str.encode('O'))
                rcv = port.readline()
                print("\n", rcv)
            elif reply == b'close':
                print("\nclosed")
                port.write(str.encode('C'))
                rcv = port.readline()
                print("\n", rcv)            
                
except (KeyboardInterrupt, SystemExit):
    pass                           
except Exception as ex:
    print('Python error with no Exception handler:')
    print('Traceback error:', ex)
    traceback.print_exc()
finally:
                      
    sys.exit()
