from tkinter import *
from tkinter import scrolledtext
from PIL import Image,ImageTk

window = Tk()

window.title("DSM app")
window.configure(background = '#23272a')
#window.geometry('1280x900')

def clicked():
    mssg = "Welcome " + txt.get()
    welcome_msg_lbl.configure(text = mssg)

def verify():
    #print('\nVerified')
    console_text_box.delete(1.0,END)
    console_text_box.insert(INSERT,'\n\nVerified\n')

main_name_lbl = Label(window, text = 'Depth Sensing Module Server', font=("Arial Bold", 20), bg = '#23272a', fg = 'white')
main_name_lbl.grid(column = 1, row  = 0)

welcome_msg_lbl  = Label(window, text = 'Welcome messgae box', font=("Arial Bold", 15), bg = '#23272a', fg = 'white')
welcome_msg_lbl.grid(column = 2, row = 1)

txt = Entry(window, width = 20, bg = '#2c2f33', fg = 'white')#, state = 'disabled')
txt.grid(column = 0, row = 1)
txt.focus()

welcome_btn = Button(window, text = 'Click to welcome', bg = '#2c2f33', fg = 'white', command = clicked)
welcome_btn.grid(column = 1, row = 3)

msg_name_label = Label(window, text = 'Enter your Name in Box', font=("Arial Bold", 10), bg = '#23272a', fg = 'white')
msg_name_label.grid(column = 0, row =2)

####Defining main frames Labels
framel_lbl = Label(window, width = 30, height = 10, bg = '#2c2f33')
framel_lbl.grid(column = 0 , row = 4)

live_depth_lbl = Label(window, width = 30, height = 10, bg = '#2c2f33')
live_depth_lbl.grid(column = 1, row = 4)

live_depth_color_tag_lbl = Label(window, width = 30, height = 1, bg = '#2c2f33')
live_depth_color_tag_lbl.grid(column = 1, row = 5)

open_door_button = Button(window, text = 'Click to Verify!!', bg = '#2c2f33', fg = 'white', command = verify)
open_door_button.grid(column = 2, row = 6)


console_text_box = scrolledtext.ScrolledText(window, width = 20, height = 10, bg = '#2c2f33', fg = 'white')
console_text_box.grid(column = 0, row = 7)
####methods to use it####
#To insert --->  txt.insert(INSERT,'You text goes here')
#To Delete ---> txt.delete(1.0,END)


gate_image_lbl = Label(window, width = 20, height = 10, bg = '#2c2f33')
gate_image_lbl.grid(column = 1, row = 7)


window.mainloop()