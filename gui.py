import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from keras import models
from keras.utils import load_img
from keras.utils.image_utils import img_to_array
import tensorflow as tf 
import numpy as np

#load the trained model to classify sign
from keras.models import load_model
new_model = load_model('AI_Final.h5')

#dictionary to label all traffic signs class.
classes = ({ 1:'Bo canh cung',
            2:'Chau chau', 
            3:'Gian', 
            4:'Muoi', 
            5:'Sau buom', 
            6:'Ve', 
             })

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('NHẬN DẠNG SÂU HẠI')
top.configure(background='#2c2d2e')

label=Label(top,background='#2c2d2e', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    #image = Image.open(file_path)
    image = load_img(file_path,target_size=(150,150))
    image=img_to_array(image)
    image=image.astype('float32')
    image=image/255
    image=np.expand_dims(image,axis=0)
    #pred = model.predict_classes([image])[0]
    result = int(np.argmax(new_model.predict(image),axis =1))
    sign = classes[result+1]
    print(sign)
    label.configure(foreground='#0c5782', text=sign) 

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#0c5782', foreground='white',font=('roboto',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='',background='#2c2d2e')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload",command=upload_image,padx=10,pady=5)
upload.configure(background='#0c5782', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="NHẬN DẠNG SÂU HẠI",pady=20, font=('arial',20,'bold'))
heading.configure(background='#2c2d2e',foreground='#364156')
heading.pack()
top.mainloop()