import cv2
import tensorflow as tf
import numpy as np
from datetime import datetime


#Dictionary for Class Lables to the corresponding output layer for the CNN M1_D3
class_label_lookup={0:'Zero',1:'One',2:'Two',3:'Three',4:'Four',5:'Five',6:'A',7:'I',8:'Undefined'}

#Load the pretrained tensorflow Classification CNN model
loaded_model = tf.keras.models.load_model('Models/M1_D3')

#Initialising Input dimension on which the CNN was trained 
#Converts image to specified dimension before Classification
input_height = 64
input_width = 64

#Initialise Camera capture 
cap = cv2.VideoCapture(0)
width  = int(cap.get(3))  # float
height = int(cap.get(4)) # float
fps = cap.get(cv2.CAP_PROP_FPS)


#Initialise the Recognition area
rect_pt1 = (int(width*0.70),int(height*0.20))
rect_pt2 = (int(width),int(height*0.80))


#Display window Width and Height
disp_window_width = int(width*0.7)
disp_window_height = height

#Initialise Frame Count
count=0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,int(fps),(width,height))


#Initialise window number
window_no = 1

#Initialise Recognition Frequency after N Frames
N=6

#Initialise the last N deteions to be tracked
last_n = 10
last_n_detections = [np.NaN]*last_n


#Load App Pictures
sayhi = cv2.imread('Images/five-gesture.png',cv2.IMREAD_UNCHANGED)
bgr = sayhi[:,:,:3]
sayhi_width,sayhi_height = int(disp_window_width/1.5),int(disp_window_width/5)
sayhi = cv2.resize(bgr,(sayhi_width,sayhi_height))
sayhi_height_offset = int(disp_window_height/2)-int(sayhi_height/2)
sayhi_width_offset = int(disp_window_width/2)-int(sayhi_width/2)

order_menu = cv2.imread('Images/Window2_Order_Menu.png',cv2.IMREAD_UNCHANGED)
bgr = order_menu[:,:,:3]
order_menu_width,order_menu_height = int(disp_window_width),int(disp_window_width/1.5)
order_menu = cv2.resize(bgr,(order_menu_width,order_menu_height))
order_menu_height_offset = int(disp_window_height/2)-int(order_menu_height/2)
order_menu_width_offset = int(disp_window_width/2)-int(order_menu_width/2)


oops_menu_width,oops_menu_height = int(disp_window_width),int(disp_window_width/1.5)

oops_one = cv2.imread('Images/Oops_one.png',cv2.IMREAD_UNCHANGED)[:,:,:3]
oops_one = cv2.resize(oops_one,(oops_menu_width,oops_menu_height))

oops_two = cv2.imread('Images/Oops_Two.png',cv2.IMREAD_UNCHANGED)[:,:,:3]
oops_two = cv2.resize(oops_two,(oops_menu_width,oops_menu_height))

oops_three = cv2.imread('Images/Oops_Three.png',cv2.IMREAD_UNCHANGED)[:,:,:3]
oops_three = cv2.resize(oops_three,(oops_menu_width,oops_menu_height))

oops_four = cv2.imread('Images/Oops_Four.png',cv2.IMREAD_UNCHANGED)[:,:,:3]
oops_four = cv2.resize(oops_four,(oops_menu_width,oops_menu_height))

oops_menu_height_offset = int(disp_window_height/2)-int(oops_menu_height/2)
oops_menu_width_offset = int(disp_window_width/2)-int(oops_menu_width/2)

#Initialise menu frame count and kill time
oops_count = 0
order_menu_count = 0
kill_after_seconds = 5
kill_after_frame = fps*kill_after_seconds

while(True):

    ret, frame = cap.read()
    
    frame = cv2.flip(frame,1)
    
    # Running the Classifier at every n'th' Frame
    if count%N==0:
        count=0
        
        #Cropping the Region of Interest to feed to the CNN
        cropped = frame[rect_pt1[1]:rect_pt2[1] , rect_pt1[0]:rect_pt2[0]]
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        '''
        #Implementing Image Enhancement Techniques here
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])       
        cropped_gray_sharp = cv2.filter2D(cropped_gray, -1, kernel)    
        cropped_gray = cropped_gray_sharp
        '''  
        cv2.rectangle(frame, rect_pt1, rect_pt2, color=(0,255,0))
        cropped_gray_inpt = cv2.resize(cropped_gray,(input_height,input_width))
        cropped_gray_inpt = cropped_gray_inpt/255
        cropped_gray_inpt = cropped_gray_inpt.reshape(-1,input_width,input_height,1)

        predicted_class = np.argmax(loaded_model.predict(cropped_gray_inpt),axis=1)
        class_name = class_label_lookup[predicted_class[0]]
        last_n_detections = last_n_detections+[class_name]
        last_n_detections = last_n_detections[-last_n:]
        #print(last_n_detections)

    else:
        cv2.rectangle(frame, rect_pt1, rect_pt2, color=(255,0,0))
    # Frame Count Incrementer
    count = count+1

    cv2.namedWindow('Hand Gesture Project', cv2.WINDOW_NORMAL)
    
        
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    
    #Show the whole frame
    # Blurring the rest of the window
    blur_region1 = frame[0:height, 0:int(width*0.70)]
    blur_region2 = frame[0:int(height*0.20), int(width*0.70):width]
    blur_region3 = frame[int(height*0.80)+1:height, int(width*0.70):width]
    
    blur_frame = cv2.GaussianBlur(blur_region1, (51,51), -5)
    frame[0:height, 0:int(width*0.70)] = blur_frame
    
    blur_frame = cv2.GaussianBlur(blur_region2, (51,51), -5)
    frame[0:int(height*0.20), int(width*0.70):width] = blur_frame
    
    blur_frame = cv2.GaussianBlur(blur_region3, (51,51), -5)
    frame[int(height*0.80)+1:height, int(width*0.70):width] = blur_frame
    
    #Adding the text for detection
    frame = cv2.putText(frame, 'Class:'+str(class_name), org=(rect_pt1[0],rect_pt1[1]+25),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(255,0,0))
    
    #Adding Datetime
    now = datetime.now()
    now_date_text = now.strftime("%d %B,%Y")
    now_time_text = now.strftime("%H : %M : %S")
    welcome_text = 'Welcome'
    date_textsize = cv2.getTextSize(now_date_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 2)[0]
    time_textsize = cv2.getTextSize(now_time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 2)[0]
    welcome_textsize = cv2.getTextSize(welcome_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    
    frame = cv2.putText(frame, now_date_text, org=(10,10+int(date_textsize[1]/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(255,0,0))
    frame = cv2.putText(frame, now_time_text, org=(width-int(time_textsize[0])-10,10+int(time_textsize[1]/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(255,0,0))
    frame = cv2.putText(frame, welcome_text, org=(int((width/2)-welcome_textsize[0]/2),10+int(welcome_textsize[1]/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(255,0,0))
    
    if window_no == 1:
        frame[sayhi_height_offset:sayhi_height_offset+sayhi_height,sayhi_width_offset:sayhi_width_offset+sayhi_width] = frame[sayhi_height_offset:sayhi_height_offset+sayhi_height,sayhi_width_offset:sayhi_width_offset+sayhi_width]*0.1+sayhi*0.9
        if last_n_detections==['Five']*last_n:
            window_no = 2
            
    elif window_no == 2:
        order_menu_count +=1
        frame[order_menu_height_offset:order_menu_height_offset+order_menu_height,order_menu_width_offset:order_menu_width_offset+order_menu_width] = frame[order_menu_height_offset:order_menu_height_offset+order_menu_height,order_menu_width_offset:order_menu_width_offset+order_menu_width]*0.1+order_menu*0.9 
        if last_n_detections==['Zero']*last_n:
            window_no = 1
        elif order_menu_count == kill_after_frame:
            order_menu_count = 0
            window_no = 1
        elif last_n_detections in [['One']*last_n,['Two']*last_n,['Three']*last_n,['Four']*last_n]:
            window_no = 3
            detected_item = last_n_detections[0]
            
    elif window_no == 3:
        oops_count +=1
        if detected_item == 'One':
            oops_pic = oops_one
        elif detected_item == 'Two':
            oops_pic = oops_two
        elif detected_item == 'Three':
            oops_pic = oops_three
        elif detected_item == 'Four':
            oops_pic = oops_three
            
        frame[oops_menu_height_offset:oops_menu_height_offset+oops_menu_height,oops_menu_width_offset:oops_menu_width_offset+oops_menu_width] =  frame[oops_menu_height_offset:oops_menu_height_offset+oops_menu_height,oops_menu_width_offset:oops_menu_width_offset+oops_menu_width]*0.1+oops_pic*0.9
            
        if oops_count==kill_after_frame:
            oops_count = 0
            window_no = 1
    #Show the Frame
    cv2.imshow('Hand Gesture Project',frame)
    out.write(frame)
    '''
    #Show the cropped and processed frame only
    cropped_gray = cv2.putText(cropped_gray, text, org=(10,20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(255,0,0))
    cv2.imshow('Hand Gesture Project',cropped_gray)
    '''    
    
    

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

