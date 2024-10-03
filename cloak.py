import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

for i in range(200):
    ret,background = cap.read()

background = np.flip(background,axis=1)

while(1):
    ret, frame = cap.read()

    img = np.flip(frame,axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (35, 35),0)
    
    #blue
    lower = np.array([94, 80, 2])
    upper = np.array([126, 255, 255])

    #orange
    # lower1 = np.array([5, 100, 100])  # Lower bound for the first range of orange
    # upper1 = np.array([15, 255, 255]) # Upper bound for the first range of orange
    # lower2 = np.array([165, 100, 100]) # Lower bound for the second range of orange (for hues near 180 degrees)
    # upper2 = np.array([180, 255, 255]) 
    # mask1 = cv2.inRange(hsv,lower1,upper1)
    # mask2 = cv2.inRange(hsv,lower2,upper2)
    # mask=mask1+mask2

    #skin
    #lower = np.array([10, 100, 100])
    #upper = np.array([25, 255, 255])
    
    #black
    # lower = np.array([0, 0, 0])
    # upper = np.array([180, 255, 30])
    mask = cv2.inRange(hsv,lower,upper)

    mask = cv2.erode(mask,np.ones((7,7),np.uint8))
    mask = cv2.dilate(mask,np.ones((19,19),np.uint8))
    
    img[np.where(mask==255)] = background[np.where(mask==255)]

    cv2.imshow('MAGIC',img)
    
    if cv2.waitKey(1) == 27:
        break
        
cv2.destroyAllWindows()
cap.release()