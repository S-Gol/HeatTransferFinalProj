import cv2
import numpy as np
import plotly.express as px
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

def getColor(i):
    value = px.colors.cyclical.Twilight[i].lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3)) 

def setup():
    dx = float(input("Enter the spatial step, in m: "))
    nx = int(input("Enter the number of horizontal cells: "))
    ny = int(input("Enter the number of vertical cells: "))
    numMat = int(input("Enter the number of materials present: "))
    img = np.zeros((nx,ny,3), np.uint8)
    cells = np.zeros((nx,ny,1))
    currentMat = 0

    #Get user input for the initial state
    #mouse callback function
    def draw(event,x,y,flags,param):
        global ix,iy,drawing,mode
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                
                cv2.rectangle(img,(ix,iy),(x,y),getColor(currentMat),-1)
                cv2.rectangle(cells,(ix,iy),(x,y),currentMat,-1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img,(ix,iy),(x,y),getColor(currentMat),-1)
            cv2.rectangle(cells,(ix,iy),(x,y),currentMat,-1)

    #Setup the OPENCV window
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image',draw)
    print("Definite materials. Press escape to end editing. Press M to switch materials.")
    #Loop until the user is done
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        #If the user presses escape, done
        if k == ord("m"):
            currentMat+=1
            if(currentMat>=numMat):
                currentMat = 0
            print(currentMat)
        elif k == 27:
            break
    #User finished, clear windows
    cv2.destroyAllWindows()
setup()