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
def getColorB(i):
    value = px.colors.cyclical.IceFire[i].lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3)) 
def subdivideGrid(n, grid):
    if n>1:
        ret = np.zeros((grid.shape[0]*n, grid.shape[1]*n))
        for x in range(0,grid.shape[0]):
            for y in range(0,grid.shape[1]):
                for i in range(0,n):
                    for j in range(0,n):
                        ret[x*nSub+i, y*nSub+j]=grid[x,y]
        return ret
    else:
        return grid

class material:
    k = float(0)
    rho = float(0)

    def __init__(self, _k, _rho):
        k = _k
        rho = _rho
class bound:
    tinf = float(0)
    h = float(0)

    def __init__(self, _h, _tinf):
        h = _h
        tinf = _tinf
#Arrays for boundary and material values
materials = []
boundaries = []
#User defined inputs
dx = float(input("Enter the spatial step, in unit length: "))
nx = int(input("Enter the number of horizontal cells: "))
ny = int(input("Enter the number of vertical cells: "))
numMat = int(input("Enter the number of materials present: "))
numBounds = int(input("Enter the number of convective boundaries present: "))
nSub = int(input("Enter number of desired subdivisions: "))

img = np.zeros((nx,ny,3), np.uint8)
cells = np.zeros((nx,ny,1))
currentMat = 1
currentB = 1

#Prompt input for materials
for m in range(0, numMat):
    rho = float(input("Enter the density of material {0}, unit mass / unit volume: ".format(str(m))))
    k  = float(input("Enter the conductivity of material {0}: ".format(str(m))))
    materials.append(material(k,rho))
for b in range(0, numBounds):
    h = float(input("Enter the convective coefficient of boundary {0}: ".format(str(b))))
    t_inf  = float(input("Enter the t-infinity of boundary {0}: ".format(str(b))))
    boundaries.append(bound(h,t_inf))
#Get user input for the initial state
#mouse callback function
def drawCell(event,x,y,flags,param):
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
cv2.setMouseCallback('image',drawCell)
print("Define materials. Press escape to end editing. Press M to switch material types.")
#Loop until the user is done
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    #If the user presses escape, done
    #If they press m, iterate materials
    if k == ord("m"):
        currentMat+=1
        if(currentMat>numMat):
            currentMat = 1
        print(currentMat)
    elif k == 27:
        break
#User finished, clear windows
cv2.destroyAllWindows()
cells = subdivideGrid(nSub, cells)
#Pad array for boundary conditions
cells = np.pad(cells, (1,1), 'constant', constant_values=(0,0))
boundaries = np.zeros(cells.shape)
img=cv2.resize(img,(img.shape[1]*nSub,img.shape[0]*nSub), interpolation=cv2.INTER_AREA)
img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT)
def drawBound(event,x,y,flags,param):


    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img,(ix,iy),(x,y),getColorB(currentB),-1)
            cv2.rectangle(boundaries,(ix,iy),(x,y),currentB,-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),getColorB(currentB),-1)
        cv2.rectangle(boundaries,(ix,iy),(x,y),currentB,-1)

#Setup the OPENCV window for boundaries
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', drawBound)
print("Define boundaries. Press escape to end editing. Press B to switch boundaries. 0 is adiabatic. ")
#Loop until the user is done
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    #If the user presses escape, done
    #If they press m, iterate materials
    if k == ord("b"):
        currentB+=1
        if(currentB>numBounds):
            currentB = 0
        print(currentB)
    elif k == 27:
        break
#User finished, clear windows
cv2.destroyAllWindows()

print("System initialized with size of " + str(cells.shape))
fig = px.imshow(boundaries)
fig.show()
fig = px.imshow(cells)
fig.show()

