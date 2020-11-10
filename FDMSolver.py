import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse
import scipy.sparse.linalg

useGS = False
drawing = False # true if mouse is pressed
ix,iy = -1,-1
def GS(a,x,b): #Gauss-Seidel iteration function for the equation a*x=b
    n = len(a)
    
    for j in range(0,n):
        v=b[j]
        
        for i in range(0,n):
            if(j!=i):
                v-=a[i,j]*x[i]
        x[j]=v/a[j,j]
    return x
def linToSq(a, stride): #Convert the 1-D array from the Gauss-Seidel solution into a 2-d array, for image use
    height = int(len(a)/stride)
    print(str(stride)+" by "+ str(height))
    n = np.zeros([stride, height])
    for x in range(0, stride):
        for y in range (0, height):
            n[x,y]=a[getN(x,y,stride)]
    return n

def getColor(i): #Get the color gradient for a material
    value = px.colors.cyclical.Twilight[i].lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3)) 
def getColorB(i): #Get the color gradient for a boundary
    value = px.colors.cyclical.IceFire[i].lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3)) 
def subdivideGrid(n, grid): #Subidivide the grid into n*n squares per original square
    if n>1:
        ret = np.zeros((grid.shape[0]*n, grid.shape[1]*n))
        for x in range(0,grid.shape[0]):
            for y in range(0,grid.shape[1]):
                for i in range(0,n):
                    for j in range(0,n):
                        ret[x*nSub+i, y*nSub+j]=grid[x,y]
        return ret
    else:
        ret = np.zeros((grid.shape[0], grid.shape[1]))
        for x in range(0,grid.shape[0]):
            for y in range(0,grid.shape[1]):
                ret[x, y]=grid[x,y]
        return ret
#Arrays for boundary and material values
rValues=[]
bRValues=[]
bTValues=[]
#User defined inputs
loadFile = input("Would you like to load a config? y/n ")
loadFile = loadFile=="y"
if not(loadFile):
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
        rValues.append(1/k)
    for b in range(0, numBounds):
        h = float(input("Enter the convective coefficient of boundary {0}: ".format(str(b))))
        t_inf  = float(input("Enter the t-infinity of boundary {0}: ".format(str(b))))
        bRValues.append(dx/h)
        bTValues.append(t_inf)
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
    print(cells.shape)
    boundaries = np.zeros(cells.shape)
    img=cv2.resize(img,(img.shape[1]*nSub,img.shape[0]*nSub), interpolation=cv2.INTER_AREA)
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT)
    #Event callback to draw boundaries
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
else:
    #Read from the config file
    path = input("Enter file path. Full folder and file name: ")

    file = open(path, "r")

    dx = float(file.readline().split("#")[0])
    nx = int(file.readline().split("#")[0])
    ny = int(file.readline().split("#")[0])
    numMat = float(file.readline().split("#")[0])
    numBounds = float(file.readline().split("#")[0])
    file.readline() #Materials
    for i in np.arange(0, numMat):
        rValues.append(float(file.readline()))
    file.readline() #BR
    for i in np.arange(0, numBounds):
        bRValues.append(float(file.readline()))
    file.readline() #BT
    for i in np.arange(0, numBounds):
        bTValues.append(float(file.readline()))

    file.readline()
    cells = np.zeros([nx,ny])
    for y in np.arange(0, ny):
        string = file.readline()
        string = string.strip()
        strings = string.split()
        for x in np.arange(0,len(strings)):
            cells[x,y]=int(strings[x])
    file.readline()
    file.readline()
    boundaries = np.zeros([nx,ny])
    for y in np.arange(0, ny):
        string = file.readline()
        string = string.strip()
        strings = string.split()
        for x in np.arange(0,len(strings)):
            boundaries[x,y]=int(strings[x])
    print(rValues)


print("System initialized with size of " + str(cells.shape))

#Generate equations
nodes = cells.shape[0]*cells.shape[1]
#Matrix of T's coefficients 
eqnArray = np.zeros([nodes,nodes])
#Matrix of constants to set the other matrix equal to 
bArray = np.zeros(nodes)
#4 directions to iterate when solving
offsets = [(1,0),(-1,0),(0,1),(0,-1)]
#Set the T-array to a default value
T=np.full(nodes,np.average(bTValues))
#Get the 1-d index of a point from the 2-d position
def getN(x,y, stride=cells.shape[0]):
    return int(x+y*stride)
shape = cells.shape
cells = cells.astype(int)
boundaries = boundaries.astype(int)

if not(loadFile):
    saveFile = input("Would you like to save a config? y/n ")
    saveFile = saveFile=="y"
else:
    saveFile = False
if saveFile:
    #Save file
    path = input("Enter file path. Full folder and file name: ")
    file = open(path, "w")
    file.write(str(dx)+"#dx \n")
    file.write(str(cells.shape[0])+"#nx \n")
    file.write(str(cells.shape[1])+"#ny \n")
    file.write(str(numMat)+"#nMat \n")
    file.write(str(numBounds)+"#numBounds \n")

    file.write("Materials: \n")
    for r in rValues:
        file.write(str(r) + "\n")
    file.write("Bounds, R: \n")
    for b in bRValues:
        file.write(str(b) + "\n")
    file.write("Bounds, T: \n")
    for b in bTValues:
        file.write(str(b) + "\n")    
    #cell materials
    file.write("CELL MATERIALS \n")
    for y in np.arange(0,shape[1]):
        for x in np.arange(0,shape[0]):
            file.write(str(cells[x,y])+" ")
        file.write("\n")
    file.write("\n BOUNDARY TYPES \n")
    for y in np.arange(0,shape[1]):
        for x in np.arange(0,shape[0]):
            file.write(str(boundaries[x,y])+" ")
        file.write("\n")
    file.close()


#Set up matrix in preparation for gauss-siedel
for y in np.arange(0,shape[1]):
    for x in np.arange(0,shape[0]):
        n=getN(x,y)
        # T is not known at this point
        #Sum(Q)=Sum(dT/R)=0
        #Add this equation to the matrix for each node
        #If the cell is valid
        if cells[x,y]!=0:
            for d in offsets:
                nX = int(x + d[0])
                nY = int(y + d[1])
                if cells[nX, nY] != 0:
                    r=rValues[cells[x,y]-1]+rValues[cells[nX,nY]-1]
                    eqnArray[n,n]+=float(1.00/r)
                    eqnArray[getN(nX, nY),n] = float(-1.00/r)
                else:
                    #If it's not a cell, it has to be a boundary
                    if boundaries[nX,nY]!=0:
                        #non-adiabiatic, so it must transfer heat convectively
                        r=rValues[cells[x,y]-1]+rValues[cells[nX,nY]-1]
                        eqnArray[n,n]+=float(1.00/r)
                        eqnArray[getN(nX, nY),n] = float(-1.00/r)
        else:
            eqnArray[n,n]=1
            if boundaries[x,y] != 0:
                bArray[n] = bTValues[boundaries[x,y]-1]

if useGS:
    #Apply the gauss-seidel iteration
    nGS=50
    for i in range(0,nGS):
        print ("\r"+str(100*i/nGS)+"%,", end='', flush=True)
        T = GS(eqnArray, T, bArray)
else:
    #T=np.linalg.solve(np.transpose(eqnArray), bArray)
    a=scipy.sparse.csr_matrix(np.transpose(eqnArray))
    T=scipy.sparse.linalg.spsolve(a, bArray)
t=linToSq(T,shape[0])
fig = go.Figure(data = go.Contour(z=t))
fig.show()
