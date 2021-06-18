import numpy as np 
import cv2 
import imutils
from numpy.core.fromnumeric import resize, shape


#INPUT
image = cv2.imread("C:/users/brand/onedrive/escritorio/ittimagen/itti.png")

#Escala de grises

def gray(image):
        return(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

# CANALES DE COLOR 

def Colores(image):

        B = image[:,:,0]
        G = image[:,:,1]
        R = image[:,:,2]
        Y = cv2.subtract(cv2.add(G,R),cv2.subtract(G,R))
        # cv2.imshow("Canales",np.hstack([B,G,R]))
        return([B,G,R,Y])
   
# FUNCIONES DE INTENSIDAD 

# Elementos de la pirámide Gaussiana 
def levels_gauss(image):

        niv = []
        for i in range(9):
            Pira = cv2.GaussianBlur(image,(15, 15),i)
            niv.append(Pira)
        return niv

# FILTRO DE GABOR (ORIENTACIÓN)

def levels_gabor(image):

    theta_0=[]
    theta_45=[]
    theta_90=[]
    theta_135=[]

    for sigma in range(9):
        for theta in [0,np.pi/4,np.pi/2,3*np.pi/4]:

            if theta == 0:
                
                g_kernel = cv2.getGaborKernel((15, 15), sigma = sigma,theta = theta,lambd=0.9,gamma=1,ktype=cv2.CV_64F)  
                filtered_img = cv2.filter2D(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_8UC3, g_kernel)
                theta_0.append(filtered_img)

            if theta == np.pi/4:

                g_kernel = cv2.getGaborKernel((15, 15),sigma = sigma,theta = theta,lambd=0.8,gamma=1,ktype=cv2.CV_64F)  
                filtered_img = cv2.filter2D(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_8UC3, g_kernel)
                theta_45.append(filtered_img)

            if theta == np.pi/2:

                g_kernel = cv2.getGaborKernel((15, 15), sigma = sigma,theta = theta,lambd=0.7,gamma=1,ktype=cv2.CV_64F)  
                filtered_img = cv2.filter2D(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_8UC3, g_kernel)
                theta_90.append(filtered_img)  

            if theta == 3*np.pi/4:

                g_kernel = cv2.getGaborKernel((15,15),sigma = sigma,theta = theta,lambd=0.5,gamma=1,ktype=cv2.CV_64F)  
                filtered_img = cv2.filter2D(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_8UC3, g_kernel)
                theta_135.append(filtered_img) 

    return[theta_0,theta_45,theta_90,theta_135]

def N(image):
    image = cv2.multiply(image, ((image.max()-image.mean())/image.max())**2)
    return image

###################################################### FEATURE MAPS #####################################################################################################

def Intensity_maps(image):

    Intensity = []
    for c in [0,1,2]:
        for s in [5,6,7]:
            I = cv2.subtract(levels_gauss(gray(image))[c],levels_gauss(gray(image))[s])
            Intensity.append(I)
    return Intensity

def color_maps(image):
    
    B = levels_gauss(Colores(image)[0])
    G = levels_gauss(Colores(image)[1])
    R = levels_gauss(Colores(image)[2])
    Y = levels_gauss(Colores(image)[3])
    RG = []
    BY = []
    for c in [1,2,3]:
        for s in [5,6,7]:
            RG.append(cv2.subtract(cv2.subtract(R[c],G[c]),cv2.subtract(G[s],R[s])))
            BY.append(cv2.subtract(cv2.subtract(B[c],Y[c]),cv2.subtract(Y[s],B[s])))

    return[RG,BY]

def orientation_maps(image):
     
    Orientation = []
    
    for theta in range(4):
        for c in [1,2,3]:
            for s in [5,6,7]:    
                I = cv2.subtract(levels_gabor(image)[theta][c],levels_gabor(image)[theta][s])
                Orientation.append(I)
    
    return Orientation
   
###################################################### Saliency Map #####################################################################################################
def C(image): 

    C = cv2.add(N(color_maps(image)[0][0]),N(color_maps(image)[0][1]))
    
    for i in range(2,len(color_maps(image)[0])):
        for j in range(2):
            C = cv2.add(C,N(color_maps(image)[j][i]))
    return C

def I(image):
    
    I = cv2.add(N(Intensity_maps(image)[0]),N(Intensity_maps(image)[1]))
    
    for i in range(2,len(Intensity_maps(image))):
        I = cv2.add(I,N(Intensity_maps(image)[i]))
    return I

def O(image):

    O = cv2.add(N(orientation_maps(image)[0]),N(orientation_maps(image)[1]))
       
    for i in range(2,len(orientation_maps(image))):
        O = cv2.add(O,N(orientation_maps(image)[i]))
    return O

def saliency_map(image):

    Saliency = N(cv2.add(cv2.add(I(image),C(image)),O(image)))
    return Saliency

if __name__ == "__main__":

    cv2.imshow("Proof",levels_gauss(image)[0])
    cv2.waitKey(0)















