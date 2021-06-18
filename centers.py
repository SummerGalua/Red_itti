import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
image = cv2.imread("C:/users/brand/onedrive/escritorio/Proof15.png")

#################################################################### Nuve de puntos ####################################################################################
#image = cv2.GaussianBlur(image,(9,9),1)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#################################################################### IDENTIFICACIÓN DE CENTROS #########################################################################

# Tenemos que aclarar el background para identificar los elementos que no coincidan con los valores de este
fondo = cv2.inRange(image,(0,0,0),(20,20,20))
num = cv2.bitwise_not(fondo)

#Eliminar ruido
kernel = np.ones((18,18),np.uint8)
num = cv2.morphologyEx(num,cv2.MORPH_OPEN,kernel)
num = cv2.morphologyEx(num,cv2.MORPH_CLOSE,kernel)

#Después detectamos los contornos 
contours,_ = cv2.findContours(num, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0,255,0), 1)

centro = []
for i in contours:
    #Calcular el centro a partir de los momentos
    momentos = cv2.moments(i)
    cx = int(momentos['m10']/momentos['m00'])
    cy = int(momentos['m01']/momentos['m00'])
 
    #Dibujar el centro
    cv2.circle(image,(cx, cy), 3, (0,0,255), -1)
    
    # Pasamos los centros a un arreglo de tuplas 
    centro.append((cx,cy))
    #Escribimos las coordenadas del centro
    #cv2.putText(image,"(x: " + str(cx) + ", y: " + str(cy) + ")",(cx+10,cy+10), font, 0.5,(255,255,255),1)
 
print(centro)
cv2.imshow("Detección",image)
cv2.waitKey()










































