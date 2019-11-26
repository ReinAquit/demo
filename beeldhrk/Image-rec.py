# -- coding: utf-8 --
"""
Created on Wed May  8 11:31:22 2019

@author: Ottelien & Rein
"""

from __future__ import division
import numpy as np
import cv2
import copy
from matplotlib import pyplot as plt
from math import cos, sin

#Laat de afbeelding in een kleiner formaat zien
def image_show(name, image_show):
    #aspect_ratio is de verhouding tussen hoogte en breedte
    show_aspect_ratio = float(image_show.shape[1])/image_show.shape[0]
    show_size = 700
    image_show = cv2.resize(image_show, (int(show_size*show_aspect_ratio), show_size))
    cv2.imshow(name,image_show)

def image_load(index_image):

    #Loading in the image ------------------------------------------------------------------------------

    if(0):
        image_location = f"D:\_School HU\Jaar 3 blok 3\Beeldherkening\Krakeling\krakeling_ ({index_image}).jpg"
    if(0):
        image_location = f"D:\_School HU\Jaar 3 blok 3\Beeldherkening\stroopwafel\stroopwafel_ ({index_image}).jpg"
    if(0):
        image_location = f"D:\_School HU\Jaar 3 blok 3\Beeldherkening\ext\koekjes_ ({index_image}).jpg"
    if(1):
        image_location = f"D:\_School HU\Jaar 3 blok 3\Beeldherkening\Code\stroopw.jpeg"

    image = cv2.imread(image_location)

    if image is None:
        print("\nERROR: Could not open image\n")
        exit()

    # Bepaal afmetingen en aantal kleurkanalen
    height = image.shape[0]
    width  = image.shape[1]
    colors = image.shape[2]    

    # Afbeelding omzetten naar het HSV kleurdomein
    cHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return cHSV, image

#Functions --------------------------------------------------------------------------------------
def stroopwafel_detect(cHSV, image):

    #create colour mask
    lower_orange = np.array([6,50,100])
    upper_orange = np.array([39,255,255])
    mask_orange  = cv2.inRange(cHSV, lower_orange,  upper_orange)

    #Morphological transformation Dilation
    kernel = np.ones((3,3),np.uint8)
    mask_orange = cv2.dilate(mask_orange,kernel,iterations = 1)

    res_orange   = cv2.bitwise_and(image,image, mask= mask_orange)
    res_orage_gray = cv2.cvtColor(res_orange, cv2.COLOR_BGR2GRAY)

    lower_orange_T1 = np.array([6,50,50])
    upper_orange_T1 = np.array([9,255,255])

    mask_orange_T1  = cv2.inRange(cHSV, lower_orange_T1,  upper_orange_T1)

    #mask_orange_T1 = ~mask_orange_T1
    res_orange_T1   = cv2.bitwise_and(image,image, mask= mask_orange_T1)

    image_show("Orange cdgrffontours",res_orange)

    res_orage_gray = cv2.cvtColor(res_orange_T1, cv2.COLOR_BGR2GRAY)
    #show mask
    #image_show("Mask_orange_T1",mask_orange_T1)
    #show mask
    #image_show("res_orange_T1",res_orange_T1)
  
    oHSV = cv2.cvtColor(res_orange, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(cHSV,100,200)

    image_show("Canny",res_orange)


    ret, thresh = cv2.threshold(mask_orange_T1, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    image_contours = res_orange.copy()
    length_c = len(cnts)
    count_correct = 0
    print(length_c)

    for i in range (0, length_c):
        cnt = cnts[i]
        area = cv2.contourArea(cnt)
        if(area > 200):
            count_correct = count_correct + 1
            epsilon = 0.1*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            cv2.drawContours(image_contours, [approx], -1, (0,255,0), 15)
        
    print(count_correct)

    percentage = (count_correct/170)*100
    if(percentage > 99):
        percentage = 100

    print("Persentage stroopwafel:", percentage)

    #if count_correct >= 150:
    #    print("Is stroopwafel")
    #else:
    #    print("Is geen stroowafel")

    #show image
    image_show("Orange contours",image_contours)

def rozekoek_detect(cHSV, image):
    #Dit zijn de marges voor de kleurfilters HIERMEE TESTEN EN SPELEN
    lower_red1 = np.array([0,50,50])
    upper_red1 = np.array([5,255,255])
    lower_orange = np.array([6,50,100])
    upper_orange = np.array([39,255,255])
    lower_magenta = np.array([136,50,50])
    upper_magenta = np.array([172,255,255])
    lower_red2 = np.array([173,50,50])
    upper_red2 = np.array([179,255,255])

    amount = 0

    #Uiteindelijke maskers binaire waarden geven.
    mask_red1    = cv2.inRange(cHSV, lower_red1,    upper_red1)
    mask_orange  = cv2.inRange(cHSV, lower_orange,  upper_orange)
    mask_magenta = cv2.inRange(cHSV, lower_magenta, upper_magenta)
    mask_red2    = cv2.inRange(cHSV, lower_red2,    upper_red2)
    mask_rozekoek = cv2.bitwise_or(mask_red1,mask_red2)
    
    #Morphological transformation Closing
    kernel = np.ones((11,11),np.uint8)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)
    mask_magenta = cv2.morphologyEx(mask_magenta, cv2.MORPH_CLOSE, kernel)
    mask_rozekoek = cv2.morphologyEx(mask_rozekoek, cv2.MORPH_CLOSE, kernel)
    
    #Morphological transformation Dilation
    kernel = np.ones((3,3),np.uint8)
    mask_orange = cv2.dilate(mask_orange,kernel,iterations = 1)
    mask_magenta = cv2.dilate(mask_magenta,kernel,iterations = 1)
    mask_rozekoek = cv2.dilate(mask_rozekoek,kernel,iterations = 1)

    # Bitwise and is een vergelijking tussen twee afbeeldingen met de mogelijkheid
    # om er een masker overheen te leggen. Hier willen we alleen een masker over de
    # afbeelding leggen en daarom gebruiken we twee keer dezelfde afbeelding.
    #res_red1    = cv2.bitwise_and(image,image, mask= mask_red1)
    #res_orange  = cv2.bitwise_and(image,image, mask= mask_orange)
    #res_magenta = cv2.bitwise_and(image,image, mask= mask_magenta)
    #res_red2    = cv2.bitwise_and(image,image, mask= mask_red2)

    res_rozekoek = cv2.bitwise_and(image,image, mask= mask_rozekoek)

    #image_show("res_rozekoek",res_rozekoek)

    cimg = cv2.cvtColor(res_rozekoek,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cimg,(21,21),0)
    result = image.copy()
    circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,20,param1=127,param2=25,minRadius=75,maxRadius=0)
    
    if circles is not None:
        circles_sorted = sorted(circles[0],key=lambda x:x[2],reverse=True)
        circles_sorted = np.uint16(np.around(circles))
        circles_sorted = np.uint16(np.around(circles))
        circles_sorted = np.round(circles[0,:]).astype("int")
    
        #Draw the detected circles
        amount = 1
        x1 = circles_sorted[0,0]
        y1 = circles_sorted[0,1]
        r1 = circles_sorted[0,2]
        cv2.circle(result, (x1, y1), r1, (0, 255, 0), 3)
    
        boundery_x = r1
        boundery_y = r1
        for (x, y, r) in circles_sorted:
            if (x < x1-boundery_x or x > x1 + boundery_x) or (y < y1-boundery_y or y > y1 + boundery_y):  
                x1 = x
                y1 = y
                boundery_x = r
                boundery_y = r
                amount = amount + 1
                cv2.circle(result, (x, y), r, (0, 255, 0), 3)
    print("aantal roze koeken: ", amount)
                         
    image_show("circle", result)

def krakeling_detect(cHSV, image):

    if(0):
        image_show("orginele krakeling",image)

    lower_brown = np.array([10,50,50])
    upper_brown = np.array([23,255,255])
    mask_brown  = cv2.inRange(cHSV, lower_brown,  upper_brown)

    if(1):
        #Morphological transformation Dilation
        kernel = np.ones((9,9),np.uint8)
        mask_brown = cv2.dilate(mask_brown,kernel,iterations = 1)

    res_brown   = cv2.bitwise_and(image,image, mask= mask_brown)
    res_brown_gray = cv2.cvtColor(res_brown, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(mask_brown, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    image_contours = res_brown.copy()
    length_c = len(cnts)
    count_correct = 0
    #print(length_c)

    #load in image and make it gray, then use canny edge on it
    edges = cv2.Canny(mask_brown,100,150,25)

    for i in range (0, length_c):
        cnt = cnts[i]
        area = cv2.contourArea(cnt)
        if(area > 1):
            count_correct = count_correct + 1
            epsilon = 0.1*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            cv2.drawContours(image_contours, [approx], -1, (0,255,0), 15)
        
    print(count_correct)

    image_contour_krakeling,amount_triangle,amount_square,amount_rectangle,amount_pentagon,amount_circle = shape_detection(mask_brown, image ,6000)

    shapes = amount_triangle + amount_square + amount_rectangle + amount_pentagon + amount_circle

    percentage = (shapes/7)*100
    if(percentage > 100):
        percentage = 32
    print("Persentage krakeling = ", percentage)

    if(shapes > 3 and shapes < 7):
        print("Is een krakeling")

    print("shapes amount: ",shapes)

    #show the output image
    if(0):
        image_show("contour", image_contour_krakeling)
        image_show("res_brown",res_brown)
        image_show("contours",image_contours)

    if(0):
        image_show("mask_brown",mask_brown)
        image_show("canny edge",edges)

def brownie_detection(cHSV, image):
    image_contour = image.copy()
    shape = "unidentified"
    amount_rectangle = 0
    #Dit zijn de marges voor de kleurfilters HIERMEE TESTEN EN SPELEN
    lower_black = np.array([0,0,0])
    upper_black = np.array([180,255,80])
    lower_red1 = np.array([0,50,50])
    upper_red1 = np.array([5,255,255])
    lower_red2 = np.array([173,50,50])
    upper_red2 = np.array([179,255,255])

    height = image.shape[0]
    width  = image.shape[1]
    colors = image.shape[2]   

    #Uiteindelijke maskers binaire waarden geven.
    mask_black  = cv2.inRange(cHSV, lower_black,  upper_black)
    mask_red1    = cv2.inRange(cHSV, lower_red1,    upper_red1)
    mask_red2    = cv2.inRange(cHSV, lower_red2,    upper_red2)

    mask_brownie = cv2.bitwise_or(mask_black, mask_red1)
    mask_brownie = cv2.bitwise_or(mask_brownie, mask_red2)

    #Morphological transformation Closing
    kernel = np.ones((23,23),np.uint8)
    mask_brownie = cv2.morphologyEx(mask_brownie, cv2.MORPH_CLOSE, kernel)

    #Morphological transformation Dilation
    kernel = np.ones((3,3),np.uint8)
    mask_brownie = cv2.dilate(mask_brownie,kernel,iterations = 1)

        
    # Bitwise and is een vergelijking tussen twee afbeeldingen met de mogelijkheid
    # om er een masker overheen te leggen. Hier willen we alleen een masker over de
    # afbeelding leggen en daarom gebruiken we twee keer dezelfde afbeelding.
    #res_black   = cv2.bitwise_and(image,image, mask= mask_black)


    #laten zien van de results en maskers
    #image_show("mask_brownie",mask_brownie)
    #image_show("mask_black",mask_black)
    
    # find contours in the thresholded image
    ret, thresh = cv2.threshold(mask_brownie, 127, 255, 0)    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    
    for c in contours_sorted: 
        #print(cv2.contourArea(c))
        if(cv2.contourArea(c) > 100000):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            # if the shape has 4 vertices, it is either a square or
    		# a rectangle
            if len(approx) == 4:
            	# compute the bounding box of the contour and use the
            	# bounding box to compute the aspect ratio
            	(x, y, w, h) = cv2.boundingRect(approx)
            	#ar = w / float(h)
            
            	 
            	shape = "rectangle"    
            else:
                shape = "unkown"
            
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]))
            cY= int((M["m01"] / M["m00"]))
            cv2.drawContours(image_contour, c, -1, (0,255,0), 10)
            cv2.putText(image_contour, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 8)
            #print(shape) 
            if(shape == "rectangle"):
                black_image = np.zeros(shape=[height, width, 3], dtype=np.uint8)
                mask_rect = cv2.rectangle(black_image, (x, y), ((x+w),(y+h)), (255, 255, 255), -1)
                mask_rect = cv2.cvtColor(mask_rect, cv2.COLOR_BGR2GRAY)
                #image_show("mask_rect", mask_rect)
                # count white pixels in the total circle
                total_pix_rect = cv2.countNonZero(mask_rect)
                
                #Calculate how much percent is in red1 en orange mask
                count_black_pix = cv2.bitwise_and(mask_rect, mask_black)
                #image_show("count_black_pix", count_black_pix)
                nzPixBrownie = cv2.countNonZero(count_black_pix)
    
                #Calculate percentage of red1 and orange in the detected cookie
                percent_black = int((nzPixBrownie*100)/total_pix_rect)
                print ("percentage black: ", percent_black, "%")
                
                if(percent_black > 45):
                    amount_rectangle = amount_rectangle +1
                    
    print("Brownies: ",amount_rectangle)                
    image_show("contour", image_contour)
       
    return 0

def shape_detection(_image ,cHSV , min_size):
    image_contour = cHSV.copy()
    shape = "unidentified"
    amount_triangle = 0
    amount_square = 0
    amount_rectangle = 0
    amount_pentagon = 0
    amount_circle = 0
    
    # find contours in the thresholded image
    ret, thresh = cv2.threshold(_image, 127, 255, 0)    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    
    for c in contours_sorted: 
        #print(cv2.contourArea(c))
        if(cv2.contourArea(c) > min_size):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            # if the shape is a triangle, it will have 3 vertices
            if len(approx) == 3:
                shape="triangle"
           
            # if the shape has 4 vertices, it is either a square or
    		# a rectangle
            elif len(approx) == 4:
            	# compute the bounding box of the contour and use the
            	# bounding box to compute the aspect ratio
            	(x, y, w, h) = cv2.boundingRect(approx)
            	ar = w / float(h)
            
            	# a square will have an aspect ratio that is approximately
            	# equal to one, otherwise, the shape is a rectangleif ar >= 0.95 and ar <= 1.05:
            	shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"    
           
            # if the shape is a pentagon, it will have 5 vertices
            elif len(approx) == 5:
                shape = "pentagon"
     
    		# otherwise, we assume the shape is a circle
            else:
                shape = "circle"
            
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]))
            cY= int((M["m01"] / M["m00"]))
            cv2.drawContours(image_contour, c, -1, (0,255,0), 10)
            cv2.putText(image_contour, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 8)

            image_show("cont", image_contour)


            #print(shape) 
            if(shape == "triangle"):
                amount_triangle = amount_triangle +1
            if(shape == "square"):
                amount_square = amount_square +1
            if(shape == "rectangle"):
                amount_rectangle = amount_rectangle +1
            if(shape == "pentagon"):
                amount_pentagon = amount_pentagon +1
            if(shape == "circle"):
                amount_circle = amount_circle +1
    
    return image_contour,amount_triangle,amount_square,amount_rectangle,amount_pentagon,amount_circle

#Main code starting from here --------------------------------------------------------------------

i = 1

_cHSV = 0
_image = 0

while(i < 91):
#for i in range (1, 91):

    _cHSV, _image = image_load(i)

    stroopwafel_detect(_cHSV, _image)
    krakeling_detect(_cHSV, _image)
    rozekoek_detect(_cHSV, _image)
    brownie_detection(_cHSV, _image)

    print("image = ", i)

    i = i + 1
    cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()    

#End of code -x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-