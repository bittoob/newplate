from array import array
from turtle import width
import streamlit as st
import cv2
import pytesseract

import numpy as np
from PIL import Image
from PIL import *


#st.set_page_config(layout="wide")



title_container = st.container()
col1, col2 = st.columns([5, 10])

with title_container:
    with col1:
        st.image('./download.png')
    with col2:
        st.title("Vehicle Number Plate Detection")
    




def gray_image(image_file):
    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    return gray

def blur_image(gray):
    blur = cv2.bilateralFilter(gray, 11, 90, 90)
    return blur

def contours(blur,image_file):
    edges = cv2.Canny(blur, 30, 200)
    cnts, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image_file.copy()
    _ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)
    cnts = sorted(cnts, key=cv2.contourArea, reverse = True)[:30]
    image_copy = image_file.copy()
    _ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)

    plate = None
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(edges_count) == 4:
            x,y,w,h = cv2.boundingRect(c)
            plate = image_file[y:y+h, x:x+w]
            break

    return plate

def binarySearch (arr, x): 
  
    l = 0
    r = len(arr)
    while (l <= r):
        m = l + ((r - l) // 2)
 
        res = (x == arr[m])
 
        # Check if x is present at mid
        if (res == 0):
            return m - 1
 
        # If x greater, ignore left half
        if (res > 0):
            l = m + 1
 
        # If x is smaller, ignore right half
        else:
            r = m - 1
 
    return -1

def main_loop():
    
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

   
      
    st.text("Original Image")
    st.image([original_image])

    #convert original image into gray image
    g = gray_image(original_image)
    # st.write("Gray image")
    # st.image([g])

    #convert gray image into blur image
    b = blur_image(g)
    # st.write("Blur image")
    # st.image([b])

    #find contours
    p = contours(b,original_image)
    # st.image([p])
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    txt = pytesseract.image_to_string(p)
    
    st.subheader("Detection of Number Plate from Vehicle Image ")
    st.subheader(txt)

    ar  = ["MH 20 EE 7598","WH20 EJ 0365,", "MH1407T8831,","=MHO2FE8819","TN 87 A 3980","-GJO5JA1 143","KL 26 45009"]


    result = binarySearch(ar,txt)
    
    if result == -1: 
        st.subheader("\n\nThe Vehicle is not allowed to visit." ) 
        st.image('./stop (1).jpg')
    else: 
        st.subheader("\n\nThe Vehicle is allowed to visit.")
        st.image('./stop (2).jpg')
    
    


if __name__ == '__main__':
    main_loop()