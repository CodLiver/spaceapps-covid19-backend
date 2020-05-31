import cv2,os
import json,random as rd
import scipy
import pandas as pd
import numpy as np
import seaborn as sns
import hvplot.pandas
import holoviews as hv
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image


ls=[each for each in os.listdir("./processed/") if ".jpg" in each]

# for each in ls:
#
#     img=cv2.imread(each)
#
#     lower_red = np.array([0,0,0])
#     upper_red = np.array([80,170,255])
#
#     mask = cv2.inRange(img, lower_red, upper_red)
#     res = cv2.bitwise_and(img,img, mask= mask)
#
#     # cv2.imshow("name",res)
#     cv2.imwrite("processed/red_"+each,res)

#
df=pd.read_csv("daily-cases-covid-19.csv")
newls=[]
# print(df.values)
for each in df.values:
    # 2,4
    splitted=each[0].split(",")
    newls.append([splitted[2][1:],splitted[4]])
    # print(splitted[2][1:],splitted[4])


"""
14march start

"""

imgBase=cv2.imread("./spain10.PNG")
imgBase=cv2.resize(imgBase,(1456,650))
"(650, 1456)"

compiletime=[]
for each in range(len(ls)):

    img=cv2.imread("./processed/"+ls[each])
    grayed=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold=grayed[np.where(grayed> 80)]
    print(newls[each][0],newls[each][1],cv2.countNonZero(threshold)//100)#
    compiletime.append([newls[each][0],newls[each][1],cv2.countNonZero(threshold)//100])
    # print(ls[each])
    res = cv2.addWeighted(imgBase,0.5,img,1,0)

    cv2.imshow("name",res)#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("gif/gif_"+each,res)
    cv2.waitKey(300)#30

# import imageio
# images = []
# for filename in ls:
#     images.append(imageio.imread("./gif/"+filename))
# imageio.mimsave('movie.gif', images)



"""

    img=cv2.imread(each)



    lower_red = np.array([0,0,0])
    upper_red = np.array([80,170,255])

    mask = cv2.inRange(img, lower_red, upper_red)
    res = cv2.bitwise_and(img,img, mask= mask)

    # cv2.imshow("name",res)
    cv2.imwrite("processed/red_"+each,res)
        # cv2.waitKey(0)


"""
"""

    # lower_red = np.array([0,0,50]) sota
    # upper_red = np.array([150,150,200])

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower_red = np.array([30,150,50])
    # upper_red = np.array([255,255,180])
    #
    # mask = cv2.inRange(hsv, lower_red, upper_red)


"""
