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



# import imageio
#
# ls=[each for each in os.listdir("./gif/") if ".jpg" in each]
# images = []
# for filename in ls:
#     images.append(imageio.imread("./gif/"+filename))
# imageio.mimsave('movie.gif', images)

ls=[each for each in os.listdir("./processed/") if ".jpg" in each]


# for each in ls:# if path is "./"
#
#     img=cv2.imread("./o2/"+each)
#
#
#     lower_red = np.array([0,0,0])
#     upper_red = np.array([80,170,255])
#
#     mask = cv2.inRange(img, lower_red, upper_red)
#     res = cv2.bitwise_and(img,img, mask= mask)
#
#     # cv2.imshow("name",res)
#     cv2.imwrite("processed/red_"+each,res)# file/pref


df=pd.read_csv("daily-cases-covid-19_uk.csv")
df=df[df["iso_code"]=="GBR"][["date","new_cases"]].iloc[1:]
# df
newls=[]
# print(df.values)
for each in df.values:
    newls.append([each[0],each[1]])
#
#
#
# ls=[each for each in os.listdir("./processed/") if ".jpg" in each]
# imgBase=cv2.imread("./uk.png")
# imgBase=cv2.resize(imgBase,(1456,650))
# "(650, 1456)"
#
compiletime=[]
for each in range(len(ls)):

    img=cv2.imread("./processed/"+ls[each])
    # print(ls[each])
    grayed=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold=grayed[np.where(grayed> 80)]
    # print(newls[each][0],newls[each][1],cv2.countNonZero(threshold)//100)#
    compiletime.append([newls[each][0],newls[each][1],cv2.countNonZero(threshold)//100])
    res = cv2.addWeighted(imgBase,0.5,img,1,0)

    # cv2.imwrite("./gif/gif_"+ls[each],res)

    cv2.imshow("name",res)#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.waitKey(300)#30



t=[]
c=[]
n=[]

# compileDict={"date":[],"covid":[],"NO2":[]}
for each in compiletime:
    t.append(each[0])
    c.append(int(each[1]))
    n.append(each[2])


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Covid-19 Daily Case', color=color)
ax1.plot(t, c, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('NO2 levels', color=color)  # we already handled the x-label with ax1
ax2.plot(t, n, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.axvline("2020-03-23", 0, 1, label='Lockdown announced',color="xkcd:magenta")
plt.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.figsize=(300,150)
plt.show()
