import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join
import PIL

os.chdir('./output')

range_avg = 7

file_list = [f for f in os.listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]

print(file_list)

output_dir = os.getcwd()

os.chdir('../')

#os.mkdir('./averaged')
os.chdir('./averaged')



img_shapes = np.shape(np.asarray(PIL.Image.open(output_dir+'/'+file_list[0]))[:,:,0])

print('\nIMAGES ARE OF DIMENSTION', img_shapes)


print(img_shapes[0])
print(img_shapes[1],'\n')

for i in range(range_avg,len(file_list)-range_avg):
    img_array = np.zeros((img_shapes[0],img_shapes[1],2*range_avg+1))
    for n in range(-range_avg,range_avg):
        print('GETTING IMAGE', file_list[i+n])
        img_array[:,:,n] = np.mean(np.asarray(PIL.Image.open(output_dir+'/'+file_list[i+n])),axis=2)
        
    new_img = np.zeros(img_shapes)
    
    for ix in range(img_shapes[0]):
        for iy in range(img_shapes[1]):
            new_img[ix,iy] = np.mean(img_array[ix,iy,:])
    new_img = new_img.astype(np.uint8)
    new_img = PIL.Image.fromarray(new_img)
    new_img.save(file_list[i])
    
    
    
    
    
    

