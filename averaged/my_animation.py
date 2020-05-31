import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import imageio
import glob
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.pylab as pyl
import os


# =============================================================================
# year = 365*24*60*60
# G_const = 6.67408e-11 #m3 kg-1 s-2
# G = G_const
# solar_mass = 2e30
# #print('\nCODE STARTS for real\n')
# AU = 1.5e11
# mu = G * solar_mass
# 
# bodies_name = 'bodies_1000_run_1.npy'
# ts_name = 'ts_1000_run_1.npy'
# outfile = 'outfile.png'
# 
# motions = np.load(bodies_name,allow_pickle=True)
# #print(motions)
# print('MOTIONS LOADED')
# ts_data = np.load(ts_name,allow_pickle=True)
# print('TS DATA LOADED')
# ts_len = ts_data[0]
# ts_list = ts_data[1]
# ts2_list = ts_data[2]
# total_t = ts_data[3]
# n_particles = ts_data[4]
# 
# def plot_output(bodies,limit,outfile):
#     fig = plt.figure(figsize=(7,7))
#     colours = ['r','b','g','y','m','c']
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.cla()
#     max_range,max_range_z = 0,0
#     for current_body in bodies: 
#         if current_body["name"] == "sun":
#             ax.scatter(current_body["x"][0], current_body["y"][0], current_body["z"][0],s=2e2,c='orange')
#         max_dim_x,max_dim_y,max_dim_z = max(current_body["x"]),max(current_body["y"]),max(current_body["z"])
#         if max_dim_x > max_range:
#             max_range = max_dim_x
#         if max_dim_y > max_range:
#             max_range = max_dim_y
#         if max_dim_z > max_range_z:
#             max_range_z = max_dim_z
#         if current_body["mass"] >1e5:
#             ax.plot(current_body["x"][:limit], current_body["y"][:limit], current_body["z"][:limit], c = 'red', label = current_body["name"])
#             ax.scatter(current_body["x"][limit], current_body["y"][limit], current_body["z"][limit], c = 'red')
#             ax.scatter(current_body["x"][0], current_body["y"][0], current_body["z"][0], c = 'red',marker='x')
# 
#         else:
#             ax.plot(current_body["x"][:limit], current_body["y"][:limit], current_body["z"][:limit], c = 'black', label = current_body["name"],alpha=0.05)
#             ax.scatter(current_body["x"][limit], current_body["y"][limit], current_body["z"][limit], c = 'black')
#             ax.scatter(current_body["x"][0], current_body["y"][0], current_body["z"][0], c = 'black',marker='x')
# 
#     ax.set_xlim([-max_range,max_range])    
#     ax.set_ylim([-max_range,max_range])
#     ax.set_zlim([-max_range,max_range])
#     #ax.set_axis_off()
#     if outfile:
#         plt.savefig(outfile)
#         plt.close()
#     else:
#         plt.show()
#         
# def plot_output_2d(bodies,limit,outfile):
#     fig = plt.figure(figsize=(14,7))
#     colours = ['r','b','g','y','m','c']
#     ax = fig.add_subplot(1,2,1)
#     ax2 = fig.add_subplot(1,2,2)
#     ax.cla()
#     max_range,max_range_z = 0,0
#     planets_done = []
#     a_list = []
#     for current_body in bodies:
#         try:
#             e2 = np.zeros(limit)
#             for i in range(limit):
#                 v_vec = np.array([current_body["xv"][i],current_body["yv"][i],current_body["zv"][i]])
#                 r_vec = np.array([current_body["x"][i],current_body["y"][i],current_body["z"][i]])
#                 L_vec = np.cross(r_vec,v_vec)
#                 mod_r = np.sqrt(np.dot(r_vec,r_vec))
#                 e_vec = np.subtract(np.cross(v_vec,L_vec)/mu,r_vec/mod_r)
#                 
#                 e2[i] = np.sqrt(np.dot(e_vec,e_vec))
#                 
#             xv2 = np.array(current_body["xv"])**2
#             yv2 = np.array(current_body["yv"])**2
#             zv2 = np.array(current_body["zv"])**2
#             
#             x2 = np.array(current_body["x"])**2
#             y2 = np.array(current_body["y"])**2
#             z2 = np.array(current_body["z"])**2   
#             
#             vel2 =     np.add(xv2,np.add(yv2,zv2))
#     
#             r = np.sqrt(    np.add(x2,np.add(y2,z2))      )
#             r[np.where(r==0)] = 1.5e11
#         except:
#             pass
#         #print('len r:',len(r))
#         try:
#             a = 1/(     2/r   -  vel2/(  6.67408e-11*2e30       )           )
#             a_fact = np.array(a)/np.mean(np.array(a))
#             a = a[:limit]
#             a_list.append(a[limit-1])
#         except:
#             a_fact = np.full(len(r),1)
#             pass
#         
#         
#         if current_body["name"] == "sun":
#             ax.scatter(current_body["x"][0], current_body["y"][0],s=2e2,c='orange')
#         if max(current_body["x"]) < 3e12 and max(current_body["y"]) < 3e12 and max(current_body["z"]) < 3e12:
#             max_dim_x,max_dim_y,max_dim_z = max(current_body["x"]),max(current_body["y"]),max(current_body["z"])
#         if max_dim_x > max_range:
#             max_range = max_dim_x
#         if max_dim_y > max_range:
#             max_range = max_dim_y
#         if max_dim_z > max_range_z:
#             max_range_z = max_dim_z
#         try:
#             if current_body["mass"] >1e5:
#                 if current_body["name"] not in planets_done:
#                 #ax.plot(current_body["x"][:limit], current_body["y"][:limit], c = 'red', label = current_body["name"])
#                     ax.scatter(current_body["x"][limit-1], current_body["y"][limit-1], c = 'red')
#                     ax.scatter(current_body["x"][0], current_body["y"][0], c = 'red',marker='x')
#                     planets_done.append(current_body["name"])
#                     try:
#                         ax2.plot(a,e2,c='red')
#                         ax2.scatter(r[limit-1],e2[limit-1],c='red',marker='x')
#                         ax2.scatter(a[limit-1],e2[limit-1],c='red')
#                     except:
#                         pass
#                     
#                 
#             elif e2[-1] >= 1 or a_fact[-1] < 0 :
#                 ax.plot(current_body["x"][:limit], current_body["y"][:limit], c = 'cyan', label = current_body["name"],alpha=0.5)
#                 ax.scatter(current_body["x"][limit-1], current_body["y"][limit-1], c = 'cyan')
#                 ax.scatter(current_body["x"][0], current_body["y"][0], c = 'cyan',marker='x')
#                 try:
#                     ax2.plot(a,e2,c='cyan')
#                     ax2.scatter(r[limit-1],e2[limit-1],c='cyan',marker='x')
#                     #print(a[limit-1],e2[limit-1])
#                     ax2.scatter(a[limit-1],e2[limit-1],c='cyan')
#                 except:
#                     pass
#                 
#             elif max(np.abs(np.full(len(r),1)-a_fact)) > 0.4:
#                 ax.plot(current_body["x"][:limit], current_body["y"][:limit], c = 'black', label = current_body["name"],alpha=0.25)
#                 ax.scatter(current_body["x"][limit-1], current_body["y"][limit-1], c = 'black')
#                 ax.scatter(current_body["x"][0], current_body["y"][0], c = 'black',marker='x')
#                 try:
#                     ax2.plot(a[::4],e2[::4],c='black')
#                     ax2.scatter(r[limit-1],e2[limit-1],c='black',marker='x')
#                     #print(a[limit-1],e2[limit-1])
#                     ax2.scatter(a[limit-1],e2[limit-1],c='black')
#                 except:
#                     pass
#                 
#             else:
#                 #ax.plot(current_body["x"][:limit], current_body["y"][:limit], c = 'blue', label = current_body["name"],alpha=0.05)
#                 ax.scatter(current_body["x"][limit-1], current_body["y"][limit-1], c = 'navy')
#                 ax.scatter(current_body["x"][0], current_body["y"][0], c = 'navy',marker='x')
#                 try:
#                     #ax2.plot(a,e2,c='navy')
#                     ax2.scatter(r[limit-1],e2[limit-1],c='navy',marker='x',alpha=0.1)
#                     #print(a,e2)
#                     #print(limit)
#                     #print(a[limit-1],e2[limit-1])
#                     ax2.scatter(a[limit-1],e2[limit-1],c='navy',alpha=0.5)
#                 except:
#                     pass
#         except:
#             pass
#             
#     
#     ax2.set_xlabel(r'Semi-major axis $a$, Distance from sun $r$ $(m)$')
#     ax2.set_ylabel(r'Eccentricity $e$')
#     
#     ax2.set_ylim([-0.2,1.2])
#     ax2.set_xlim([-0.2e11,3e12])
#     
#     a_mean = np.mean(np.array(a_list))
#     theta = np.linspace(0,2*np.pi,720)
#     
#     x = np.zeros(len(theta))
#     y = np.zeros(len(theta))
#     
#     for i in range(len(x)):
#         x[i] = np.cos(theta[i])*a_mean
#         y[i] = np.sin(theta[i])*a_mean
#     
#     ax.plot(x,y,c='orange')
#     ax.plot([a_mean,a_mean],[-1e20,1e20],c='orange')
#     ax.plot([-a_mean,-a_mean],[-1e20,1e20],c='orange')
#     ax.plot([-1e20,1e20],[a_mean,a_mean],c='orange')
#     ax.plot([-1e20,1e20],[-a_mean,-a_mean],c='orange')
#     
#     ax.set_xlim([-1.2*max_range,1.2*max_range])    
#     ax.set_ylim([-1.2*max_range,1.2*max_range])
#     #ax.set_axis_off()
#     if outfile:
#         plt.savefig(outfile)
#         plt.close()
#     else:
#         plt.show()
# 
#         
#         
# n_steps = len(motions[0]["x"])
# 
# for i in range(470,int((n_steps-1))):
#     
#     motions_copy = np.copy(motions)
#     print(i)
#     #try:
#     plot_output_2d(motions_copy,i,str(i)+'_test.png')
#        
# 
# # =============================================================================
# #     except:
# #         print(i,'failed')
# #         plt.close()
# #         pass
# # =============================================================================
# 
# 
#    
# 
# 
# 
# 
# 
# 
# 
# =============================================================================
filenames = glob.glob('*.jpeg')
print(filenames)
filenames = sorted(glob.glob('*.jpg'), key=os.path.getmtime)
print(filenames)
images = []

with imageio.get_writer('test_quick.mp4',fps = 25,mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# =============================================================================
# for filename in filenames:
#     os.remove(filename)
# =============================================================================
