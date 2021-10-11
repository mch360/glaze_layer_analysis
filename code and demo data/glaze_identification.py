# July042020 This code handles glaze layer identification and plots after image alignment

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.optimize import curve_fit
from scipy import stats
import range_checker
import matplotlib.ticker as mtick

### automatic append data fitting result to a summary excel file
# from openpyxl import load_workbook
import pandas as pd

## gauss gussian distribution fit
def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

###################### block setting ####################


###300C

x_start = 220
y_start = 150
spacer = 320
width = spacer

hue = [40,120]
value = [100,256]
centerize = 0



# settings, all loaded imgs are trimmed.


folder_name = '300C_plate'
folder_loc = '/%s'%(folder_name)


img_optics = cv2.imread('%s/OM_trim.png'%(folder_loc), flags=1)

img_height = np.loadtxt('%s/Height_%s_trim.txt'%(folder_loc,folder_name), delimiter=',')
img_height_color = cv2.imread('%s/Zygo_trim.png'%(folder_loc), flags=1)


max_height = np.nanmax(np.nanmax(img_height))
min_height = np.nanmin(np.nanmin(img_height))
print("overall max_height =", max_height)
print("overall min_height =", min_height)



## identify glaze layer with HSV ver of optics
img_optics_hsv = cv2.cvtColor(img_optics, cv2.COLOR_BGR2HSV)

OM_hue = img_optics_hsv[:,:,0]
OM_value = img_optics_hsv [:,:,2]

## hue and value optimization [min, max]:


Hue_mask = np.logical_and(OM_hue > hue[0], OM_hue <hue[1])
value_mask = np.logical_and(OM_value > value[0], OM_value < value[1])

glaze = np.logical_and(Hue_mask > 0.5, value_mask > 0.5)
glaze = glaze.astype(float)


## closing
kernel_close = np.ones((7,7),np.uint8)
closing = cv2.morphologyEx(glaze,cv2.MORPH_CLOSE,kernel_close )
glazed_layer = closing

#### wear scar range marker############

# set of rectangles
## [x,y] of left up corner and bottom right corner
rectangles = []
rectangles.append([[190, 0], [600, 600]])




### hightlight none glaze but high area
non_glaze_high =np.logical_and(glaze<0.5, img_height>5) ## img_height>0.655, img_height <0.660, 
non_glaze_high = non_glaze_high.astype(float);
# cv2.imshow("glazed_layer", glazed_layer)

print('glazed_layer shape = ( %i X %i )'%(glazed_layer.shape[1],glazed_layer.shape[0]))
# print('non_glazed_layer shape = ',non_glaze_high.shape)

# # ##########################################

img_optics_glaze_red = img_optics.copy()
red_trans = 0.3


# # ########################################## shade identified glaze layer region with red in OM

img_optics_glaze_red[:,:,2] = img_optics_glaze_red[:,:,2] * (1- red_trans * glazed_layer) + red_trans * glazed_layer*255
img_optics_glaze_red[:,:,0] = img_optics_glaze_red[:,:,0] * (1 - red_trans * glazed_layer)
img_optics_glaze_red[:,:,1] = img_optics_glaze_red[:,:,1] * (1 - red_trans * glazed_layer)




# # ########################################## Calculate glaze layer coverage ##################


x_end = x_start + spacer

y_end = y_start + width

cv2.rectangle(img_optics_glaze_red, (x_start-1, y_start-1), (x_end+1, y_end+1), (0, 255, 0), 1)
cv2.rectangle(img_optics, (x_start-1, y_start-1), (x_end+1, y_end+1), (0, 255, 0), 1)
cv2.rectangle(img_optics_glaze_blue, (x_start-1, y_start-1), (x_end+1, y_end+1), (0, 255, 0), 1)
cv2.rectangle(glazed_layer, (x_start-1, y_start-1), (x_end+1, y_end+1), (0, 255, 0), 1)
cv2.rectangle(img_height_color, (x_start-1, y_start-1), (x_end+1, y_end+1), (0, 0, 0), 1)


cv2.imshow('glaze_red_%i_%i_%i'%(hue[0],hue[1],value[0]),img_optics_glaze_red)



img_height_resized = img_height[y_start:y_end, x_start:x_end]
glazed_layer_resized = glazed_layer[y_start:y_end, x_start:x_end]

np.set_printoptions(threshold=np.inf)



height_glazed = img_height_resized[glazed_layer_resized > 0.5]  ## get result at location where glaze_layer_resized ==1
height_non_glazed = img_height_resized[glazed_layer_resized < 0.5] ## get result at location where glaze_layer_resized ==0
tot_height = img_height_resized[glazed_layer_resized > -1]


loc_max_height = np.nanmax(img_height_resized)
loc_min_height = np.nanmin(img_height_resized)
loc_avg = np.average(img_height_resized)
loc_std = np.std(img_height_resized)
glz_avg = np.average(height_glazed)
glz_std = np.std(height_glazed)
n_glz_avg = np.average(height_non_glazed)
n_glz_std = np.std(height_non_glazed)

print("overall_avg =",np.average(img_height_resized))
print("glz avg_height =", glz_avg)
print("glz std =", glz_std)
print("n_glz avg_height =", np.average(height_non_glazed))
print("n_glz std =", np.std(height_non_glazed))
## chi-square test

glaze_higher_than_avg = (height_glazed>=loc_avg).sum()
glaze_lower_than_avg = (height_glazed<loc_avg).sum()
non_glaze_higher_than_avg=(height_non_glazed>=loc_avg).sum()
non_glaze_lower_than_avg=(height_non_glazed<loc_avg).sum()

# print('glaze_high',glaze_higher_than_avg)
# print('glaze_low',glaze_lower_than_avg)
# print('non_glaze_high',non_glaze_higher_than_avg)
# print('non_glaze_los',non_glaze_lower_than_avg)
# #################################  Histgram
bin_num = 50

kwargs = dict(alpha=0.3, bins=bin_num)
print(np.shape(height_glazed))
print(np.shape(height_non_glazed))
glaze_coverage = np.shape(height_glazed)[0]/(spacer*width)
# glaze layer coverage
print('glaze layer coverage = ', glaze_coverage)

### plot overlap
fig1, ax1 = plt.subplots()

ax1.hist(height_glazed, label = 'glaze layer', stacked=True,color='red',**kwargs) 
ax1.hist(height_non_glazed, label = 'not glaze layer', stacked=True,color='#1f77b4',**kwargs)

ax1.set_xlabel('Relative Height($\mu$m)')
ax1.set_ylabel('Pixcel Count')

title = '%s, location x = %i, y = %i, spacer = %i' %(folder_name, x_start,y_start,spacer)
ax1.set_title(title)



ax1.legend(loc = 'upper left')
# print('glaze layer fit',params_g,'\n',sigma_g)  
# print('non glaze layer fit',params_n,'\n',sigma_n)
ax1.set_xlim(-60,40)
# ax1.set_ylim(0,1)  

# ######## save image #################
# plt.savefig('%s/%s_full coverage.png'%(folder_loc,title),)
# cv2.imwrite('%s/sq_red_%i_%i_%i_y%i.png'%(folder_loc, hue[0],hue[1],value[0],y_start),img_optics_glaze_red)
# cv2.imwrite('%s/sq_OM_y%i.png'%(folder_loc, y_start),img_optics)
# cv2.imwrite('%s/sq_Zygo_y%i.png'%(folder_loc, y_start),img_height_color)



##################### static T test ######################
t_test = stats.ttest_ind(height_non_glazed,height_glazed,equal_var = False)
print('t-test: t = %f, p = %f'%(t_test[0],t_test[1]))

plt.show()

cv2.waitKey(0)


