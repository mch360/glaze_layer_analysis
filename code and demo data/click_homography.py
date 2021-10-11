# tool of given a image pair, click 4 corner point to get ground truth homography

import cv2
import numpy as np
import warnings
import math


# settings

# load image as 3 channels color image
## image1: OM
## image2: Height map


folder_name = '300C_plate'
folder_loc = '/%s'%(folder_name)

## read-in height and normalize it to gray scale image

img2_original = cv2.imread('%s/org_res_color.jpeg'%(folder_loc), flags=1)
img2_original_noscaled = np.loadtxt('%s/%s.txt'%(folder_loc,folder_name), delimiter=',')

max_height = np.nanmax(np.nanmax(img2_original_noscaled))
min_height = np.nanmin(np.nanmin(img2_original_noscaled))
print("max_height =", max_height)
print("min_height =", min_height)
img2_original_scaled = (img2_original_noscaled - min_height) / (max_height - min_height) ## scale hight to 0-1, for display only

# ratio to scale height
height_scale = 0.25

img2 = cv2.resize(img2_original_scaled,(0,0),fx = height_scale, fy = height_scale)
img2_noscaled = cv2.resize(img2_original_noscaled,(0,0),fx = height_scale, fy = height_scale)


img1 = cv2.imread('%s/OM_%s.tif'%(folder_loc,folder_name), flags=1)

# saved wrapped img1 and resized img2
img1_warpped_save = '%s/OM_warped.png'%(folder_loc)

cv2.imwrite('%s/Height_resized.png'%(folder_loc), (img2 * 255).astype(np.uint8))
np.savetxt('%s/Height_%s_resized.txt'%(folder_loc, folder_name), img2_noscaled, delimiter=',') ## save scaled image(height resize), use this txt for future alignment

# Hest_filename = setfolder + '/rgb_nir.txt'

# magnifier scalec
magnifier_scale = 5.0
half_patch_size = 80

# flip comparision
flip = False
x_flip, y_flip = 0, 0


# cut image patch with black borders if part of patch out of image range
def cut_image_with_border(img, cx, cy, half_patch_size):
    img_pad = cv2.copyMakeBorder(img, half_patch_size, half_patch_size,
        half_patch_size, half_patch_size, cv2.BORDER_CONSTANT, value=(0,0,0))
    return img_pad[int(cy):int(cy)+2*half_patch_size+1, 
        int(cx):int(cx)+2*half_patch_size+1]


# image click callback
def callback_img1_click_point(event, x, y, flags, param):
    global img1_plot, imgpt
    # add one point to list
    if event == cv2.EVENT_LBUTTONDOWN:
        # add current position to list
        imgpt1.append((x, y))
        # display current point in the image
        cv2.circle(img1_plot, (int(x), int(y)), 1, (0,255,0), thickness=3)
        cv2.putText(img1_plot, str(len(imgpt1)), (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, 
            1, (0,255,0))
        # check list length, if > 4 give warning
        if len(imgpt1) > 4:
            warnings.warn('img1 point list length > 4', UserWarning)

def callback_img2_click_point(event, x, y, flags, param):
    global img2_plot, imgpt2
    # add one point to list
    if event == cv2.EVENT_LBUTTONDOWN:
        # add current position to list
        imgpt2.append((x, y))
        # display current point in the image
        cv2.circle(img2_plot, (int(x), int(y)), 1, (0,255,0), thickness=3)
        cv2.putText(img2_plot, str(len(imgpt2)), (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, 
            1, (0,255,0))
        # check list length, if > 4 give warning
        if len(imgpt2) > 4:
            warnings.warn('img2 point list length > 4', UserWarning)


# rect images magnifier callback
def callback_warp_magnifier(event, x, y, flags, param):
    global magnifier_scale, img1_warp, img2, flip, x_flip, y_flip

    # add one point to list
    if event == cv2.EVENT_LBUTTONDOWN:
        # set x and y from manual call
        if x == None or y == None:
            x, y = x_flip, y_flip
        else:
            x_flip, y_flip = x, y

        # refresh image 1 & 2 warp
        img1_warp_plot = img1_warp.copy()
        img2_plot = img2.copy()
        # plot a new magnifier box on image 1 & 2
        cv2.rectangle(img1_warp_plot, (int(x)-half_patch_size, int(y)-half_patch_size), 
            (int(x)+half_patch_size, int(y)+half_patch_size), (0,255,0), thickness=1)
        cv2.rectangle(img2_plot, (int(x)-half_patch_size, int(y)-half_patch_size), 
            (int(x)+half_patch_size, int(y)+half_patch_size), (0,255,0), thickness=1)
        cv2.imshow(imgwarpwin1, img1_warp_plot)
        cv2.imshow(imgwarpwin2, img2_plot)
        # enlarge image 1 & 2 selected area and show in seperate window
        patch1 = cut_image_with_border(img1_warp, x, y, half_patch_size)
        patch2 = cut_image_with_border(img2, x, y, half_patch_size)

        
        # overlap
        if flip:
            patch_overlap = patch1
        else:
            patch_overlap = patch2
        width = int(float(patch_overlap.shape[1]) * magnifier_scale)
        height = int(float(patch_overlap.shape[0]) * magnifier_scale)
        patch_overlap = cv2.resize(patch_overlap, (width, height), interpolation = cv2.INTER_NEAREST)
        cv2.imshow('overlap', patch_overlap)


# subpixel refine callback

# tmp click cache
subp_click_cache = []

def call_subpixel_refinement(event, x, y, flags, param):
    global subp_click_cache, imgsubp
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(subp_click_cache) < 2:
            # append click point list
            subp_click_cache.append((x,y))
            # plot cross +30/-30 is cross size
            cv2.line(imgsubp, (int(x)+30, int(y)), (int(x)-30, int(y)), (0,255,0))
            cv2.line(imgsubp, (int(x), int(y)+30), (int(x), int(y)-30), (0,255,0))
    


# #################################################
# image for click plot
img1_plot = img1.copy()
img2_plot = img2.copy()

# list of image points, 
imgpt1, imgpt2 = [], []
imgsubpt1, imgsubpt2 = [], []
# window names
imgwin1, imgwin2 = 'image 1', 'image 2' 
imgwarpwin1, imgwarpwin2 = 'image 1 warp', 'image 2 warp' 
imgsubpwin = 'sub-pixel click'

# setup the mouse callback function
cv2.namedWindow(imgwin1)
cv2.setMouseCallback(imgwin1, callback_img1_click_point)
cv2.namedWindow(imgwin2)
cv2.setMouseCallback(imgwin2, callback_img2_click_point)


# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow(imgwin1, img1_plot)
    cv2.imshow(imgwin2, img2_plot)
    key = cv2.waitKey(1)

    # if the 'r' key is pressed, reset the point list
    if key == ord("r"):
        print('clear')
        img1_plot = img1.copy()
        img2_plot = img2.copy()
        imgpt1, imgpt2 = [], []
    # if the 'c' key is pressed, break from the loop and calculate homography
    elif key == ord("c"):
        print('calculating ...')
        break
    # if the 'q' key is pressed, exit 
    elif key == ord("q"):
        print('exit')
        cv2.destroyAllWindows()
        exit(0)


# check if point list lengths are equal
if len(imgpt1) != len(imgpt2):
    raise ValueError('Input point length must be equal')


cv2.destroyWindow(imgwin1)
cv2.destroyWindow(imgwin2)


# #################################################
# refine the corner by enlarged views and precise click
cv2.namedWindow(imgsubpwin)
cv2.setMouseCallback(imgsubpwin, call_subpixel_refinement)

# iterate each corner pair
for i in range(len(imgpt1)):
    subp_click_cache = []

    # show patch pair
    patch1 = cut_image_with_border(img1, imgpt1[i][0], imgpt1[i][1], half_patch_size)
    patch2 = cut_image_with_border(img2, imgpt2[i][0], imgpt2[i][1], half_patch_size)
    
    # increase contrast for height images
    patch2 = patch2 - np.nanmin(np.nanmin(patch2))
    patch2 = patch2 / np.nanmax(np.nanmax(patch2))
    # convert to 8bit int
    patch2 = 255 * patch2 # Now scale by 255
    patch2 = patch2.astype(np.uint8)

    ## color enhancement_ for fine click
    patch2ColorCoded = cv2.applyColorMap(patch2, cv2.COLORMAP_HSV)
    imgsubp = np.concatenate((patch1, patch2ColorCoded), axis=1)  # patch 2 use patch2 instead of patch2Colorcoded for gray scale


    width = int(float(imgsubp.shape[1]) * magnifier_scale)
    height = int(float(imgsubp.shape[0]) * magnifier_scale)
    imgsubp = cv2.resize(imgsubp, (width, height), interpolation=cv2.INTER_NEAREST)

    print('width =', width)
    print('height =', height)

    # keep looping until the 'q' key is pressed
    while True:
        # wait for a keypress
        cv2.imshow(imgsubpwin, imgsubp)
        key = cv2.waitKey(1)

        # if the 'c' key is pressed, continue 
        if key == ord("c"):
            break

        # if the 'q' key is pressed, exit 
        if key == ord("q"):
            print('exit')
            cv2.destroyAllWindows()
            exit(0)

    # calculate sub-pixel accurary
    print(subp_click_cache)

    # swap left and right point if click right point first
    if (subp_click_cache[0][0] > subp_click_cache[1][0]):
        subp_click_cache = [subp_click_cache[1], subp_click_cache[0]]

    # calculate precise points
    xl = imgpt1[i][0] + (subp_click_cache[0][0] - magnifier_scale/2.0)/magnifier_scale\
        - half_patch_size
    yl = imgpt1[i][1] + (subp_click_cache[0][1] - magnifier_scale/2.0)/magnifier_scale\
        - half_patch_size
    xr = imgpt2[i][0] + (subp_click_cache[1][0] - magnifier_scale/2.0)/magnifier_scale\
        - half_patch_size - (2*half_patch_size+1)
    yr = imgpt2[i][1] + (subp_click_cache[1][1] - magnifier_scale/2.0)/magnifier_scale\
        - half_patch_size
    imgsubpt1.append((xl, yl))
    imgsubpt2.append((xr, yr))


cv2.destroyWindow(imgsubpwin)


# #################################################
# calculate homography using point list

## pick first 2 from imagesubpt1 and 2 for error calculation, save the rest for Homography calculation
imgsubpt1_err = imgsubpt1[0:2]
imgsubpt2_err = imgsubpt2[0:2]
imgsubpt1_Hest = imgsubpt1[2:]
imgsubpt2_Hest = imgsubpt2[2:]
# use 4 points exact method now
Hest, _ = cv2.findHomography(np.array(imgsubpt1_Hest), np.array(imgsubpt2_Hest), method=0)
print('Estimated Homography :', Hest)

### save Hest in seperate txt for future use. format saved is np.array
np.savetxt('%s/Homography_%s.txt'%(folder_loc,folder_name), Hest, delimiter=',')


# rect img1 to img2 frame using Hest, maintaining
w2, h2 = img2.shape[1], img2.shape[0]
img1_warp = cv2.warpPerspective(img1, Hest, (w2, h2))

# save warpped image
cv2.imwrite(img1_warpped_save, img1_warp)

# show rect images
cv2.imshow(imgwarpwin1, img1_warp)
cv2.imshow(imgwarpwin2, img2)
cv2.setMouseCallback(imgwarpwin1, callback_warp_magnifier)
cv2.setMouseCallback(imgwarpwin2, callback_warp_magnifier)

#print error, save error in err_list for txt saving, calculate avg err
err_list = []
for i in range(len(imgsubpt1_err)):
    testpt1 = imgsubpt1_err[i]
    testpt2 = imgsubpt2_err[i]

    testpt1_array = np.array([testpt1[0],testpt1[1],1])
    testpt2_array = np.dot(Hest, testpt1_array)
    err_x = testpt2[0] - testpt2_array[0]/testpt2_array[2]
    err_y = testpt2[1] - testpt2_array[1]/testpt2_array[2]
    err = (err_x**2 + err_y**2)**(1/2)
    err_list.append(err)
    print('err%i = '%(i),err)
avg_err = sum(err_list)/len(err_list)
print('avg error = ', avg_err)

# keep looping until the 'q' key is pressed
while True:
    # wait for a keypress
    key = cv2.waitKey(1)

    # if the 'q' key is pressed, exit 
    if key == ord("q"):
        print('exit')
        cv2.destroyAllWindows()
        exit(0)
    elif key == ord("f"):
        flip = not flip
        callback_warp_magnifier(cv2.EVENT_LBUTTONDOWN, None, None,
            None, None)


# close all open windows
cv2.destroyAllWindows()

