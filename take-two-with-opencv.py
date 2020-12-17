import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2

'''
For continuous-intensity images:
https://answers.opencv.org/question/189428/connectedcomponents-like-function-for-grayscale-image/

For binary images
ConnectedComponentsWithStats:
https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
'''

# each color is defined by [(Rmin, Rmax),(Gmin, Gmax),(Bmin, Bmax)]
colors_dict = {
    'yellow': [(200,255),(200,255),(0,50)],
    'cyan': [(0,130),(150,255),(150,255)],
    'pink': [(200,255),(0,60),(200,255)],
    'blue': [(0,60),(0,100),(200,255)],
}

img = io.imread('image1.jpg')

# io.imshow(img)
# plt.show()

'''
Fast way to isolate Muxo colors:
https://stackoverflow.com/questions/51997022/fast-way-to-apply-custom-function-to-every-pixel-in-image
'''

for color_name, color in colors_dict.items():
    R_RANGE = np.array([(color[0][0] <= r and r <= color[0][1]) for r in range(256)])
    G_RANGE = np.array([(color[1][0] <= g and g <= color[1][1]) for g in range(256)])
    B_RANGE = np.array([(color[2][0] <= b and b <= color[2][1]) for b in range(256)])

    # Calculate and apply mask
    # np.logical_and applies element-wise
    mask = (np.logical_and(np.logical_and(R_RANGE[img[:,:,0]], G_RANGE[img[:,:,1]])\
    , B_RANGE[img[:,:,2]]))

    # using masks is significantly faster than two Python for loops
    # whether_color is a single-channel image with the same 2d shape as img
    # each pixel is 1 if the corresponding pixel in img is the current MuxoColor
    # otherwise, a pixel in whether_color is 0
    whether_color = np.zeros(img.shape[0:2])
    whether_color[mask] = True
    whether_color[np.invert(mask)] = False

    '''
    Convert image to uint8 encoding so that it plays nicely with medianBlur:
    https://stackoverflow.com/questions/11337499/how-to-convert-an-image-from-np-uint16-to-np-uint8
    '''
    whether_color = cv2.convertScaleAbs(whether_color)

    # io.imshow(whether_color)
    # plt.show()
    # io.imsave('whether_color.png', whether_color)

    '''
    Blurring eliminates groups of MuxoColor pixels that are not circles around nests
    medianBlur ensures output values are either 1 or 0
    This preserves binary property of the image, needed for calling 
    connectedComponents
    '''
    blur = cv2.medianBlur(whether_color, 11)

    # io.imshow(blur)
    # plt.show()

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(blur)

    # io.imshow(labels)
    # plt.show()

    # segment 0 is the background
    num_circles = retval - 1
    print("There are", num_circles, "circles with color", color_name)

    for i in range(retval):
        print(stats[i,cv2.CC_STAT_TOP], stats[i,cv2.CC_STAT_LEFT])


