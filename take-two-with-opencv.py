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


# circleColors = [red, crimson, maroon, blue, cyan, magenta, yellow, purple, pink]
# yellow range = (200-255, 200-255, 0-50)
# cyan range = (0-60, 200-255, 200-255)
# pink range = (200-255, 0-60, 200-255)
# blue range = (0-60, 0-100, 200-255)
def isYellow(triple):
  return (triple[0] <= 255 and triple[0] >= 200 and triple[1] <= 255 and triple[1] >= 200 and triple[2] <= 50 and triple[2] >= 0)

def isCyan(triple):
  return (triple[0] <= 130 and triple[0] >= 0 and triple[1] <= 255 and triple[1] >= 150 and triple[2] <= 255 and triple[2] >= 150)

def isPink(triple):
  return (triple[0] <= 255 and triple[0] >= 200 and triple[1] <= 60 and triple[1] >= 0 and triple[2] <= 255 and triple[2] >= 200)

def isBlue(triple):
  return (triple[0] <= 60 and triple[0] >= 0 and triple[1] <= 100 and triple[1] >= 0 and triple[2] <= 255 and triple[2] >= 200)
    
def isMuxoColor(triple):
  if isPink(triple):
    return isPink
  elif isCyan(triple):
    return isCyan
  elif isBlue(triple):
    return isBlue
  elif isYellow(triple):
    return isYellow
  else:
    return False


img = io.imread('image1.jpg')

print(img.shape[0:2])

io.imshow(img)
plt.show()

'''
Fast way to isolate Muxo colors:
https://stackoverflow.com/questions/51997022/fast-way-to-apply-custom-function-to-every-pixel-in-image
'''

R_RANGE = np.array([(0 <= r and r <= 130) for r in range(256)])
G_RANGE = np.array([(150 <= g and g <= 255) for g in range(256)])
B_RANGE = np.array([(150 <= b and b <= 255) for b in range(256)])

# Calculate and apply mask
# np.logical_and applies element-wise
mask = (np.logical_and(np.logical_and(R_RANGE[img[:,:,0]], G_RANGE[img[:,:,1]])\
, B_RANGE[img[:,:,2]]))

# using masks is significantly faster than two Python for loops
whether_cyan = np.zeros(img.shape[0:2])
whether_cyan[mask] = True
whether_cyan[np.invert(mask)] = False

'''
Convert image to uint8 encoding so that it plays nicely with medianBlur:
https://stackoverflow.com/questions/11337499/how-to-convert-an-image-from-np-uint16-to-np-uint8
'''
whether_cyan = cv2.convertScaleAbs(whether_cyan)

# whether_cyan = np.zeros(img.shape[0:2])
# for i, row in enumerate(img):
#   for j, px in enumerate(row):
#     whether_cyan[i,j] = int(isCyan(px))
# whether_cyan = io.imread('whether_cyan.png')

io.imshow(whether_cyan)
plt.show()
# io.imsave('whether_cyan.png', whether_cyan)

# blur = cv2.blur(whether_cyan, (10,10))
blur = cv2.medianBlur(whether_cyan, 11)

io.imshow(blur)
plt.show()


retval, labels, stats, centroids = cv2.connectedComponentsWithStats(blur)

# print(labels)

io.imshow(labels)

print(retval)

for i in range(retval):
  print(stats[i,cv2.CC_STAT_TOP], stats[i,cv2.CC_STAT_LEFT])


plt.show()