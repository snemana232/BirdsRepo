#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io
import sys
sys.setrecursionlimit(2000000)


# In[2]:


'''
!pip install -U -q PyDrive
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# choose a local (colab) directory to store the data.
local_download_path = os.path.expanduser('~/birds_data')
try:
  os.makedirs(local_download_path)
except: pass

# 2. Auto-iterate using the query syntax
#    https://developers.google.com/drive/v2/web/search-parameters
file_list = drive.ListFile(
    {'q': "'1-kILh7-OMeCYny7rlycroie7scKvmDQ-' in parents"}).GetList()

for f in file_list:
  # 3. Create & download by id.
  # print('title: %s, id: %s' % (f['title'], f['id']))
  # fname = os.path.join(local_download_path, f['title'])
  # print('downloading to {}'.format(fname))
  # f_ = drive.CreateFile({'id': f['id']})
  # f_.GetContentFile(fname)
  if f['title'] == '20180620_BISC_BirdFlight_SOKE_0362_copy.jpg':
    print('title: %s, id: %s' % (f['title'], f['id']))
    fname = os.path.join(local_download_path, f['title'])
    print('downloading to {}'.format(fname))
    f_ = drive.CreateFile({'id': f['id']})
    f_.GetContentFile(fname)
'''


# In[3]:


#pathnames = glob.glob('/root/birds_data/20180620_BISC_BirdFlight_SOKE_0362_copy.jpg', recursive=True)
#print(pathnames)
#img = mpimg.imread(pathnames[0]) #The processed image
img = io.imread('image1.jpg')
plt.imshow(img)


# In[4]:


#arr = skimage.io.imread('20180620_BISC_BirdFlight_SOKE_0362_copy.jpg') In case we need more speed? Messy to work with 3D array though


# In[5]:


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


# In[6]:


row1 = img[0]


# In[7]:


# THIS APPROACH DOES NOT WORK (it finds a global xmin and xmax, ymin and ymax
# for each color)
# i.e. two yellow circles in the same column will be marked together in one really 
# tall bounding box
# approach without recursion (hopefully faster because of how NumPy uses C)

# each color in the list is of the form [[rmin, rmax],[gmin,gmax],[bmin,bmax]]
'''
for color in color_list:
  whether_color = color[0][0] < img[:,:0] and color[0][1] > img[:,:,0] and \
  color[1][0] < img[:,:,1] and color[1][1] > img[:,:,1] and \
  color[2][0] < img[:,:,2] and color[2][1] > img[:,;.2]

  # list of [x,y] pairs that have the given color
  array_of_locations_of_color = np.argwhere(whether_color), 
  [xmax,ymax] = np.amax(array_of_locations_of_color, axis=0)
  [xmin,ymin] = np.amin(array_of_locations_of_color, axis=0)
'''


# In[8]:


boundingCoordinates = [] #[[rmin, rmax], [cmin, cmax]], [[rmin, rmax], [cmin, cmax]]
                            #Circle 1                      #Circle 2

img = img[325 : 637, 1257 : 1601]
visited_map = np.zeros(img.shape[:2], dtype=bool)
curr_bounds = [[len(img) + 1, -1],[len(img[0]) + 1, -1]]
#new_visited_map = np.zeroes(img.shape)
'''
print(img.shape)
print(len(img) + 1)
print(-1)
print(len(img[0]) + 1)
print(-1)
'''
def find_circles(img):
    numCircles = 0
    first = False
    for i, row in enumerate(img):
        for j, triple in enumerate(row):
            color_func = isMuxoColor(triple)
            #is Muxo color returns a function
            if (color_func):
                visited_map = np.zeros(img.shape, dtype=bool) # makes this an array of all False
                #new_visited_map = np.zeroes(img.shape)
                curr_bounds = [[len(img) + 1, -1],[len(img[0]) + 1, -1]] # will become [[rmin,rmax],[cmin,cmax]]
                # boundingCoordinates.append(curr_bounds)
                print("starting coordinates: " + str(i) + ", " + str(j))
                #new_visited_map[i][j] = [255, 255, 255]
                boundingCoordinates.append(find_coordinates(img, i, j, color_func))
                rmin = boundingCoordinates[-1][0][0]
                rmax = boundingCoordinates[-1][0][1]
                cmin = boundingCoordinates[-1][1][0]
                cmax = boundingCoordinates[-1][1][1]
                img[rmin : rmax, cmin: cmax] = [0, 0, 0]
                #to account for handwriting, filter out bounding boxes where 
                #(ymin-ymax) is very different from (xmin-xmax)
                first = True
            if (first):
                break
        if(first):
            break
        
        

        
#the [row, column] rgb triple we have flagged as part of a circle

# walks along contiguous segment of image with given color
# returns [[rmin,rmax],[cmin,cmax]] for that segment -> will be used to
# give coordinates of the bounding box later
def find_coordinates(img, row_id, col_id, color_func):
    visited_map[row_id][col_id] = True
    if visited_map[row_id-1][col_id] and visited_map[row_id][col_id-1] and visited_map[row_id+1][col_id] and visited_map[row_id][col_id+1] and visited_map[row_id+1][col_id+1] and visited_map[row_id-1][col_id-1] and visited_map[row_id-1][col_id+1] and visited_map[row_id+1][col_id-1]:
        # print("first")
        return curr_bounds 
    if not color_func(img[row_id][col_id]): # out of circle
        # print("second")
        return curr_bounds
    #new_visited_map[row_id][col_id] = [255, 255, 255]
    
    # need to change to <= and >= instead of < and >
    if row_id < curr_bounds[0][0]:
        curr_bounds[0][0] = row_id
        # call on rmin, cmin, cmax
        find_coordinates(img, row_id - 1, col_id, color_func)
        find_coordinates(img, row_id, col_id + 1, color_func)
        find_coordinates(img, row_id, col_id - 1, color_func)
    elif row_id > curr_bounds[0][1]:
        curr_bounds[0][1] = row_id
        # call on rmax, cmin, cmax
        find_coordinates(img, row_id + 1, col_id, color_func)
        find_coordinates(img, row_id, col_id + 1, color_func)
        find_coordinates(img, row_id, col_id - 1, color_func)
    elif col_id < curr_bounds[1][0]:
        curr_bounds[1][0] = col_id
        # call on rmin, rmax, cmin
        find_coordinates(img, row_id, col_id - 1, color_func)
        find_coordinates(img, row_id - 1, col_id, color_func)
        find_coordinates(img, row_id + 1, col_id, color_func)
    elif col_id > curr_bounds[1][1]:
        curr_bounds[1][1] = col_id
        # call on rmin, rmax, cmax
        find_coordinates(img, row_id, col_id + 1, color_func)
        find_coordinates(img, row_id - 1, col_id, color_func)
        find_coordinates(img, row_id + 1, col_id, color_func)
    else:
        return curr_bounds
    # call on diagonals
    find_coordinates(img, row_id - 1, col_id - 1, color_func)
    find_coordinates(img, row_id + 1, col_id + 1, color_func)
    find_coordinates(img, row_id - 1, col_id + 1, color_func)
    find_coordinates(img, row_id + 1, col_id - 1, color_func)
    # print(curr_bounds)
    '''
    if (row < boundingCoordinates[i][0][0]): #currx < (xmin, y)
    boundingCoordinates[i][0] = [row, col]
    if (row > boundingCoordinates[i][1][0]): #currx > (xmax, y)
    boundingCoordinates[i][1] = [row, col]
    if (col < boundingCoordinates[i][2][1]): #curry < (x, ymin)
    boundingCoordinates[i][2] = [row, col]
    if (col > boundingCoordinates[i][3][1]): #curry > (x, ymax)
    boundingCoordinates[i][3] = [row, col]
    '''
    # print("third")
    return curr_bounds
  


# In[9]:


#Overlay rectangles defined by coordinates onto copy of original image


# Another approach: https://www.codingame.com/playgrounds/38470/how-to-detect-circles-in-images
# 
# Canney edge detection --> find strong edges (colored circles will be especially
# strong) --> step through circles (parameterized by r and theta) and add points
# that are shared by Canney strong edges --> this (should) eliminate handwriting
# 
# 
# --> /then/ draw bounding boxes (by taking xmin, xmax, ymin, ymax) for each
# parameterized circle
# 
# 

# In[10]:


find_circles(img)


# Our function is tail-recursive, so this should work:
# https://stackoverflow.com/questions/13591970/does-python-optimize-tail-recursion
# 
# (we just need to make it a while loop)
# 
# And this might end up being important (passing a list to simultaneous recursive calls): https://docs.python.org/3/library/copy.html

# In[11]:


print(boundingCoordinates)


# open CV function: identify colors in an image by specifying their rgb boundaries
# https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
# 

# In[12]:


print(visited_map)


# In[13]:


plt.imshow(img)


# In[14]:


plt.imshow(visited_map)


# In[ ]:




