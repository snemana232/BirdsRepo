#!/usr/bin/env python
# coding: utf-8

# In[57]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io
import sys
import random
sys.setrecursionlimit(10000000)


# In[59]:

# renamed version of below image
img = io.imread('image1.jpg')

calls_counter = 0


# In[60]:


#arr = skimage.io.imread('20180620_BISC_BirdFlight_SOKE_0362_copy.jpg') In case we need more speed? Messy to work with 3D array though


# In[61]:

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


# In[63]:


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

# Produces image to check whether color functions are properly defined
'''
only_cyan = np.zeros(img.shape, dtype=np.uint8)
for i,row in enumerate(img):
    for j,px in enumerate(row):
        if isCyan(px):
            only_cyan[i,j] = px
        else:
            continue

io.imsave('onlyCyan.png', only_cyan)
exit()
'''

# In[64]:

# a list of dicts
# each dict has keys: rmin, rmax, cmin, cmax
# each dict gives the coordinates for a bounding box around a hand-drawn circle

boundingCoordinates = [] 

# crop the image
# crop of just cyan circle
img = img[325 : 637, 1257 : 1601]
# experimental crop
# img = img[325 : 637, 1440 : 1601]


# finds all hand-drawn closed curves that are drawn in valid Muxo colors
# adds a bounding box dict to boundingCoordinates for each hand-drawn circle
def find_circles(img):
    numCircles = 0
    first = False
    for i, row in enumerate(img):
        for j, triple in enumerate(row):
            #is Muxo color returns a function
            color_func = isMuxoColor(triple)
            
            # functions are truthy in Python
            if (color_func):
                visited_map = np.zeros(img.shape[:2], dtype=bool) # makes this an array of all False
                curr_bounds = {'rmin': len(img) + 1,
                               'rmax': -1,
                               'cmin': len(img[0]) + 1,
                               'cmax': -1
                              }
                print("starting coordinates: " + str(i) + ", " + str(j))

                visited_map, curr_bounds = find_coordinates_breadth_first(img, i, j, color_func, visited_map, curr_bounds)
                
                boundingCoordinates.append(curr_bounds)
                
                # black out this bounding box in the image
                rmin = boundingCoordinates[-1]["rmin"]
                rmax = boundingCoordinates[-1]["rmax"]
                cmin = boundingCoordinates[-1]["cmin"]
                cmax = boundingCoordinates[-1]["cmax"]
                img[rmin : rmax, cmin: cmax] = [0, 0, 0]
                '''
                TODO: to account for handwriting, filter out bounding boxes where 
                (ymin-ymax) is very different from (xmin-xmax)
                '''
                
                
                first = True
            if (first):
<<<<<<< HEAD
#                 break
                pass
        if(first):
#             break
            pass
=======
                break
        if(first):
            break
>>>>>>> 1e7c04db0412900a9ca3f6cd589eaa1f01e75239
            
    return visited_map, curr_bounds
        
    
        
# helper function for find_coordinates()
# returns True if all 8 neighbors of a pixel are visited; False otherwise
def surrounded_by_visited(visited_map, row_id, col_id):
    return visited_map[row_id-1][col_id] and\
            visited_map[row_id][col_id-1] and\
            visited_map[row_id+1][col_id] and\
            visited_map[row_id][col_id+1] and\
            visited_map[row_id+1][col_id+1] and\
            visited_map[row_id-1][col_id-1] and\
            visited_map[row_id-1][col_id+1] and\
            visited_map[row_id+1][col_id-1]

# helper function for find_coordinates_breadth_first()
# returns list of (row, col) pairs that are neighbors of input location and not yet visited
def get_unvisited_neighbors(visited_map, row_id, col_id):
    unvisited_neighbors = []
    
    if not visited_map[row_id+1][col_id]:
        unvisited_neighbors.append((row_id+1, col_id))
    
    if not visited_map[row_id-1][col_id]:
        unvisited_neighbors.append((row_id-1, col_id))
        
    if not visited_map[row_id][col_id+1]:
        unvisited_neighbors.append((row_id, col_id+1))
        
    if not visited_map[row_id][col_id-1]:
        unvisited_neighbors.append((row_id, col_id-1))
        
    if not visited_map[row_id+1][col_id+1]:
        unvisited_neighbors.append((row_id+1, col_id+1))
        
    if not visited_map[row_id-1][col_id+1]:
        unvisited_neighbors.append((row_id-1, col_id+1))
    
    if not visited_map[row_id+1][col_id-1]:
        unvisited_neighbors.append((row_id+1, col_id-1))
        
    if not visited_map[row_id-1][col_id-1]:
        unvisited_neighbors.append((row_id-1, col_id-1))
        
    return unvisited_neighbors

# modified bfs / dfs when locations_queue is made into a stack (add happens at the front)
def find_coordinates_breadth_first(img, row_id, col_id, color_func, visited_map, curr_bounds):
#     print('row is', row_id)
#     print('col is', col_id)
    global calls_counter
    
    found_bottom_point = False
    
    visited_map[row_id][col_id] = True
    
    # list of (row_id, col_id) pairs, in order of location to visit
    locations_queue = []
    secondary_queue = []
    
    # lst.extend(arg) appends all elements of arg (arg is a list)
    '''
    for BSF:
    
    locations_queue.extend(get_unvisited_neighbors(visited_map, row_id, col_id))
    '''
    # for DFS
    temp = get_unvisited_neighbors(visited_map, row_id, col_id)
    temp.extend(locations_queue)
    locations_queue = temp
    
    print(locations_queue)
   
        
    # empty lists are falsy in Python
    while (locations_queue or secondary_queue):
        if locations_queue:
            calls_counter = calls_counter + 1
            if (calls_counter % 200) == 0:
                    io.imsave('visited.jpg', 255 * visited_map)

            # locations_queue has at least one element
            first_loc = locations_queue.pop(0)
    #         print(calls_counter)
    
            print('first loc is', first_loc)
            visited_map[first_loc[0], first_loc[1]] = True

            if not color_func(img[first_loc[0]][first_loc[1]]):
                continue

            # DOES NOT actually indicate search is over 
            # (this was a flaw in previous logic - there are lots of ways to end up cornered)
            '''
            if surrounded_by_visited(visited_map, first_loc[0], first_loc[1]):
                print("surrounded by visited - ending search")
                return visited_map, curr_bounds
            '''
            # need to reach bottom of the curve (we're starting at the top)
            if not found_bottom_point:
                if first_loc[0] - row_id > 50 and col_id == first_loc[1]:
                    print('made it to the bottom')
                    found_bottom_point = True
           
            # done if bottom found and have no more unvisited neighbors
            if found_bottom_point and (not secondary_queue):
                print('made it all the way around - ending search')
                return visited_map, curr_bounds

            elif first_loc[0] < curr_bounds["rmin"]:
                curr_bounds["rmin"] = first_loc[0]

                # far from initial side of figure (we always start at the top --> around rmin), allow backtracking
                if (first_loc[0] - curr_bounds['rmin']) < 20:
                    print("Owie 1")
                    visited_map[first_loc[0]+1][first_loc[1]] = True
    #             visited_map[first_loc[0]+1][first_loc[1]+1] = True
    #             visited_map[first_loc[0]+1][first_loc[1]-1] = True
    #             visited_map[first_loc[0]][first_loc[1]+1] = True
    #             visited_map[first_loc[0]][first_loc[1]-1] = True

            elif first_loc[0] > curr_bounds["rmax"]:
                curr_bounds["rmax"] = first_loc[0]

                if (first_loc[0] - curr_bounds['rmin']) < 20: 
                    print("Owie 2")
                    visited_map[first_loc[0]-1][first_loc[1]] = True
    #             visited_map[first_loc[0]-1][first_loc[1]+1] = True
    #             visited_map[first_loc[0]-1][first_loc[1]-1] = True
    #             visited_map[first_loc[0]][first_loc[1]+1] = True
    #             visited_map[first_loc[0]][first_loc[1]-1] = True
            elif first_loc[1] < curr_bounds["cmin"]:
                curr_bounds["cmin"] = first_loc[1]

                # far from initial side of figure (we always start at the top --> around rmin), allow backtracking
                if (first_loc[0] - curr_bounds['rmin']) < 20:
                    print("Owie 3")
                    visited_map[first_loc[0]][first_loc[1]+1] = True
    #             visited_map[first_loc[0]+1][first_loc[1]+1] = True
    #             visited_map[first_loc[0]-1][first_loc[1]+1] = True
    #             visited_map[first_loc[0]+1][first_loc[1]] = True
    #             visited_map[first_loc[0]-1][first_loc[1]] = True
            elif first_loc[1] > curr_bounds["cmax"]:
                curr_bounds["cmax"] = first_loc[1]

                # far from initial side of figure (we always start at the top --> around rmin), allow backtracking
                if (first_loc[0] - curr_bounds['rmin']) < 20:
                    print("Owie 4")
                    visited_map[first_loc[0]][first_loc[1]-1] = True
    #             visited_map[first_loc[0]+1][first_loc[1]-1] = True
    #             visited_map[first_loc[0]-1][first_loc[1]-1] = True
    #             visited_map[first_loc[0]+1][first_loc[1]] = True
    #             visited_map[first_loc[0]-1][first_loc[1]] = True
            else:
                # this location does not get closer to the goal
                secondary_queue.extend(get_unvisited_neighbors(visited_map, first_loc[0], first_loc[1]))
#                 random.shuffle(secondary_queue)
                continue

    #         print("YOOOOOOO!")

            locations_queue.extend(get_unvisited_neighbors(visited_map, first_loc[0], first_loc[1]))
    #         print(locations_queue)
        else:
            print('first loc is ', first_loc)
            
            esc_location = secondary_queue.pop(len(secondary_queue)-1)
            
            
#             esc_location = secondary_queue.pop(0)
            
            print('now going to ', esc_location)
            locations_queue.append(esc_location)
#             pass
    # after the while loop
    print("outside the while loop")
    return visited_map, curr_bounds
    
#the [row, column] rgb triple we have flagged as part of a circle

# walks along contiguous segment of image with given color
# returns [[rmin,rmax],[cmin,cmax]] for that segment -> will be used to
# give coordinates of the bounding box later
# amounts to modified dfs
def find_coordinates(img, row_id, col_id, color_func, visited_map, curr_bounds):
    global calls_counter
    calls_counter = calls_counter + 1
    
    visited_map[row_id][col_id] = True
    
    if surrounded_by_visited(visited_map, row_id, col_id):
        return visited_map, curr_bounds 
    if not color_func(img[row_id][col_id]):
        # no longer in region of same color
        return visited_map, curr_bounds
    
    # need to change to <= and >= instead of < and >
    if row_id < curr_bounds["rmin"]:
        if (calls_counter % 200) == 0:
            io.imsave('visited.jpg', 255 * visited_map)
        
        curr_bounds["rmin"] = row_id
        # call on smaller row, larger and smaller cols
        visited_map, curr_bounds = find_coordinates(img, row_id - 1, col_id, color_func, visited_map, curr_bounds)
        visited_map, curr_bounds = find_coordinates(img, row_id, col_id + 1, color_func, visited_map, curr_bounds)
        visited_map, curr_bounds = find_coordinates(img, row_id, col_id - 1, color_func, visited_map, curr_bounds)
    elif row_id > curr_bounds["rmax"]:
        
        if (calls_counter % 200) == 0:
            io.imsave('visited.jpg', 255 * visited_map)
        
        curr_bounds["rmax"] = row_id
        # call on larger row, larger and smaller cols
        visited_map, curr_bounds = find_coordinates(img, row_id + 1, col_id, color_func, visited_map, curr_bounds)
        visited_map, curr_bounds = find_coordinates(img, row_id, col_id + 1, color_func, visited_map, curr_bounds)
        visited_map, curr_bounds = find_coordinates(img, row_id, col_id - 1, color_func, visited_map, curr_bounds)
    
    # else:
       # return visited_map, curr_bounds

    
    elif col_id < curr_bounds["cmin"]:
        
        if (calls_counter % 200) == 0:
            io.imsave('visited.jpg', 255 * visited_map)
            
        curr_bounds["cmin"] = col_id
        # call smaller col, larger and smaller rows
        visited_map, curr_bounds = find_coordinates(img, row_id, col_id - 1, color_func, visited_map, curr_bounds)
        visited_map, curr_bounds = find_coordinates(img, row_id - 1, col_id, color_func, visited_map, curr_bounds)
        visited_map, curr_bounds = find_coordinates(img, row_id + 1, col_id, color_func, visited_map, curr_bounds)
    elif col_id > curr_bounds["cmax"]:
        
        if (calls_counter % 200) == 0:
            io.imsave('visited.jpg', 255 * visited_map)
        
        curr_bounds["cmax"] = col_id
        # call on rmin, rmax, cmax
        visited_map, curr_bounds = find_coordinates(img, row_id, col_id + 1, color_func, visited_map, curr_bounds)
        visited_map, curr_bounds = find_coordinates(img, row_id - 1, col_id, color_func, visited_map, curr_bounds)
        visited_map, curr_bounds = find_coordinates(img, row_id + 1, col_id, color_func, visited_map, curr_bounds)
    else:
        return visited_map, curr_bounds
    
    # call on diagonals in all cases
    
    visited_map, curr_bounds = find_coordinates(img, row_id - 1, col_id - 1, color_func, visited_map, curr_bounds)
    visited_map, curr_bounds = find_coordinates(img, row_id + 1, col_id + 1, color_func, visited_map, curr_bounds)  
    visited_map, curr_bounds = find_coordinates(img, row_id - 1, col_id + 1, color_func, visited_map, curr_bounds)
    visited_map, curr_bounds = find_coordinates(img, row_id + 1, col_id - 1, color_func, visited_map, curr_bounds)
    
    
    # print(curr_bounds)
    print("third")
    return visited_map, curr_bounds
    
  

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

# In[66]:


vis_map, c_bounds = find_circles(img)
'''
crop given by:
img = img[325 : 637, 1257 : 1601]
'''
c_bounds['rmin'] = c_bounds['rmin'] + 325
c_bounds['rmax'] = c_bounds['rmax'] + 325
c_bounds['cmin'] = c_bounds['cmin'] + 1257
c_bounds['cmax'] = c_bounds['cmax'] + 1257
print(c_bounds)


# In[67]:


# print(boundingCoordinates)


# open CV function: identify colors in an image by specifying their rgb boundaries
# https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
# 

# In[68]:


# print(visited_map)

# 255 factor used to make visited map visible
io.imsave('visited.jpg', 255 * vis_map)
io.imsave('blackout-img.png', img)

# In[69]:
# plt.imshow(img)
# In[70]:
# plt.imshow(visited_map)

# In[ ]: