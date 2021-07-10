import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
import os
import glob
import xml.etree.ElementTree as gfg

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
    'plum-violet': [(120,170),(50,100),(120,170)],
}

'''
Gets bounding boxes for input img and returns list of bounding box dicts
Format of returned list: [{xmin, ymin, xmax, ymax}...]
'''
def get_bounding_boxes(img, path):
    box_list = []

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
        whether_color = np.empty(img.shape[0:2])
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

        '''
        Eliminate "circles" that are just flowers or other small regions of
        contiguous Muxo color
        Condition removal on the area of each connected component
        '''
        MIN_AREA = 2000

        num_circles = 0

        for i in range(retval):
            if i == 0:
                # skip over the background
                continue
            elif stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
                # ignore components that are too small to be nest circles
                continue
            else:
                print("Coords for circle", num_circles, "are:")
                num_circles = num_circles + 1

                top = stats[i,cv2.CC_STAT_TOP]
                left = stats[i,cv2.CC_STAT_LEFT]
                bottom = top + stats[i, cv2.CC_STAT_HEIGHT] - 1
                right = left + stats[i, cv2.CC_STAT_WIDTH] - 1
                print("Top:", top)
                print("Left:", left)
                print("Bottom:", bottom)
                print("Right:", right)

                print("(This is connected component number", i,")")

                box_list.append(
                    {'xmin': left,
                    'ymin': top,
                    'xmax': right,
                    'ymax': bottom})

                # Draw bounding boxes on original image
                img[top,left:right] = [0,0,0]
                img[bottom,left:right] = [0,0,0]
                img[top:bottom,left] = [0,0,0]
                img[top:bottom,right] = [0,0,0]

        # if i == (retval-1):
        #     print("There are", num_circles, "circles with Muxo color", color_name)
        #     io.imshow(img)
        #     plt.show()  
    return box_list

def generate_annotations(path, image_dims, bounding_box_list):
    img_filename = os.path.basename(path)
    annotation_filename = img_filename[:-3] + 'xml'


    root = gfg.Element("annotation") 
      
    b1 = gfg.SubElement(root, "folder") 
    b1.text = "train"
    b2 = gfg.SubElement(root, "filename") 
    b2.text = img_filename
      
    m2 = gfg.SubElement(root, "path")
    m2.text = path 
      
    c1 = gfg.SubElement(root, "source") 
    c2 = gfg.SubElement(c1, "database") 
    c2.text = "Unknown"
      
    s = gfg.SubElement(root, "size")
    
    h = gfg.SubElement(s, "height")
    h.text = str(image_dims[0])
    w = gfg.SubElement(s, "width")
    w.text = str(image_dims[1])
    d = gfg.SubElement(s, "depth")
    d.text = str(image_dims[2])

    seg = gfg.SubElement(root, "segmented")
    seg.text = "0"
      
    for box in bounding_box_list:
        obj = gfg.SubElement(root, "object")
        n = gfg.SubElement(obj, "name")
        n.text = "nest"
        p = gfg.SubElement(obj, "pose")
        p.text = "Unspecified"
        t = gfg.SubElement(obj, "truncated")
        t.text = "0"
        d = gfg.SubElement(obj, "difficult")
        d.text = "0"
        bndbox = gfg.SubElement(obj, "bndbox")

        xmin = gfg.SubElement(bndbox, "xmin")
        xmin.text = str(box['xmin'])
        xmax = gfg.SubElement(bndbox, "xmax")
        xmax.text = str(box['xmax'])
        ymin = gfg.SubElement(bndbox, "ymin")
        ymin.text = str(box['ymin'])
        ymax = gfg.SubElement(bndbox, "ymax")
        ymax.text = str(box['ymax'])
      
    tree = gfg.ElementTree(root) 
    print("Now saving:", annotation_filename)
      
    with open("birds-dataset/train/annotations/" + annotation_filename, "wb") as files: 
        tree.write(files)

if __name__ == "__main__":

    pathname = "/Users/charlie/Desktop/VSFS - Bird Photos/20201022_BISC_BirdFlight/**/*.jpg"
    # .iglob() returns iterator without storing all images at once
    image_paths = glob.glob(pathname, recursive=True)

    print("Found %s image paths" % len(image_paths))

    for path in image_paths:
        print("\n\nNow analyzing ", path)
        img = io.imread(path)
        bounding_box_list = get_bounding_boxes(img, path)
        # generate_annotations(path, img.shape, bounding_box_list)
        print("showing image:")
        io.imshow(img)
        print("showing plot:")
        plt.show()


