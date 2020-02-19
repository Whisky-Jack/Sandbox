import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from skimage import io
import csv
from Pillow import image


num_copies = 500

curr_dir = "/home/josh/ML/simple_shapes/train/root"
dest_dir = "/home/josh/ML/simple_shapes/train/root"
#"/home/josh/ML/simple_shapes/root"


#set the image names for the 9 categories
circ_red = ["circle", "red"]
circ_blue = ["circle", "blue"]
circ_green = ["circle", "green"]


square_red = ["square", "red"]
square_blue = ["square", "blue"]
square_green = ["square", "green"]

triangle_red = ["triangle", "red"]
triangle_blue = ["triangle", "blue"]
triangle_green = ["triangle", "green"]

cat_names = [circ_red, circ_blue, circ_green,
            square_red, square_blue, square_green,
            triangle_red, triangle_blue, triangle_green]
#set the directories for the 9 categories
circ_red_dir = dest_dir + "/circle"
circ_blue_dir = dest_dir + "/circle"
circ_green_dir = dest_dir + "/circle"

square_red_dir = dest_dir + "/square"
square_blue_dir = dest_dir + "/square"
square_green_dir = dest_dir + "/square"

triangle_red_dir = dest_dir + "/triangle"
triangle_blue_dir =dest_dir + "/triangle"
triangle_green_dir = dest_dir + "/triangle"

cat_dirs = [circ_red_dir, circ_blue_dir, circ_green_dir,
            square_red_dir, square_blue_dir, square_green_dir,
            triangle_red_dir, triangle_blue_dir, triangle_green_dir]
#Append them to list

#image = io.imread(circ_red[0] + "_" + circ_red[1] + ".png")
#shutil.copy("/home/josh/ML/simple_shapes/root/circle_red.png", "/home/josh/ML/simple_shapes/root/fuck_off.png")

#initialize arrays
file_names = []
shapes = []
colours = []


#For each category in the list
#with open('testfile')
for i in range(0, len(cat_names)):
    img_name = cat_names[i][0] + "_" + cat_names[i][1]
    dir = cat_dirs[i]
    for idx in range(1, num_copies):
        new_path = os.path.join(dir, img_name + str(idx) + ".png")
        #print(new_img_name)
        #print(os.path.join(dir, new_img_name))
        shutil.copy(os.path.join(curr_dir, img_name + ".png"), new_path)#os.path.join(dir, new_img_name))
        file_names.append(new_path)
        shapes.append(cat_names[i][0])
        colours.append(cat_names[i][1])
    #Loop through, duplicating the image, inserting in its subfolder
    #and copying its labels to the csv
df = pd.DataFrame(data={"image":file_names, "shapes": shapes, "colours":colours})
df.to_csv(os.path.join(dest_dir , "./testFile.csv"), sep=",", index=False)
#dataset is in the form [image_name, shape category, color] as a csv
#print(img_name)
