import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from skimage import io
import csv
from PIL import Image

num_copies = 5

img_dir = "./samples"
dest_dir = "../test"
#"../train"


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

#Add them to list
cat_dirs = [circ_red_dir, circ_blue_dir, circ_green_dir,
            square_red_dir, square_blue_dir, square_green_dir,
            triangle_red_dir, triangle_blue_dir, triangle_green_dir]


#initialize arrays
file_names = []
shapes = []
colours = []


#For each category in the list
for i in range(0, len(cat_names)):
    img_name = cat_names[i][0] + "_" + cat_names[i][1]
    dir = cat_dirs[i]
    #Loop through, duplicating the image, inserting in its subfolder
    #and copying its labels to the csv
    for idx in range(1, num_copies):
        old_path = os.path.join(img_dir, img_name + ".png")
        new_path = os.path.join(dir, img_name + str(idx) + ".jpg")

        image = Image.open(old_path)
        temp = image.copy()
        temp.save(new_path)
        
        file_names.append(new_path)
        shapes.append(cat_names[i][0])
        colours.append(cat_names[i][1])
idx_to_label = {
    'circle': 0,
    'square': 1,
    'triangle': 2,
    'red': 3,
    'green': 4,
    'blue': 5 
}
shapes = list(map(idx_to_label.get, shapes))
colours = list(map(idx_to_label.get, colours))
print(type(shapes))
df = pd.DataFrame(data={"image":file_names, "shapes": shapes, "colours":colours})
df.to_csv(os.path.join(dest_dir , "./testFile.csv"), sep=",", index=False)
#dataset is in the form [image_name, shape category, color] as a csv
