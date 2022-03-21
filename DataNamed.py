import PIL.Image
import os
import re

import pandas as pd
from matplotlib import pyplot as plt


def convert(address):
    dir = "./Data/" + address + "/"
    file_list = os.listdir(dir)
    index = 1
    for filename in file_list:
        path = ''
        path = dir+filename
        image = PIL.Image.open(path)
        if filename.endswith("jpeg") or filename.endswith("png"):
            image = image.convert('RGB')
        try:
            address = re.sub('\s+', '', str).strip() +"_"
            image.save("./1/" + address + str(index) + ".jpg")
        except:
            print("error: " + path)
        index += 1

def get_number(address):
    dir = "./Data/" + address + "/"
    all_files = os.listdir(dir)  # os.curdir 表示当前目录 curdir:currentdirectory
    type_dict = dict()
    for each_file in all_files:
            type_dict.setdefault(address, 0)
            type_dict[address] += 1

    for each_type in type_dict:
        print("The number of {} images: {}".format(each_type, type_dict[each_type]))
    return type_dict[each_type]

def print_data(dir, data_list):

    plt.figure(figsize = [10,10])
    plt.bar(x=dir, height=data_list, color = ['r', 'g', 'b', 'yellow', 'black'])
    plt.title("Datasets of Masks")
    plt.xlabel("Type")
    plt.ylabel("Data")
    plt.show()

if __name__ == '__main__':
    dir = ["Surgical Mask", "Cloth Mask", "No Face Mask", "N95 Mask", "N95 Mask With Valve"]
    '''
     for address in dir:
         convert(address)
    '''
    data_list = []
    for address in dir:
        data_list.append(get_number(address))
    print_data(dir, data_list)