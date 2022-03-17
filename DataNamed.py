import PIL.Image
import os
import re

def convert(address):
    dir = "./" + address + "/"
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

if __name__ == '__main__':
   dir = ["Surgical Mask", "Cloth Mask", "No Face Mask", "N95 Mask", "N95 Mask With Valve"]
   for address in dir:
    convert(address)
