import PIL.Image
import os

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
            image.save("./1/"+"SurgicalMask_" + str(index) + ".jpg")
        except:
            print("error: " + path)
        index += 1

if __name__ == '__main__':
   dir = ["Surgical mask", "Cloth mask", "No face mask", "N95 mask", "N95 mask with valve"]
   for address in dir:
    convert(address)
