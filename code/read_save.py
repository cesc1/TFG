import os
import sys
import numpy as np
import pickle
from PIL import Image
import lmdb

## hex (str) to ascii
def hex_to_ascii(hex):
    bytes_obj = bytes.fromhex(hex)
    return bytes_obj.decode("ASCII")


## Read image files (png) in folder
def read_images_files(folder_path, name_label = "img",  extension = ".png", color = "full"):
    images, labels = [], []
    images_names = [image_name for image_name in os.listdir(folder_path) if extension in image_name]

    for i, image_name in enumerate(images_names):
        if color == "full":
            images.append(np.array(Image.open(folder_path + f"{image_name}")))
        elif color == "bw":
            images.append(np.array(Image.open(folder_path + f"{image_name}").convert('L')))
        elif color == "bi":
            images.append(np.array(Image.open(folder_path + f"{image_name}").convert('1')))
        labels.append(name_label) # name_label + f"_id{i}"

    return np.array(images), np.array(labels)


# Read all folders with png files 
def read_images_folders(folder_path, extension = ".png", color = "full", ascii_label = False):
    images, labels = [], []

    for image_class in os.listdir(folder_path):
        imgs_path = folder_path + f"{image_class}/" + f"train_{image_class}/"
        if ascii_label:  # canvi label, estetic
            image_class = hex_to_ascii(image_class)
        img_c, labs_c = read_images_files(imgs_path, image_class, extension = extension, color = color)
        images.append(img_c)
        labels.append(labs_c)
        sys.stdout.write("\r" + f"Class {image_class} loaded...") # solucio alternativa, ja que sembla que print no va en jupyter

    return images, labels


class LMBD_Image: # opció d'utilitzar la llibreria caffe
    def __init__(self, image, label):
        self.shape = image.shape
        self.image = image.tobytes()
        self.label = label
    
    def get_image(self):
        image = np.frombuffer(self.image, dtype = np.bool_)
        return image.reshape(*self.shape)
    

def save_images_lmdb(folder_path, file_name, images, labels):
    map_size = images.nbytes * 10

    env = lmdb.open(folder_path + file_name, map_size = map_size)

    with env.begin(write = True) as txn: # txn = transaction
        for i in range(len(images)):
            value = LMBD_Image(images[i], labels[i])
            txn.put(f"{i:09}".encode("ascii"), pickle.dumps(value))
    env.close()


def read_images_lmdb(folder_path, file_name):
    env = lmdb.open(folder_path + file_name, readonly = True)
    images, labels = [], []

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            data = txn.get(key)
            data_obj = pickle.loads(data)
            images.append(data_obj.get_image())
            labels.append(data_obj.label)
    env.close()

    return np.array(images), labels


def read_folder_lmdb(folder_path, iterator = range(30, 40)):
    images, labels = [], []

    for i in iterator:
        img, l = read_images_lmdb(folder_path, str(i))
        images.append(img)
        labels.append(l)

    return np.concatenate(images, axis = 0), np.concatenate(labels)


# Read iteravely the folders with png files, and save them in lmdb 
def png_to_lmdb(folder_path, save_folder_path, extension = ".png", color = "full", ascii_label = False):
    
    for image_class in os.listdir(folder_path):
        imgs_path = folder_path + f"{image_class}/" + f"train_{image_class}/"
        file_name = image_class
        if ascii_label:  # canvi label, estetic
            image_class = hex_to_ascii(image_class)
        img_c, labs_c = read_images_files(imgs_path, image_class, extension = extension, color = color)
        save_images_lmdb(save_folder_path, file_name, img_c, labs_c)
        sys.stdout.write("\r" + f"Class {image_class} loaded and saved...") # solucio alternativa, ja que sembla que print no va en jupyter

