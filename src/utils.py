import re
from glob import glob

from PIL import Image

import torch
from torch import nn
from torchvision import models


def load_image(filename) :
    img = Image.open(filename)
    img = img.convert('RGB')
    return img

def load_data_from_folder(folder_path = './data/images/*.jpg'):
  filenames = glob(folder_path)

  classes = set()

  data = []
  labels = []

  for image in filenames:
    class_name = re.split('(\d+)', image.split('/')[-1])[0][:-1]
    classes.add(class_name)

    img = load_image(image)

    data.append(img)
    labels.append(class_name)

  # convert classnames to indices
  class2idx = {cl: idx for idx, cl in enumerate(classes)}        
  labels = torch.Tensor(list(map(lambda x: class2idx[x], labels))).long()

  return list(zip(data, labels)), labels