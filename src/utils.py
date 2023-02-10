import re
from glob import glob
from PIL import Image
from torch import nn
from torchvision import models


def freeze_params(model_params):
    for param in model_params:
        param.requires_grad = False
        
def unfreeze_params(model_params):
    for param in model_params:
        param.requires_grad = True
        
def get_trainable(model_params):
    return (p for p in model_params if p.requires_grad)

def load_image(filename) :
    img = Image.open(filename)
    img = img.convert('RGB')
    return img

def get_data(folder_path = './data/images/*.jpg'):
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

def load_model(base_model, device, pretrained, out_features, freeze = False):

  if base_model == 'resnet18':
    net = models.resnet18(pretrained=pretrained)
  elif base_model == 'resnet34':
    net = models.resnet34(pretrained=pretrained)
  elif base_model == 'resnet50':
    net = models.resnet50(pretrained=pretrained)
  else:
    raise Exception("Base model currently unssoported, please choose among resnet18, resnet34 or resnet50")

  
  net = net.cuda() if device.type != 'cpu' else net

  if freeze:
    freeze_params(net.parameters())
  else:
    unfreeze_params(net.parameters())


  num_ftrs = net.fc.in_features
  net.fc = nn.Linear(in_features=num_ftrs, out_features=out_features, bias=True)
  net.fc = net.fc.cuda() if device.type != 'cpu' else net.fc

  return net