import logging
import numpy as np

import torch
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

def load_model(base_model, device, pretrained, out_features, layers_to_freeze, freeze_all = False):
  
  if base_model == 'resnet18':
    net = models.resnet18(pretrained=pretrained)
  elif base_model == 'resnet34':
    net = models.resnet34(pretrained=pretrained)
  elif base_model == 'resnet50':
    net = models.resnet50(pretrained=pretrained)
  else:
    raise Exception(f"{base_model} model currently unssoported, please choose among resnet18, resnet34 or resnet50")

  
  net = net.cuda() if device.type != 'cpu' else net

  if layers_to_freeze:
    for layer in layers_to_freeze:
      freeze_params(net.layer.parameters())

  if freeze_all:
    freeze_params(net.parameters())
  else:
    unfreeze_params(net.parameters())


  num_ftrs = net.fc.in_features
  net.fc = nn.Linear(in_features=num_ftrs, out_features=out_features, bias=True)
  net.fc = net.fc.cuda() if device.type != 'cpu' else net.fc

  return net

def train_model(model, optimizer, criterion, n_epochs, train_dataloader, test_dataloader, device):

    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_dataloader)

    logger = logging.getLogger('Train')

    for epoch in range(1, n_epochs+1):
        running_loss = 0.0
        correct = 0
        total=0
        logger.info(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()
            
            outputs = model(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
            if (batch_idx) % 10 == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch, n_epochs, batch_idx, total_step, loss.item()))

        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/total_step)
        logger.info(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
        batch_loss = 0
        total_t=0
        correct_t=0
        with torch.no_grad():
            model.eval()
            for data_t, target_t in (test_dataloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = model(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t/total_t)
            val_loss.append(batch_loss/len(test_dataloader))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

            
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(model.state_dict(), 'model/resnet18_pat.pt')
                print('Improvement-Detected, save-model')
        model.train()

    return model

def predict(model, dataloader):
    pass