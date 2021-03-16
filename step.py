import torch
import numpy as np
from sklearn.metrics import accuracy_score

def train(encoder, train_loader, loss_fn, optimizer, device):
    """
    Performs a single training step
    Args:
    encoder: A convolutional Encoder. E.g. torch_model ConvEncoder
    train_loader: PyTorch dataloader, containing (images, images).
    loss_fn: PyTorch loss_fn, computes loss
    optimizer: PyTorch optimizer.
    Returns: Train Loss
    """
    encoder.train()  # training mode
    loss = None
    temp = []
    acc_temp = []

    for batch, (cls, x, target) in enumerate(train_loader):
        x = x.to(device)
        target = target.to(device)     
        encode = encoder(x)
        
        optimizer.zero_grad()
        loss = loss_fn(encode, target)
        loss.backward()
        optimizer.step()
        
        temp.append(loss.item())
        acc_temp.append(accuracy_score(target.cpu().detach(), np.argmax(encode.cpu().detach(), axis=1)))
        print(f"\rTraining:Batch[{batch}]..loss:{temp[-1]}", end='')
    return np.mean(temp) if loss!=None else float('inf'), np.mean(acc_temp)


def test(encoder, test_loader, loss_fn, device):
    encoder.eval()  # testing mode
    loss = None
    temp = []
    acc_temp = []
    
    with torch.no_grad():
        for batch, (cls, x, target) in enumerate(test_loader):
            x = x.to(device)
            target = target.to(device)
            encode = encoder(x)
            loss = loss_fn(encode, target)
            
            temp.append(loss.item())
            acc_temp.append(accuracy_score(target.cpu().detach(), np.argmax(encode.cpu().detach(), axis=1)))
            print(f"\rTesting:Batch[{batch}]..loss:{temp[-1]}", end='')
    return np.mean(temp) if loss!=None else float('inf'), np.mean(acc_temp)

