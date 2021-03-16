import torchvision.transforms as T
import torch.nn as nn

# data
root = '/home/amos/workspace/DB/legos/dataset/'
transform = T.Compose([
    T.ToTensor()
])

n = 10000
split_factor = 0.75
tr_batch_size = 4
va_batch_size = 16
fl_batch_size = 8

# Train
epochs = 20
threshold = 0.5
max_loss = 0.5  # float('inf')
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-4

# model
load_dict_name = 'vgg16bn_lr_01_model.pt'
save_dict_name = 'vgg16bn_lr_01_model.pt'
# his:
# vgg16bn10k
# vgg16bn
# vgg16bn iter 100  : 1.05
# vgg16bn lr 05: 0.7
# vgg16bn lr 03: 0.7
#            01: 0.5

##  SEARCH
# embedding
load_emb_name = f'data_{n}_feature.npy'
#load_emb_name = f'data_{n}_full_layers.npy'
save_emb_name = load_emb_name

# KNN
K = 10
