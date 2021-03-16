import torch.optim as optim

from LegoDataset import *
from utils import *

# get device
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"======Current Device: {device}======")
# load data
#
legos = LegoDataset(cfg.root, cfg.transform)
print(f"======Data Summary: {len(legos)} data; {legos.num_cls} classes======")
# Split Train and Val
#
tr_size = round(len(legos) * cfg.split_factor)
va_size = len(legos) - tr_size
tr, va = data.random_split(legos, [tr_size, va_size])
tr_loader = data.DataLoader(tr, batch_size=cfg.tr_batch_size, shuffle=True, num_workers=4, pin_memory=True)
va_loader = data.DataLoader(va, batch_size=cfg.va_batch_size, num_workers=4, pin_memory=True)
print(f"======Train: {len(tr_loader)} Test: {len(va_loader)}======")
# Prepare Model
#
classifier = get_vgg16(device)
print(classifier)
assert classifier(legos[0][1].unsqueeze(0).to(device)).size() == (1, 50)
# Train Model
#
optimizer = optim.Adam(classifier.parameters(), lr=cfg.learning_rate)
load_progress(classifier)
tr_loss, val_loss, tr_accuracy, va_accuracy = train_model(classifier, tr_loader, va_loader, optimizer, device)
plot(tr_loss, val_loss, tr_accuracy, va_accuracy)
