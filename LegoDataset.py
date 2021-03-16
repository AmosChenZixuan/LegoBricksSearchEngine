import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T
import config as cfg

class LegoDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.T = transform

        self.imgs = os.listdir(root)[:cfg.n]
        
        self.num_cls=0
        self.cls = {}
        for i in self.imgs:
            y = self._get_cls(i)
            if y not in self.cls:
                self.cls[y] = self.num_cls
                self.num_cls += 1

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root, self.imgs[idx])
        # print(img_loc)
        img = Image.open(img_loc)  # .convert('RGB')

        img_t = self._T(img)
        cls = self.get_class(idx)
        return cls, img_t, self.cls[cls] #class, input, target

    def _T(self, img):
        if self.T:
            return self.T(img)
        return T.Compose([
            T.ToTensor()
        ])(img)
    
    def _get_cls(self, name):
        return name.split()[0]

    def get_raw(self, idx):
        img_loc = os.path.join(self.root, self.imgs[idx])
        # print(img_loc)
        return Image.open(img_loc)  # .convert('RGB')
    
    def get_class(self, idx):
        return self._get_cls(self.imgs[idx])
