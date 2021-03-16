import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import step
from LegoDataset import *

def get_vgg16(device):
    classifier = models.vgg16_bn(pretrained=True, progress=True)
    modules = list(classifier.children())
    modules[-1][-1] = nn.Linear(in_features=4096, out_features=50, bias=True)
    return classifier.to(device)


def load_progress(classifier):
    try:
        classifier.load_state_dict(torch.load(cfg.load_dict_name))
        classifier.eval()
        print("Load Successfully")
    except:
        import traceback
        traceback.print_exc()


def plot(trl, val, tracc, vaacc):
    plt.figure(figsize=(15, 5))
    # plot loss
    plt.subplot(1, 2, 1)
    plt.plot(trl, label='train')
    plt.plot(val, label='val')
    plt.legend()
    # plot accuracy score
    plt.subplot(1, 2, 2)
    plt.plot(tracc, label='train')
    plt.plot(vaacc, label='val')
    plt.legend()

    plt.show()


def train_model(classifier, tr_loader, va_loader, optimizer, device):
    trl, val = [], []
    tracc, vaacc = [], []
    torch.cuda.empty_cache()
    max_loss = cfg.max_loss
    for ep in tqdm(range(cfg.epochs)):
        print()
        tr_loss, tr_acc = step.train(classifier, tr_loader, cfg.loss_fn, optimizer, device)
        print(f"Epochs = {ep}, Training Loss : {tr_loss}")
        va_loss, va_acc = step.test(classifier, va_loader, cfg.loss_fn, device)
        print(f"Epochs = {ep}, Validation Loss : {va_loss}")

        trl.append(tr_loss)
        val.append(va_loss)
        tracc.append(tr_acc)
        vaacc.append(va_acc)

        plot(trl, val, tracc, vaacc)

        if va_loss < max_loss:
            print("Validation Loss decreased, saving new best model")
            torch.save(classifier.state_dict(), cfg.save_dict_name)
            max_loss = va_loss

        if tr_loss < cfg.threshold and va_loss < cfg.threshold:
            print('=======Early Termination=======')
            print(f"Epochs = {ep}, Training Loss : {tr_loss}, Validation Loss : {va_loss}")
            break
    return trl, val, tracc, vaacc


def get_embedding(encoder, full_loader, device, pretrained=False):
    cap = cfg.n
    embedding = None
    if pretrained:
        try:
            return np.load(cfg.load_emb_name)
        except:
            print('Pretrained Embedding not found.')
    print('Creating...')
    for batch, (idx, x, y) in enumerate(full_loader):
        if batch * cfg.fl_batch_size > cap:
            break
        print(f"\rBatch[{batch}]..{idx}", end='')
        with torch.no_grad():
            x = x.to(device)
            feature = encoder(x)
            embedding = torch.cat((embedding, feature), 0) if embedding is not None else feature
    # Convert embedding to numpy and save them
    numpy_embedding = embedding.cpu().detach().numpy()
    num_images = numpy_embedding.shape[0]

    # Save the embeddings for complete dataset, not just train
    flattened_embedding = numpy_embedding.reshape((num_images, -1))
    np.save(cfg.save_emb_name, flattened_embedding)
    print("Saving to ", cfg.save_emb_name)
    return flattened_embedding


def search(encoder, path, knn, device):
    img = Image.open(path).convert('RGB')
    img = cfg.transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    feature = encoder(img)

    feature = feature.cpu().detach().numpy().reshape((1, -1))
    _, indices = knn.kneighbors(feature)
    # print(feature.shape)
    # print(np.mean(feature[0]))
    return indices.tolist()


def get_random_img(full = False, test = False, path = None):
    if path is not None:
        return path
    directory = os.listdir(cfg.root)
    if not full:
        directory = directory[:cfg.n]
    elif test:
        directory = directory[cfg.n:]
    return np.random.choice(directory)


def query(legos, encoder, knn, device):
    while True:
        print("========Enter File Name | 'Default':Sample From Seen Imgs | '1':Sample From full dataset | '2':Sample "
              "From Unseen Imgs========")
        arg = input(">>>").strip()

        full, test, path = False, False, None
        if arg == 'q':
            break
        elif arg == '1':
            full = True
        elif arg == '2':
            full = True
            test = True
        elif arg.endswith('.png'):
            path = arg
        path = get_random_img(full, test, path)
        print('Query:', path)
        query_cls = path.split()[0]
        plt.figure(figsize=(20, 15))

        plt.subplot(3, 5, 1)
        plt.imshow(Image.open(cfg.root + path))
        plt.title("Query")

        idx = search(encoder, cfg.root + path, knn, device)[0]
        #
        for i in range(len(idx)):
            img = legos.get_raw(idx[i])
            plt.subplot(3, 5, i + 6)

            cls = legos.get_class(idx[i])
            if cls == query_cls:
                cls += '*'
            plt.title(str(idx[i]) + '-' + cls)
            plt.imshow(img)
        plt.show()
        #
        total_c, c = 0, 0
        for i in range(cfg.n):
            pth = legos.imgs[i]
            if pth.split()[0] == query_cls:
                if i in idx:
                    print(i, pth, '*')
                    c += 1
                total_c += 1
        print(f"======In Data: {total_c}; Retrieved: {c}/{cfg.K}======")







