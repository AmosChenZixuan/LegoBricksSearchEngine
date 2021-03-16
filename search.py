from utils import *
from LegoDataset import *
from sklearn.neighbors import NearestNeighbors
# get device
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"======Current Device: {device}======")
# load encoder
#
encoder = get_vgg16(device)
load_progress(encoder)
encoder = nn.Sequential(*list(encoder.children()))[:-1]
#print(encoder.eval())
# load data
#
legos = LegoDataset(cfg.root, cfg.transform)
print(f"======Data Summary: {len(legos)} data; {legos.num_cls} classes======")
full_loader = data.DataLoader(legos, batch_size=cfg.fl_batch_size, shuffle=False, num_workers=4, pin_memory=True)
# create
#
flattened_embedding = get_embedding(encoder, full_loader, device, pretrained=False)
print(f"======Embedding Shape: {flattened_embedding.shape}======")
knn = NearestNeighbors(n_neighbors=cfg.K, metric="cosine")
knn.fit(flattened_embedding)
# Query
#
query(legos, encoder, knn, device)
