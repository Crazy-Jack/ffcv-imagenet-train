import torch
import clip
from PIL import Image
import os 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
from tqdm import tqdm

image = preprocess(Image.open("/home/ylz1122/nips2023_shape_vs_texture/topk-neurons-visualization-supp/shapebiasbench/airplane1-chair2.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    image_features = torch.nn.functional.normalize(image_features).float().detach().cpu()
    

# iter over all val images
class MyDataset(Dataset):
    def __init__(self, dir="/home/ylz1122/data/imagenet1k/ILSVRC/Data/CLS-LOC/val_ori"):
        super().__init__()
        self.dir = dir 
        list_img = os.listdir(self.dir)
        self.img_paths = [os.path.join(self.dir, i) for i in list_img]
      
    def __getitem__(self, idx):
         
        img_ = Image.open(self.img_paths[idx]).convert('RGB')

        # Preprocess image
        tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5), (0.5)),])
        img = tfms(img_)
        return img, self.img_paths[idx]
    def __len__(self):
        return len(self.img_paths)


# if not os.path.isfile("feat_bank.pt"):
if True:
    mydat = MyDataset()
    loader = DataLoader(mydat, batch_size=256, shuffle=False, sampler=None,
            batch_sampler=None, num_workers=1)

    images_path_bank = []
    feat_bank = []
    for i,paths in tqdm(loader, total=len(loader)):
        # print(f"i {i.shape} path {paths[0]}")
        i = i.to(device)
        with torch.no_grad():
            i_feat = model.encode_image(i)
            i_feat = torch.nn.functional.normalize(i_feat).detach().cpu()
        for path_i in paths:
            images_path_bank.append(path_i)
        feat_bank.append(i_feat)

    feat_bank = torch.cat(feat_bank, dim=0)
    torch.save(feat_bank, "feat_bank.pt")
    print(feat_bank.shape)
else:
    feat_bank = torch.load("feat_bank.pt")

feat_bank = feat_bank.float()
cosine_sim = torch.matmul(feat_bank, torch.transpose(image_features, 0, 1)) 
cosine_sim = cosine_sim.flatten()
print(f"cosine_sim {cosine_sim.shape}")

value, index = torch.topk(cosine_sim, 10)

print(f"topk 10 similar:")
for i in index:
    print(images_path_bank[i])
    # print(i)


    