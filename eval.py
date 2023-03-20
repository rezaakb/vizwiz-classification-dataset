import os
import argparse
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as torch_models

import timm 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--ann_path', default='dataset/annotations.json')
    parser.add_argument('-i', '--images_path', default='dataset/images')
    parser.add_argument('-m', '--model_name', default='vgg19')
    args = parser.parse_args()
    
    in_to_vizwiz = json.load(open(os.path.join(args.ann_path,'IN_to_VizWiz.json')))
    indices_in_1k = list(map(int, in_to_vizwiz.keys()))

    flaws_dict = {'OBS': 0,
                  'DRK': 1,        
                  'BLR': 2,
                  'BRT': 3,
                  'ROT': 4,
                  'FRM': 5,
                  'NON': 6}
    

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])       


    class VizWizClassification(Dataset):
        def __init__(self, transform=None):

            ann = json.load(open(args.ann_path))

            all_images = list(ann.keys())

            self.labels = torch.zeros((len(all_images),200))
            self.images = []
            self.flaws = torch.zeros((len(all_images),7)).to(torch.bool)

            for j, image_id in enumerate(all_images):
                self.images.append(os.path.join(args.images_path,image_id))
                for i in ann[image_id]['labels']:
                    self.labels[j,in_to_vizwiz[str(i)]]=1
                if ann[image_id]['flaws'] is not None:
                    for key,value in ann[image_id]['flaws'].items():
                        if key!='OTH':
                            self.flaws[j,flaws_dict[key]]=value

            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = Image.open(self.images[idx]).convert('RGB')
            label = self.labels[idx]
            flaws = self.flaws[idx]
            if self.transform:
                image = self.transform(image)

            return image, label, flaws




    dataset = VizWizClassification(test_transform)

    vizwiz_loader = torch.utils.data.DataLoader(dataset,batch_size=64, shuffle=False)
    
    print('Dataset is loaded.')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def train(model, vizwiz_loader):

        correct = np.zeros(8)
        total = np.zeros(8)

        with torch.no_grad():

            for images, labels, flaws in vizwiz_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)[:,indices_in_1k]

                pred = outputs.data.max(1)[1]

                tmp = labels.gather(1, pred.unsqueeze(1))

                correct[7] += tmp.sum()
                total[7] += tmp.size(0)

                for j in range(7):
                    correct[j] += tmp[flaws[:,j]].sum()
                    total[j] += (flaws[:,j]).sum()

        return correct/total

    model = timm.create_model(args.model_name, pretrained=True).to(device)
    model.eval()
    
    print('Model is loaded.')
    print('Training started.')
    
    results = train(model, vizwiz_loader)

    print('The accuracy of',args.model_name,'on VizWiz-Classification =============')

    types = ['Obscured', 'Dark', 'Blur', 'Bright', 'Rotated', 'Framing', 'Clean Images', 'Corrupted Images', 'All Images']

    for i in range(8):
        if i!=7:
            print(types[i],'->',round(100*acc[i], 4))
        else:
            print(types[i],'->',round(100*np.mean(acc[0:6]),4))
        
if __name__ == '__main__':
    main()
