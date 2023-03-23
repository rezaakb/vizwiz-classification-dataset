import os
import argparse
import json
import numpy as np
from PIL import Image
from datetime import datetime

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
    parser.add_argument('-p', '--prediction_path', default='prediction')
    parser.add_argument('-m', '--model_name', default='vgg19')
    parser.add_argument('-b', '--batch_size', type=int, default='64')

    args = parser.parse_args()
    
    
    annotations = json.load(open(args.ann_path))    
    indices_in_1k = [d['id'] for d in annotations['categories']]

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])       

    class VizWizClassification(Dataset):
        def __init__(self, annotations, transform=None):
            self.images = [os.path.join(args.images_path,str(path)) for path in annotations['images']]
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = Image.open(self.images[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.images[idx].split("/")[2]

    dataset = VizWizClassification(annotations,test_transform)
    vizwiz_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, shuffle=False)

    
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
    print('Testing started.')
    
    results = []
    
    with torch.no_grad():
        for images, images_path in vizwiz_loader:
            images = images.to(device)
            outputs = model(images)[:,indices_in_1k]
            pred = list(outputs.data.max(1)[1].cpu())
            results.extend([(images_path[i],indices_in_1k[pred[i]]) for i in range(len(pred))])
            
    file_path = os.path.join(args.prediction_path, datetime.now().strftime("prediction-%m-%d-%Y-%H:%M:%S.json"))
    with open(file_path, 'w') as outfile:
        json.dump(results, outfile)
    
    print('Prediction file saved in', file_path)
    
if __name__ == '__main__':
    main()
