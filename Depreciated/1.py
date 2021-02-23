
import pickle
import os
import io
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms, utils
from skimage import transform
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional


image_dir = r"C:\Users\Leo's PC\Desktop\images"
csv_dir = r"C:\Users\Leo's PC\Desktop\AVA.txt"



class AVADataset(Dataset):
    """AVA dataset."""

    def __init__(self, csv_file, file_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            file_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.csv = pd.read_csv(csv_file, sep=' ')
        self.file_dir = file_dir
        self.transform = transform
        self.diction = {}
        for _, _, files in os.walk(self.file_dir):
            self.img_name_array = files
        print(type(self.img_name_array))
        for index, row in self.csv.iterrows():
            image_name = row[1]
            if (str(image_name) + '.jpg') in self.img_name_array:
                rating_array = np.array(row[2:12])
                avg_rat = np.argmax(rating_array) + 1
                self.diction[index] = [image_name, avg_rat]
                if index % 10000 == 0:
                    print('idx', index, 'img_name', image_name, 'avg_rat', avg_rat)

    def __len__(self):
        return len(self.diction)

    def __getitem__(self, idx):
        img_name = self.diction[idx][0]
        rat_avg = self.diction[idx][1]
        directory = self.file_dir + "\\" + str(img_name) + '.jpg'
        image = cv2.imread(directory, cv2.IMREAD_COLOR)
        sample = {'image': np.array(image, dtype=float), 'rating': np.array(rat_avg, dtype=float)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'rating': rating}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        rating = np.array(rating)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'rating': torch.from_numpy(rating)}



AVA = AVADataset(csv_file=csv_dir, file_dir=image_dir, transform=transforms.Compose([Rescale((299, 299)), ToTensor()]))


train_loader = DataLoader(dataset=AVA, batch_size=10, shuffle=True)
val_loader = DataLoader(dataset=AVA, batch_size=10, shuffle=False)


inception_v3 = torchvision.models.inception_v3(pretrained=True)
inception_v3.fc = nn.Linear(in_features=inception_v3.fc.in_features, out_features=1024)



class LeoNet(nn.Module):
    def __init__(self):
        super(LeoNet, self).__init__()
        self.inception = inception_v3  # input shape is 299*299*3
        self.fc1 = nn.Linear(in_features=1024, out_features=512, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.fc3 = nn.Linear(in_features=256, out_features=64, bias=True)
        self.fc4 = nn.Linear(in_features=64, out_features=16, bias=True)
        self.fc5 = nn.Linear(in_features=16, out_features=1, bias=True)

    def forward(self, x):
        x, _ = inception_v3(x)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

    def name(self):
        return "LeoNet"



model = LeoNet()
'''
if torch.cuda.device_count() > 1:
 print("Let's use", torch.cuda.device_count(), "GPUs!")
 model = nn.DataParallel(model)
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()


for epoch in range(10):
    ave_loss = 0
    for batch_idx, diction in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = diction['image'], diction['rating']
        target = target.resize_([10])
        x, target = x.float(), target.long()
        x, target = x.to(device), target.to(device)
        x, target = Variable(x), Variable(target)
        out = model(x) * 10
        if (batch_idx + 1) % 100 == 0:
            print(target, out)
        loss = criterion(out, target)
        ave_loss = ave_loss * 0.9 + loss.item() * 0.1
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx + 1, loss))

    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, diction in enumerate(val_loader):
        x, target = diction['image'], diction['rating']
        x, target = x.float(), target.long()
        x, target = x.to(device), target.to(device)
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.item() * 0.1

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(val_loader):
            print(
                '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                    epoch, batch_idx + 1, ave_loss, correct_cnt.item() * 1.0 / total_cnt))

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['rating'].size())
    img = sample_batched['image']
    img = np.array(img[2]).transpose((1, 2, 0))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if i_batch == 3:
        break

for i in range(len(AVA)):
    sample = AVA[i]

    print(i, sample['image'].size(), sample['rating'].size())

    if i == 3:
        break
