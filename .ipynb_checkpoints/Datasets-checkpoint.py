#!/usr/bin/env python
# coding: utf-8


import os
import io
import time
import pickle
import pandas as pd
import numpy as np
import shutil

import cv2
from skimage import transform
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import Datasets


class Rescale(object):
    
    """
    Rescale the image in a sample to a given size.

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

        img = cv2.resize(image, (new_h, new_w))

        return {'image': img, 'rating': rating}


class ToTensor(object):
    
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        rating = np.array(rating)
        image = image.transpose((2, 0, 1)) #swap color axis because: numpy image: H x W x C & torch image: C X H X W
        return {'image': torch.from_numpy(image),
                 'rating': torch.from_numpy(rating)}

    
class AADBDataset(Dataset):
    """AADB dataset."""

    def __init__(self, csv_file, file_dir, start, end, class_size=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            file_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            start&end: specify the range in the raw data to construct the dataset from. For spliting calidation and test sets.
            class_size = make the amount of images in each rating group equal to class_size.
        """
        self.csv = pd.read_csv(csv_file, sep=' ')
        self.file_dir = file_dir
        self.transform = transform
        self.diction = {}
        self.start = start
        self.end = end
        self.class_size = class_size
        self.img_name_array = []

        for files in os.listdir(self.file_dir): #get a list of file names in the dataset folder
            self.img_name_array.append(files)

        self.img_name_array = self.img_name_array[start:end] #limit the range of file names to start:end
        print(len(self.img_name_array), 'images found in image dir')
        print(len(self.csv), 'entries found in the CSV file')

        index = 0
        self.rat_distribution = np.zeros(10, dtype=np.int64)
        print("index =", index, "AADB Dataset initialization begin...")
        print("Rating distribution initialized: ", self.rat_distribution)
        for csv_idx, row in self.csv.iterrows(): #traverse the enitre csv file
            image_name = row[0]
            if (str(image_name)) in self.img_name_array: 
                #only add [image_name, avg_rat] to diction when it's in the img_name_array
               
                avg_rat = row[1]
                if class_size: #when it's needed to limit class amount
                    if self.rat_distribution[int(np.floor(avg_rat * 10) - 1)] < class_size: #only enter if under limit
                        self.diction[index] = [image_name, avg_rat] #append
                        self.rat_distribution[int(np.floor(avg_rat * 10) - 1)] += 1 #update rat_distribution
                        index += 1 #update index
                else: #when class amount is not specified. Just append.
                    self.diction[index] = [image_name, avg_rat] #append
                    self.rat_distribution[int(np.floor(avg_rat * 10) - 1)] += 1 #update rat_distribution
                    index += 1 #update index
                if csv_idx % 1000 == 0:
                    print('csv_idx:', csv_idx, 'index:', index, 'img_name', image_name, 'avg_rat', avg_rat,"\n",
                          " - Current rating distribution is: ", self.rat_distribution)
            
        print("Stage 1 loading complete. Current rating distribution is: ", self.rat_distribution)

        plt.bar(np.arange(10), self.rat_distribution, 0.35) #(indeces, data, width)
        plt.ylabel('Number of Pictures')
        plt.title('Current Rating Distribution')
        plt.xticks(np.arange(10), ('0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'))
        plt.show()
        
        #now that all classes are <= class_limit, we need to make all of them = class limit
        if class_size: #only proceed when it's needed to limit class amount
            self.distribution_multiplier = np.ceil(np.divide(np.full(10, self.class_size), self.rat_distribution))
            #np.ceil rounds up to provide enough images in each class. np.fill creates an array with 9 elements = class_size
            print("Stage 2 begin. Target class size is:", class_size, "Distribution multiplier is:", self.distribution_multiplier)
            self.additional_diction = {}
            for item in self.diction.items(): #traverse the self.diction
                img_name = item[1][0]
                avg_rat = item[1][1]
                if self.rat_distribution[avg_rat - 2] < class_size: #only enter if < class_limit
                    for i in range(int(self.distribution_multiplier[avg_rat - 2])):
                        self.additional_diction[len(self.diction) + len(self.additional_diction)] = [image_name, avg_rat]
                        #append the same image at the end of additional_diction (distribution_multiplier) times
                    self.rat_distribution[avg_rat - 2] += self.distribution_multiplier[avg_rat - 2] #update rat_distribution
            self.diction.update(self.additional_diction) #combine diction and additional_diction
                    
        print("AADB Dataset initialization complete. Rating distribution is: ", self.rat_distribution, "\n",
        "contains", len(self.diction), "items.")
        
        plt.bar(np.arange(10), self.rat_distribution, 0.35) #(indeces, data, width)
        plt.ylabel('Number of Pictures')
        plt.title('Current Rating Distribution')
        plt.xticks(np.arange(10), ('0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'))
        plt.show()
                        
                        
    def __len__(self):
        return len(self.diction)

    
    def __getitem__(self, idx):
        img_name = self.diction[idx][0]
        rat_avg = self.diction[idx][1]
        directory = self.file_dir + "\\" + str(img_name)
        image = cv2.imread(directory, cv2.IMREAD_COLOR)
        sample = {'image': np.array(image, dtype=float), 'rating': np.array(rat_avg, dtype=float)}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
class AVADataset(Dataset):
    """AVA dataset."""

    def __init__(self, csv_file, file_dir, start, end, class_size=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            file_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            start&end: specify the range in the raw data to construct the dataset from. For spliting calidation and test sets.
            class_size = make the amount of images in each rating group equal to class_size.
        """
        self.csv = pd.read_csv(csv_file, sep=' ')
        self.file_dir = file_dir
        self.transform = transform
        self.diction = {}
        self.start = start
        self.end = end
        self.class_size = class_size

        for _, _, files in os.walk(self.file_dir): #get a list of file names in the dataset folder
            self.img_name_array = files

        self.img_name_array = self.img_name_array[start:end] #limit the range of file names to start:end

        index = 0
        self.rat_distribution = []
        
        print("index =", index, "AVA Dataset initialization begin...")
        print("Rating distribution initialized: ", self.rat_distribution)
        
        for csv_idx, row in self.csv.iterrows(): #traverse the enitre csv file
            image_name = row[1]
            
            if (str(image_name) + '.jpg') in self.img_name_array: 
                #only add [image_name, avg_rat] to diction when it's in the img_name_array
               
                rating_array = np.array(row[2:12])
                
                # take average
                total_num_ratings = np.sum(rating_array)
                rating_array = np.multiply(rating_array, np.arange(0, 10))
                avg_rat = np.divide(np.sum(rating_array), total_num_ratings)
                
                '''
                #taking the mode
                avg_rat = np.argmax(rating_array) + 1
                '''
                
                if class_size: #when it's needed to limit class amount
                    if self.rat_distribution[avg_rat - 1] < class_size: #only enter if under limit
                        self.diction[index] = [image_name, avg_rat] #append
                        self.rat_distribution.append(avg_rat) #update rat_distribution
                        index += 1 #update index
                        
                else: #when class amount is not specified. Just append.
                    self.diction[index] = [image_name, avg_rat] #append
                    self.rat_distribution.append(avg_rat) #update rat_distribution
                    index += 1 #update index
                    
                # if csv_idx % 10000 == 0:
                    # print('csv_idx:', csv_idx, 'index:', index, 'img_name', image_name, 'avg_rat', avg_rat,"\n", " - Current rating distribution is: ", self.rat_distribution)
            
        # print("Stage 1 loading complete. Current rating distribution is: ", self.rat_distribution)
        
        #print(len(self.rat_distribution))
        print(self.rat_distribution[0])
        
        plt.hist(self.rat_distribution) #(indeces, data, width)
        #plt.ylabel('Number of Pictures')
        #plt.title('Current Rating Distribution')
        #plt.xticks(np.arange(10), ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
        plt.show()
        
        #now that all classes are <= class_limit, we need to make all of them = class limit
        if class_size: #only proceed when it's needed to limit class amount
            self.distribution_multiplier = np.ceil(np.divide(np.full(10, self.class_size), self.rat_distribution))
            #np.ceil rounds up to provide enough images in each class. np.fill creates an array with 9 elements = class_size
            print("Stage 2 begin. Target class size is:", class_size, "Distribution multiplier is:", self.distribution_multiplier)
            
            self.additional_diction = {}
            for item in self.diction.items(): #traverse the self.diction
                img_name = item[1][0]
                avg_rat = item[1][1]
                
                if self.rat_distribution[avg_rat - 1] < class_size: #only enter if < class_limit
                    for i in range(int(self.distribution_multiplier[avg_rat - 2])):
                        self.additional_diction[len(self.diction) + len(self.additional_diction)] = [image_name, avg_rat]
                        #append the same image at the end of additional_diction (distribution_multiplier) times
                    self.rat_distribution[avg_rat - 1] += self.distribution_multiplier[avg_rat - 2] #update rat_distribution
            self.diction.update(self.additional_diction) #combine diction and additional_diction
                    
        # print("AVA Dataset initialization complete. Rating distribution is: ", self.rat_distribution, "\n", "contains", len(self.diction), "items.")
        
        #plt.bar(np.arange(10), self.rat_distribution, 0.5) #(indeces, data, width)
        #plt.ylabel('Number of Pictures')
        #plt.title('Score Distribution of the AVA Dataset')
        #plt.xticks(np.arange(10), ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
        #plt.show()
                        
                        
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
    
    
class AVABinaryDataset(Dataset):
    """AVA dataset."""

    def __init__(self, csv_file, file_dir, start, end, class_size=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            file_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            start&end: specify the range in the raw data to construct the dataset from. For spliting calidation and test sets.
            class_size = make the amount of images in each rating group equal to class_size.
        """
        self.csv = pd.read_csv(csv_file, sep=' ')
        self.file_dir = file_dir
        self.transform = transform
        self.diction = {}
        self.start = start
        self.end = end
        self.class_size = class_size

        for _, _, files in os.walk(self.file_dir): #get a list of file names in the dataset folder
            self.img_name_array = files

        self.img_name_array = self.img_name_array[start:end] #limit the range of file names to start:end

        index = 0
        self.rat_distribution = np.zeros(2, dtype=np.int64)
        print("index =", index, "AVA Dataset initialization begin...")
        print("Rating distribution initialized: ", self.rat_distribution)
        
        for csv_idx, row in self.csv.iterrows(): #traverse the enitre csv file
            image_name = row[1]
            if (str(image_name) + '.jpg') in self.img_name_array: 
                #only add [image_name, avg_rat] to diction when it's in the img_name_array     
                rating_array = np.array(row[2:12])
                
                avg_rat = np.argmax(rating_array) + 1
                
                if avg_rat >= 7:
                    avg_rat = 1
                elif avg_rat <= 4:
                    avg_rat = 0
                else: avg_rat = -1
                
                if class_size: #when it's needed to limit class amount
                    if avg_rat != -1 and self.rat_distribution[avg_rat] < class_size: #only enter if under limit
                        self.diction[index] = [image_name, avg_rat] #append
                        self.rat_distribution[avg_rat] += 1 #update rat_distribution
                        index += 1 #update index
                else: #when class amount is not specified. Just append.
                    self.diction[index] = [image_name, avg_rat] #append
                    self.rat_distribution[avg_rat] += 1 #update rat_distribution
                    index += 1 #update index
                if csv_idx % 10000 == 0:
                    print('csv_idx:', csv_idx, 'index:', index, 'img_name', image_name, 'avg_rat', avg_rat,"\n",
                          " - Current rating distribution is: ", self.rat_distribution)
            
        print("Stage 1 loading complete. Current rating distribution is: ", self.rat_distribution)

        plt.bar(np.arange(2), self.rat_distribution, 0.5) #(indeces, data, width)
        plt.ylabel('Number of Pictures')
        plt.title('Score Distribution of AVA Binary')
        plt.xticks(np.arange(2), ('0', '1'))
        plt.show()
        
        #now that all classes are <= class_limit, we need to make all of them = class limit
        if class_size: #only proceed when it's needed to limit class amount
            self.distribution_multiplier = np.ceil(np.divide(np.full(2, self.class_size), self.rat_distribution))
            #np.ceil rounds up to provide enough images in each class. np.fill creates an array with 9 elements = class_size
            print("Stage 2 begin. Target class size is:", class_size, "Distribution multiplier is:", self.distribution_multiplier)
            self.additional_diction = {}
            for item in self.diction.items(): #traverse the self.diction
                img_name = item[1][0]
                avg_rat = item[1][1]
                if self.rat_distribution[avg_rat - 1] < class_size: #only enter if < class_limit
                    for i in range(int(self.distribution_multiplier[avg_rat - 2])):
                        self.additional_diction[len(self.diction) + len(self.additional_diction)] = [image_name, avg_rat]
                        #append the same image at the end of additional_diction (distribution_multiplier) times
                    self.rat_distribution[avg_rat - 1] += self.distribution_multiplier[avg_rat - 2] #update rat_distribution
            self.diction.update(self.additional_diction) #combine diction and additional_diction
                    
        print("AVA Dataset initialization complete. Rating distribution is: ", self.rat_distribution, "\n",
        "contains", len(self.diction), "items.")
        
        plt.bar(np.arange(2), self.rat_distribution, 0.5) #(indeces, data, width)
        plt.ylabel('Number of Pictures')
        plt.title('Score Distribution of the AVA Dataset')
        plt.xticks(np.arange(2), ('0', '1'))
        plt.show()
                        
                        
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
    
    
class AVABinaryDataset_ten(Dataset):
    """AVA dataset."""

    def __init__(self, csv_file, file_dir, start, end, percent=0.1, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            file_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            start&end: specify the range in the raw data to construct the dataset from. For spliting calidation and test sets.
            class_size = make the amount of images in each rating group equal to class_size.
        """
        self.csv = pd.read_csv(csv_file, sep=' ')
        self.file_dir = file_dir
        self.transform = transform
        self.diction = {}
        self.start = start
        self.end = end
        self.percent = percent
        self.images2include = {}
        self.img_name_array = []

        for files in os.listdir(self.file_dir): #get a list of file names in the dataset folder
            self.img_name_array.append(files)

        # self.img_name_array = self.img_name_array[start:end] #limit the range of file names to start:end

        self.rat_distribution = np.zeros(2, dtype=np.int64)

        for csv_idx, row in self.csv.iterrows(): #traverse the enitre csv file
            image_name = row[1]
            if (str(image_name) + '.jpg') in self.img_name_array:    
                rating_array = np.array(row[2:12])

                total_num_ratings = np.sum(rating_array)
                rating_array = np.multiply(rating_array, np.arange(0, 10))
                avg_rat = np.divide(np.sum(rating_array), total_num_ratings)

                self.images2include[str(image_name)] = avg_rat
       
        self.images2include = dict(list(self.images2include.items())[self.start:self.end])
        self.images2include = dict(sorted(self.images2include.items(), key=lambda x: x[1]))
        print(len(self.images2include), 'images are found within the specified range')
        print('Extremes:', list(self.images2include.items())[0], list(self.images2include.items())[-1])

        self.bottom = dict(list(self.images2include.items())[0:int(self.percent * len(self.images2include))])
        self.top = dict(list(self.images2include.items())[int((1-self.percent) * len(self.images2include)):len(self.images2include)])
        
        print(len(self.bottom), 'images with label 0.', len(self.top), 'images with label 1.')
        
        self.idx = 0
        for img_name, rating in self.bottom.items():
            self.diction[self.idx] = [img_name, 0]
            self.rat_distribution[0] += 1
            self.idx += 1
        for img_name, rating in self.top.items():
            self.diction[self.idx] = [img_name, 1]
            self.rat_distribution[1] += 1
            self.idx += 1
        
        plt.bar(np.arange(2), self.rat_distribution, 0.5) #(indeces, data, width)
        plt.ylabel('Number of Pictures')
        plt.title('Score Distribution of the AVA Dataset')
        plt.xticks(np.arange(2), ('0', '1'))
        plt.show()
                        
                        
    def __len__(self):
        return len(self.diction)

    
    def __getitem__(self, idx):
        img_name = self.diction[idx][0]
        rat_avg = self.diction[idx][1]
        directory = os.path.join(self.file_dir, str(img_name) + '.jpg')
        image = cv2.imread(directory, cv2.IMREAD_COLOR)
        sample = {'image': image, 'rating': np.array(rat_avg, dtype=float)}
        if self.transform:
            sample = self.transform(sample)
        return sample
    

class Rescale_list(object):
    
    """
    Rescale the image in a sample to a given size.

    Args:
     output_size (tuple or int): Desired output size. If tuple, output is
         matched to output_size. If int, smaller of image edges is matched
         to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imagelist = sample['image']
        
        outlist = []
        
        for image in imagelist:
            h, w = image.shape[:2]

            new_h, new_w = self.output_size
            new_h, new_w = int(new_h), int(new_w)

            outlist.append(cv2.resize(image, (new_h, new_w)))
            
        outlist[0] = cv2.resize(imagelist[0], (488, 448))
                
        return {'image': outlist, 'scene':sample['scene'], 'rating':sample['rating']}


class ToNumpy_list(object):
    
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        imagelist, scene, rating = sample['image'], sample['scene'], sample['rating']
        
        scene = np.array(scene)
        rating = np.array(rating)
    
        outlist = []
        
        for image in imagelist:
            if len(image.shape) == 3:
                image = image.transpose((2, 0, 1)) #swap color axis because: numpy image: H x W x C & torch image: C X H X W
            elif len(image.shape) == 2:
                image = np.expand_dims(image, axis=0)
            image = np.array(image)
            outlist.append(image)
            
        return {'image': outlist, 'scene':scene, 'rating': torch.from_numpy(rating)}
    

def clean_cache(cache_folder):
    file_list = os.listdir(cache_folder)

    if len(file_list) > 260000:
        raise RuntimeError('Too many images. Check directory.')

    start_time = time.time()
    print('Cleaning', len(file_list), 'cached files.')

    for file in file_list:
        os.remove(os.path.join(cache_folder, file))

    elapsed_time = time.time() - start_time
    print('Cleaning complete. Took', elapsed_time, 'seconds.')
    
    
class AVAFeatureDataset_Binary_percent(Dataset):
    """AVA dataset."""

    def __init__(self, csv_file, file_dir, pkl_dir, start, end, percent=0.1, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            file_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            start&end: specify the range in the raw data to construct the dataset from. For spliting calidation and test sets.
            class_size = make the amount of images in each rating group equal to class_size.
        """
        self.csv = pd.read_csv(csv_file, sep=' ')
        self.file_dir = file_dir
        self.pkl_dir = pkl_dir
        self.transform = transform
        self.diction = {}
        self.start = start
        self.end = end
        self.percent = percent
        self.images2include = {}
        self.img_name_array = []

        self.img_name_array = os.listdir(self.file_dir) #get a list of subfolder names in the dataset folder
        print(len(self.img_name_array), 'images found in the feature map directory.')
        
        self.CACHED = False
        self.CACHED_TRANSFORM = False

        self.rat_distribution = np.zeros(2, dtype=np.int64)

        for csv_idx, row in self.csv.iterrows(): #traverse the enitre csv file
            image_name = row[1]
            if (str(image_name)) in self.img_name_array:    
                rating_array = np.array(row[2:12])

                total_num_ratings = np.sum(rating_array)
                rating_array = np.multiply(rating_array, np.arange(0, 10))
                avg_rat = np.divide(np.sum(rating_array), total_num_ratings)

                self.images2include[str(image_name)] = avg_rat
       
        self.images2include = dict(list(self.images2include.items())[self.start:self.end])
        self.images2include = dict(sorted(self.images2include.items(), key=lambda x: x[1]))
        print(len(self.images2include), 'images are found within the specified range')
        print('Extremes:', list(self.images2include.items())[0], list(self.images2include.items())[-1])

        self.bottom = dict(list(self.images2include.items())[0:int(self.percent * len(self.images2include))])
        self.top = dict(list(self.images2include.items())[int((1-self.percent) * len(self.images2include)):len(self.images2include)])
        
        print(len(self.bottom), 'images with label 0.', len(self.top), 'images with label 1.')
        
        self.idx = 0
        for img_name, rating in self.bottom.items():
            self.diction[self.idx] = [img_name, 0]
            self.rat_distribution[0] += 1
            self.idx += 1
        for img_name, rating in self.top.items():
            self.diction[self.idx] = [img_name, 1]
            self.rat_distribution[1] += 1
            self.idx += 1
        
        with open('./2048nodes.pkl', 'rb') as fp:
            self.scene2048 = pickle.load(fp)
        
        plt.bar(np.arange(2), self.rat_distribution, 0.5) #(indeces, data, width)
        plt.ylabel('Number of Pictures')
        plt.title('Score Distribution of the AVA Dataset')
        plt.xticks(np.arange(2), ('0', '1'))
        plt.show()
                        
                        
    def __len__(self):
        return len(self.diction)

    
    def __getitem__(self, idx):
        img_name = self.diction[idx][0]
        rat_avg = self.diction[idx][1]
        
        if self.CACHED: 
            directory = self.cache_folder
        else:
            directory = self.pkl_dir
            
        with open(os.path.join(directory, str(img_name) + '.pkl'), 'rb') as fp:
            image = pickle.load(fp)

        sample = {'image': image, 'scene':self.scene2048[int(img_name)], 'rating': np.array(rat_avg, dtype=float)}
        
        if self.transform and not self.CACHED_TRANSFORM:
            sample = self.transform(sample)
            
        return sample     
    
    
    def build_cache(self, cache_folder, TRANSFORM=True):
        self.cache_folder = cache_folder
        
        max_possible_allocation = shutil.disk_usage(cache_folder)[2] / 1000000 * 0.9 / 1.6

        if len(self.diction) > max_possible_allocation:
            raise RuntimeError('Not enough disk space for caching.')

        start_time = time.time()
        print('Start building cache')

        if not TRANSFORM:
            self.CACHED_TRANSFORM = False
            for _, item in self.diction.items():
                image_name = item[0]
                shutil.copyfile(os.path.join(self.pkl_dir, str(image_name) + '.pkl'), os.path.join(cache_folder, str(image_name) + '.pkl'))

        else:
            self.CACHED_TRANSFORM = True
            
            for _, item in self.diction.items():
                
                image_name = item[0]
                
                with open(os.path.join(self.pkl_dir, str(image_name) + '.pkl'), 'rb') as fp:
                        image = pickle.load(fp)

                sample = {'image': image, 'scene':self.scene2048[1000], 'rating': np.array(0, dtype=float)}   

                sample = self.transform(sample)

                with open(os.path.join(cache_folder, str(image_name) + '.pkl'), 'wb') as fp:
                    pickle.dump(sample['image'], fp)
            
        elapsed_time = time.time() - start_time
        print('Cacheing complete. Took', elapsed_time, 'seconds.')
        
        self.CACHED = True
        
        
    def clean_cache(self):
        clean_cache(self.cache_folder)
        
        self.CACHED = False
        
        
class AVAFeatureDataset(Dataset):
    """AVA Feature dataset with smooth score."""

    def __init__(self, csv_file, file_dir, pkl_dir, start, end, cache_folder, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            file_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            start&end: specify the range in the raw data to construct the dataset from. For spliting calidation and test sets.
            class_size = make the amount of images in each rating group equal to class_size.
        """
        
        self.csv = pd.read_csv(csv_file, sep=' ')
        self.file_dir = file_dir
        self.pkl_dir = pkl_dir
        self.transform = transform
        self.diction = {}
        self.start = start
        self.end = end
        self.images2include = {}
        self.img_name_array = []
        self.cache_folder = cache_folder

        self.img_name_array = os.listdir(self.file_dir) #get a list of subfolder names in the dataset folder
        print(len(self.img_name_array), 'images found in the feature map directory.')
        
        self.CACHED = False
        self.CACHED_TRANSFORM = False

        self.rat_distribution = []

        for csv_idx, row in self.csv.iterrows(): #traverse the enitre csv file
            image_name = row[1]
            if (str(image_name)) in self.img_name_array:    
                rating_array = np.array(row[2:12])
                
                
                # take average
                total_num_ratings = np.sum(rating_array)
                rating_array = np.multiply(rating_array, np.arange(0, 10))
                avg_rat = np.divide(np.sum(rating_array), total_num_ratings)
                
                
                # take mode
                # avg_rat = np.argmax(rating_array) + 1

                self.images2include[str(image_name)] = avg_rat
                self.rat_distribution.append(avg_rat)
       
        self.images2include = dict(list(self.images2include.items())[self.start:self.end])
        self.images2include = dict(sorted(self.images2include.items(), key=lambda x: x[1]))
        print(len(self.images2include), 'images are found within the specified range')
        print('Extremes:', list(self.images2include.items())[0], list(self.images2include.items())[-1])
        
        self.idx = 0
        for img_name, rating in self.images2include.items():
            self.diction[self.idx] = [img_name, rating]
            self.idx += 1

        with open('./2048nodes.pkl', 'rb') as fp:
            self.scene2048 = pickle.load(fp)
        
        plt.ylabel('Number of Pictures')
        plt.title('Score Distribution of the AVA Dataset')
        plt.hist(self.rat_distribution)
        plt.show()
                        
                        
    def __len__(self):
        return len(self.diction)

    
    def __getitem__(self, idx):
        img_name = self.diction[idx][0]
        rat_avg = self.diction[idx][1]
        
        if self.CACHED: 
            directory = self.cache_folder
        else:
            directory = self.pkl_dir
            
        with open(os.path.join(directory, str(img_name) + '.pkl'), 'rb') as fp:
            image = pickle.load(fp)

        sample = {'image': image, 'scene':self.scene2048[int(img_name)], 'rating': np.array(rat_avg, dtype=float)}
        
        if self.transform and not self.CACHED_TRANSFORM:
            sample = self.transform(sample)
            
        return sample     
    
    
    def build_cache(self, cache_folder, TRANSFORM=True):
        self.cache_folder = cache_folder
        
        max_possible_allocation = shutil.disk_usage(cache_folder)[2] / 1000000 * 0.9 / 1.6

        if len(self.diction) > max_possible_allocation:
            raise RuntimeError('Not enough disk space for caching.')

        start_time = time.time()
        print('Start building cache')

        if not TRANSFORM:
            self.CACHED_TRANSFORM = False
            for _, item in self.diction.items():
                image_name = item[0]
                shutil.copyfile(os.path.join(self.pkl_dir, str(image_name) + '.pkl'), os.path.join(cache_folder, str(image_name) + '.pkl'))

        else:
            self.CACHED_TRANSFORM = True
            
            for _, item in self.diction.items():
                
                image_name = item[0]
                
                with open(os.path.join(self.pkl_dir, str(image_name) + '.pkl'), 'rb') as fp:
                        image = pickle.load(fp)

                sample = {'image': image, 'scene':self.scene2048[1000], 'rating': np.array(0, dtype=float)}   

                sample = self.transform(sample)

                with open(os.path.join(cache_folder, str(image_name) + '.pkl'), 'wb') as fp:
                    pickle.dump(sample['image'], fp)
            
        elapsed_time = time.time() - start_time
        print('Cacheing complete. Took', elapsed_time, 'seconds.')
        
        self.CACHED = True
        
        
    def clean_cache(self):
        clean_cache(self.cache_folder)
        
        self.CACHED = False