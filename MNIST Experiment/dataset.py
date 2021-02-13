from __future__ import print_function
import os
import numpy as np
import pandas as pd
import pickle
import csv
from PIL import Image
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

def load_mnist_from_foler(batch_size, d_ratio, folder):
    img_size=28
    transform=transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize(img_size),transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    datasize = 10000
    indices = list(range(datasize))
    test_indices = indices[:d_ratio]
    test_dataset = datasets.ImageFolder(folder, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,sampler=SubsetRandomSampler(test_indices))
    
    return test_dataloader

