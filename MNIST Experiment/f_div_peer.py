from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import argparse
import csv
from PIL import Image
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from skimage.util import random_noise
from dataset import *
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader, _imshow, TRAINED_MODEL_PATH
from advertorch.test_utils import LeNet5
from advertorch.attacks import LinfPGDAttack

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--seed', type=int, default=10086)
parser.add_argument('--epsilon', type=int)
parser.add_argument('--data_ratio', type=int, default=10000)
parser.add_argument('--noise', type=str)
parser.add_argument('--divergence', help='f-divergence', default = 'Total-Variation')
parser.add_argument('--batchsize', type=int, default=1)


# Hyper-parameters
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.manual_seed(args.seed)
batch_size = args.batchsize

hidden_size = 256
image_size = 784
batch_size = args.batchsize
d_ratio = args.data_ratio

def save_noisy_image(img, name):
    if img.size(1) == 3:
        img = img.view(img.size(0), 3, 32, 32)
        save_image(img, name)
    else:
        img = img.view(img.size(0), 1, 28, 28)
        save_image(img, name)

# Adopt True for the first time
generate_peer = True

# Generate peer images to folder
if generate_peer == True:
    filename = "mnist_lenet5_clntrained.pt"
    model_peer = LeNet5()
    model_peer.load_state_dict(
        torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
    model_peer.cuda()
    model_peer.eval()
    peer_loader = get_mnist_test_loader(batch_size=1, shuffle = False)
    adversary = LinfPGDAttack(model_peer, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    base_folder = f"PeerReport/MNIST/Peer_reference/1"
    os.makedirs(base_folder, exist_ok=True)
    count = 0
    for cln_data, true_label in peer_loader:
        cln_data, true_label = cln_data.cuda(), true_label.cuda()
        adv_untargeted = adversary.perturb(cln_data, true_label)
        image_path = os.path.join(base_folder, f"{count}.png")
        save_noisy_image(adv_untargeted, image_path)
        count += 1
    print("Generating Peer report finished!!!")
    

noise_type = args.noise
epsi = args.epsilon / 100

# Generate PGDA noise reports to folder
if noise_type == 'PGDA':
    filename = "mnist_lenet5_clntrained.pt"
    model = LeNet5()
    model.load_state_dict(
        torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
    model.cuda()
    model.eval()
    loader = get_mnist_test_loader(batch_size=1, shuffle = False)
    adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsi, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    base_folder = f"PeerReport/MNIST/PGDA/{epsi}/{epsi}/1"
    os.makedirs(base_folder, exist_ok=True)
    count = 0
    for cln_data, true_label in loader:
        cln_data, true_label = cln_data.cuda(), true_label.cuda()
        adv_untargeted = adversary.perturb(cln_data, true_label)
        image_path = os.path.join(base_folder, f"{count}.png")
        save_noisy_image(adv_untargeted, image_path)
        count += 1
    print("Generating Noise report finished!!!")

# Generate Gaussian/Speckle noise reports to folder
else:
    transform_noise = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    trainset_noise = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform_noise
        )
    trainloader_noise = torch.utils.data.DataLoader(trainset_noise,batch_size=1,shuffle=False
    )
    if noise_type == 'Gaussian':
        base_folder = f"PeerReport/MNIST/Gaussian/{epsi}/{epsi}/1"
        os.makedirs(base_folder, exist_ok=True)
        count = 0
        for cln_data, true_label in trainloader_noise:
            image_path = os.path.join(base_folder, f"{count}.png")
            gauss_img = torch.tensor(random_noise(cln_data, mode='gaussian', mean=0, var=epsi, clip=True))
            save_noisy_image(gauss_img, image_path)
            count += 1
        print("Generating Noise report finished!!!")
    elif noise_type == 'Speckle':
        base_folder = f"PeerReport/MNIST/Speckle/{epsi}/{epsi}/1"
        os.makedirs(base_folder, exist_ok=True)
        count = 0
        for cln_data, true_label in trainloader_noise:
            image_path = os.path.join(base_folder, f"{count}.png")
            gauss_img = torch.tensor(random_noise(cln_data, mode='speckle', mean=0, var=epsi, clip=True))
            save_noisy_image(gauss_img, image_path)
            count += 1
        print("Generating Noise report finished!!!")


mnist_folder = f"PeerReport/MNIST/Peer_reference"
mnist_loader = load_mnist_from_foler(batch_size, d_ratio, mnist_folder)

# Noise type: PGDA, Gaussian, Speckle
noise_folder = f"PeerReport/MNIST/{noise_type}/{epsi}/{epsi}"
noise_loader = load_mnist_from_foler(batch_size, d_ratio, noise_folder)


# Discriminator: load a public available discriminator model which is pre-trained on the MNIST training dataset
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())
D.load_state_dict(torch.load('D.ckpt'))
D = D.cuda()

# Variation form of f-divergence losses
class f_divergence:
    def __init__(self, div_name):
        self.div_name = div_name

    def actv(self, DX_score, DG_score):
        """ Compute batch loss for discriminator using f-divergence metric """

        if self.div_name == 'Total-Variation':
            return -(torch.mean(0.5*torch.tanh(DX_score)) \
                        - torch.mean(0.5*torch.tanh(DG_score)))

        elif self.div_name == 'KL':
            return -(torch.mean(DX_score) - torch.mean(torch.exp(DG_score-1)))

        elif self.div_name == 'Reverse-KL':
            return -(torch.mean(-torch.exp(DX_score)) - torch.mean(-1-DG_score))

        elif self.div_name == 'Pearson':
            return -(torch.mean(DX_score) - torch.mean(torch.mul(DG_score, DG_score) / 4. + DG_score))

        elif self.div_name == 'Squared-Hellinger':
            return -(torch.mean(1-torch.exp(DX_score)) \
                        - torch.mean((1-torch.exp(DG_score))/(torch.exp(DG_score))))

        elif self.div_name == 'Jenson-Shannon':
            return -(torch.mean(- torch.log(1. + torch.exp(-DX_score))) + torch.log(torch.tensor(2.)) - torch.mean(DG_score + torch.log(1. + torch.exp(-DG_score))) - torch.log(torch.tensor(2.)))


# Calculate f-score based on ground-truth reports and submitted noisified reports
class Cal_f_score:
    """ Object to hold data iterators, calculate f-scores
    """
    def __init__(self, train_iter, noise_iter):

        self.train_iter = train_iter
        self.noise_iter = noise_iter

    def eval(self, div_name):
        
        self.loss_fnc = f_divergence(div_name)
        epoch_steps = len(self.train_iter)
        D_losses = []
        for _ in tqdm(range(epoch_steps)):
            cln_images = self.process_batch(self.train_iter)
            noise_images = self.process_batch(self.noise_iter)
            DX_score = D(cln_images)
            DG_score = D(noise_images)
            D_loss = self.loss_fnc.actv(DX_score/2 + DG_score/2, DX_score * DG_score)
            D_losses.append(D_loss.item())
        df = pd.DataFrame(D_losses, columns = ['Payment'])
        df.to_csv(f"PeerResults/{noise_type}_{d_ratio}_{epsi}_{div_name}.csv", index = False)


    def process_batch(self, iterator):
        images, _ = next(iter(iterator))
        images = (images.view(images.shape[0], -1)).cuda()
        return images
    
     

if __name__ == '__main__':

    # Init trainer
    mechanism = Cal_f_score(mnist_loader, noise_loader)
    
    # Calculate f-score payment
    mechanism.eval(args.divergence)
    print("All done!!!")

