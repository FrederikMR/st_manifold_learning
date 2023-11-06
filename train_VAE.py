# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 23:04:19 2021

@author: Frederik
"""

#%% Sources

"""
Sources:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

#%% Modules

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse
import numpy as np

#Own files
import ManLearn

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--model', default="DTU",
                        type=str)
    parser.add_argument('--path', default="../../../../Data/",
                        type=str)
    parser.add_argument('--save_step', default=100,
                        type=int)
    
    #Training-size
    parser.add_argument('--num_img', default=0.8, #0.8
                        type=float)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--workers', default=2,
                        type=int)
    parser.add_argument('--epochs', default=50000, #50000
                        type=int)
    parser.add_argument('--batch_size', default=100,
                        type=int)
    parser.add_argument('--lr', default=0.0002,
                        type=float)

    #Continue training or not
    parser.add_argument('--con_training', default=0,
                        type=int)


    args = parser.parse_args()
    return args

#%% Main loop

def main():

    args = parse_args()
    train_loss_elbo = [] #Elbo loss
    train_loss_rec = [] #Reconstruction loss
    train_loss_kld = [] #KLD loss
    epochs = args.epochs

    if args.model == "SVHN":
        model = ManLearn.VAE_SVHN().to(args.device) #Model used
        data_path = ''.join((args.path, 'SVHN/'))
        model_path = ''.join((args.path, 'SVHN/params.pt'))
        dataset, dataset_subset, trainloader = ManLearn.load_svhn(data_path,
                                                                  batch_size=args.batch_size,
                                                                  img_size=32,
                                                                  frac=args.num_img,
                                                                  workers=args.workers)
    elif args.model == "CelebA":
        model = ManLearn.VAE_CELEBA().to(args.device)
        data_path = ''.join((args.path, 'CelebA/'))
        model_path = ''.join((args.path, 'CelebA/params.pt'))
        dataset, dataset_subset, trainloader = ManLearn.load_celeba(data_path,
                                                                  batch_size=args.batch_size,
                                                                  img_size=64,
                                                                  frac=args.num_img,
                                                                  workers=args.workers)
    elif args.model == "MNIST":
        model = ManLearn.VAE_MNIST().to(args.device)
        data_path = ''.join((args.path, 'MNIST/'))
        model_path = ''.join((args.path, 'MNIST/params.pt'))
        dataset, dataset_subset, trainloader = ManLearn.load_mnist(data_path,
                                                                  batch_size=args.batch_size,
                                                                  img_size=28,
                                                                  frac=args.num_img,
                                                                  workers=args.workers)
    elif args.model == "DTU":
        model = ManLearn.VAE_DTU().to(args.device)
        data_path = ''.join((args.path, 'DTU_orbit/orbit/'))
        model_path = ''.join((args.path, 'DTU/params.pt'))
        dataset, dataset_subset, trainloader = ManLearn.load_dtu(data_path,
                                                                  batch_size=args.batch_size,
                                                                  img_size=64,
                                                                  frac=args.num_img,
                                                                  workers=args.workers)
        
    N = len(trainloader.dataset)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.con_training:
        checkpoint = torch.load(model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        elbo = checkpoint['ELBO']
        rec_loss = checkpoint['rec_loss']
        kld_loss = checkpoint['KLD']

        train_loss_elbo = elbo
        train_loss_rec = rec_loss
        train_loss_kld = kld_loss
    else:
        last_epoch = 0

    model.train()
    for epoch in range(last_epoch, epochs):
        running_loss_elbo = 0.0
        running_loss_rec = 0.0
        running_loss_kld = 0.0
        for x in trainloader:
            #x = x.to(args.device) #If DATA is not saved to device
            dat = x[0].to(args.device)
            _, x_hat, mu, var, kld, rec_loss, elbo = model(dat)
            optimizer.zero_grad() #optimizer.zero_grad(set_to_none=True) #Based on performance tuning
            elbo.backward()
            optimizer.step()

            running_loss_elbo += elbo.item()
            running_loss_rec += rec_loss.item()
            running_loss_kld += kld.item()

            #del x, x_hat, mu, var, kld, rec_loss, elbo #In case you run out of memory

        train_epoch_loss = running_loss_elbo/N
        train_loss_elbo.append(train_epoch_loss)
        train_loss_rec.append(running_loss_rec/N)
        train_loss_kld.append(running_loss_kld/N)
        #print(f"Epoch {epoch+1}/{epochs} - loss: {train_epoch_loss:.4f}")


        if (epoch+1) % args.save_step == 0:
            torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ELBO': train_loss_elbo,
                'rec_loss': train_loss_rec,
                'KLD': train_loss_kld
                }, model_path)


    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ELBO': train_loss_elbo,
                'rec_loss': train_loss_rec,
                'KLD': train_loss_kld
                }, model_path)

    return

#%% Calling main

if __name__ == '__main__':
    main()