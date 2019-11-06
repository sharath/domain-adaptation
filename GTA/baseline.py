import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from common import save_model, save_data
from torchvision.datasets import SVHN
from .utils import get_transform
from .models import (
    get_encoder, get_classifier
)

def baseline(experiment_name, args, log_file=sys.stdout):
    transform = get_transform(args.width, args.channels)
    source_train = SVHN(root='datasets/', transform=transform, split='train')
    source_validation = SVHN(root='datasets/', transform=transform, split='test')
    
    nclasses = len(set(source_train.labels))
    
    F, F_optim = get_encoder(
        channels=args.channels, 
        filters=args.filters,
        embedding_dim=args.embedding_dim,
        lr=args.lr, betas=(args.beta1, args.beta2),
        device=args.device
    )
    
    C, C_optim = get_classifier(
        filters=args.filters,
        nclasses=nclasses,
        lr=args.lr, betas=(args.beta1, args.beta2),
        device=args.device
    )
    
    # loss function
    criterion_clf = nn.CrossEntropyLoss().to(args.device)
    losses = {'F': [], 'C': []}
    accuracy = []

    # prepare dataloaders
    source_train_loader = DataLoader(source_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    source_validation_loader = DataLoader(source_validation, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    for epoch in range(args.epochs):
        F.train()
        C.train()
        print('Iter \t F_loss \t C_loss', flush=True, file=log_file)
        for it, source_batch in enumerate(source_train_loader):
            '''
            Pre-processing
            '''
            source_images, source_labels = map(lambda v: v.to(args.device), source_batch)
            
            # compute one-hot vector
            source_labels_oh = nn.functional.one_hot(source_labels, nclasses+1).float().to(args.device)
            
            # compute emedding from source images
            source_embedding = F(source_images).view(args.batch_size, -1)
            
            '''
            Update C
            '''
            C.zero_grad()
            
            # compute classifier losses
            source_clf = C(source_embedding)
            C_loss = criterion_clf(source_clf, source_labels)
            
            # perform G optimization step
            C_loss.backward(retain_graph=True)
            C_optim.step()
            
            '''
            Update F
            '''
            F.zero_grad()
            
            # compute encoder loss from updated classifier
            source_clf = C(source_embedding)
            C_loss = criterion_clf(source_clf, source_labels)
            
            # perform F optimization step
            F_loss = C_loss
            F_loss.backward(retain_graph=True)
            F_optim.step()
            
            '''
            Bookkeeping
            '''
            losses['F'].append(F_loss.item())
            losses['C'].append(C_loss.item())
            
            if it % 10 == 0 and it != 0:
                F_avg = np.mean(losses['F'][-10:])
                C_avg = np.mean(losses['C'][-10:])
                print(f'{it:3d}\t{F_avg:3f}\t{C_avg:3f}', flush=True, file=log_file)
        
        F.eval()
        C.eval()
        correct, total = [], []
        for it, (images, labels) in enumerate(source_validation_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            embeddings = F(images).squeeze()
            predictions = C(embeddings)
            correct.append(torch.sum(torch.argmax(predictions, 1) == labels).item())
            total.append(len(images))
        accuracy.append(sum(correct)/sum(total))
        print(f'Epoch: {epoch+1:3d} \t Accuracy: {accuracy[-1]:2.3f}', flush=True, file=log_file)
        print(flush=True, file=log_file)
        
    save_model(F, 'F', experiment_name)
    save_model(C, 'C', experiment_name)
    save_data({
        'losses': losses,
        'accuracy': accuracy
    }, experiment_name)