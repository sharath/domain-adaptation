import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from common import save_model, save_data
from torchvision.datasets import SVHN, MNIST
from .utils import get_transform
from .models import (
    get_encoder, get_classifier
)


def baseline(experiment_name, args, log_file=sys.stdout):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    transform = get_transform(args.width, args.channels)
    source_train = SVHN(root='datasets/', transform=transform, split='train')
    source_validation = SVHN(root='datasets/', transform=transform, split='test')
    target_test = MNIST(root='datasets/', train=False, transform=transform)
    
    nclasses = len(set(source_train.labels))
    
    E, E_optim = get_encoder(
        channels=args.channels, 
        filters=args.filters,
        embedding_dim=args.embedding_dim,
        lr=args.lr, betas=(args.beta1, args.beta2),
        device=args.device
    )
    
    C, C_optim = get_classifier(
        embedding_dim=args.embedding_dim,
        filters=args.filters,
        nclasses=nclasses,
        lr=args.lr, betas=(args.beta1, args.beta2),
        device=args.device
    )
    
    # loss function
    criterion_clf = nn.CrossEntropyLoss().to(args.device)
    losses = {'E': [], 'C': []}
    accuracy = {'train': [], 'test': []}

    # prepare dataloaders
    source_train_loader = DataLoader(source_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    source_validation_loader = DataLoader(source_validation, batch_size=args.batch_size, shuffle=True, drop_last=False)
    target_test_loader = DataLoader(target_test, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    for epoch in range(args.epochs):
        E.train()
        C.train()
        print('Iter \t E_loss \t C_loss', flush=True, file=log_file)
        for it, source_batch in enumerate(source_train_loader):
            '''
            Pre-processing
            '''
            source_images, source_labels = map(lambda v: v.to(args.device), source_batch)
            
            # compute emedding from source images
            source_embedding = E(source_images).view(args.batch_size, -1)
            
            '''
            Update C
            '''
            C.zero_grad()
            
            # compute classifier losses
            _, source_clf = C(source_embedding)
            C_loss = criterion_clf(source_clf, source_labels)
            
            # perform classifier optimization step
            C_loss.backward(retain_graph=True)
            C_optim.step()
            
            '''
            Update E
            '''
            E.zero_grad()
            
            # compute encoder loss from updated classifier
            _, source_clf = C(source_embedding)
            C_loss = criterion_clf(source_clf, source_labels)
            
            # perform encoder optimization step
            E_loss = C_loss
            E_loss.backward(retain_graph=True)
            E_optim.step()
            
            '''
            Bookkeeping
            '''
            losses['E'].append(E_loss.item())
            losses['C'].append(C_loss.item())
            
            if it % 10 == 0 and it != 0:
                E_avg = np.mean(losses['E'][-10:])
                C_avg = np.mean(losses['C'][-10:])
                print(f'{it:3d}\t{E_avg:3f}\t{C_avg:3f}', flush=True, file=log_file)
        
        E.eval()
        C.eval()
        correct, total = [], []
        for it, (images, labels) in enumerate(source_validation_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            embeddings = E(images).squeeze()
            _, predictions = C(embeddings)
            correct.append(torch.sum(torch.argmax(predictions, 1) == labels).item())
            total.append(len(images))
        accuracy['train'].append(sum(correct)/sum(total))
        
        correct, total = [], []
        for it, (images, labels) in enumerate(target_test_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            embeddings = E(images).squeeze()
            _, predictions = C(embeddings)
            correct.append(torch.sum(torch.argmax(predictions, 1) == labels).item())
            total.append(len(images))
        accuracy['test'].append(sum(correct)/sum(total))
        
        print(f'Epoch: {epoch+1:3d} \t Train: {accuracy["train"][-1]:2.3f} \t Test: {accuracy["test"][-1]:2.3f}', flush=True, file=log_file)
        print(flush=True, file=log_file)
        
    save_model(E, 'E', experiment_name)
    save_model(C, 'C', experiment_name)
    save_data({
        'losses': losses,
        'accuracy': accuracy
    }, experiment_name)