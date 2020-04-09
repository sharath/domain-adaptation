import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from common import save_model, save_data
from torchvision.datasets import MNIST, SVHN
from .utils import get_transform
from .models import (get_decoder, get_encoder, get_classifier)

def aae(experiment_name, args, log_file=sys.stdout):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    transform = get_transform(args.width, args.channels)
    source_train = SVHN(root='datasets/', transform=transform, split='train')
    source_validation = SVHN(root='datasets/', transform=transform, split='test')
    target_train = MNIST(root='datasets/', train=True, transform=transform)
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
    
    D, D_optim = get_decoder(
        embedding_dim=args.embedding_dim,
        filters=args.filters,
        nclasses=nclasses,
        lr=args.lr, betas=(args.beta1, args.beta2),
        device=args.device
    )
    
    # loss functions
    criterion_clf = nn.CrossEntropyLoss().to(args.device)
    criterion_adv = nn.BCELoss().to(args.device)
    criterion_rec = nn.MSELoss().to(args.device)
    
    # targets for the GAN
    real_label_val = 1
    fake_label_val = 0
    real_labels = torch.FloatTensor(args.batch_size).fill_(real_label_val).to(args.device)
    fake_labels = torch.FloatTensor(args.batch_size).fill_(fake_label_val).to(args.device)
    
    losses = {'D': [], 'E': [], 'C': []}
    accuracy = {'train': [], 'test': []}

    # prepare dataloaders
    source_train_loader = DataLoader(source_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    source_validation_loader = DataLoader(source_validation, batch_size=args.batch_size, shuffle=True, drop_last=False)
    target_train_loader = DataLoader(target_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    target_test_loader = DataLoader(target_test, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    for epoch in range(args.epochs):
        E.train()
        C.train()
        print('Iter \t E_loss \t C_loss \t D_loss', flush=True, file=log_file)
        for it, (source_batch, target_batch) in enumerate(zip(source_train_loader, target_train_loader)):
            '''
            Pre-processing
            '''
            source_images, source_labels = map(lambda v: v.to(args.device), source_batch)   
            target_images, target_labels = map(lambda v: v.to(args.device), target_batch)
            
            # compute one-hot vectors
            source_labels_oh = nn.functional.one_hot(source_labels, nclasses).float().to(args.device)
            #target_labels_oh = nn.functional.one_hot(nclasses*torch.ones_like(source_labels), nclasses).float().to(args.device)

            # generate fake embeddings
            prior_embedding = torch.FloatTensor(args.batch_size, args.embedding_dim).normal_(0, 1).to(args.device)
            
            # compute emeddings from source images
            source_embedding = E(source_images).view(args.batch_size, -1)
            source_embedding_cat = torch.cat((source_labels_oh, source_embedding), 1)
            source_embedding_cat = source_embedding_cat.view(args.batch_size, -1, 1, 1)
            
            # compute emeddings from target images
            target_embedding = E(target_images).view(args.batch_size, -1)
            #target_embedding_cat = torch.cat((target_labels_oh, target_embedding), 1)
            #target_embedding_cat = target_embedding_cat.view(args.batch_size, -1, 1, 1)
        
            '''
            Update C
            '''
            D.zero_grad()
    
            # compute classifier losses using source embeddings
            source_adv, source_clf = C(source_embedding)
            source_C_adv_loss = criterion_adv(source_adv, fake_labels)
            source_C_clf_loss = criterion_clf(source_clf, source_labels)

            # compute classifier losses using target embeddings
            target_adv, target_clf = C(target_embedding)
            target_C_adv_loss = criterion_adv(target_adv, fake_labels)
            target_C_clf_loss = criterion_clf(target_clf, target_labels)
            
            # compute classifier losses on prior embeddings
            prior_adv, prior_clf = C(prior_embedding)
            prior_C_adv_loss = criterion_adv(prior_adv, real_labels)
            
            # perform classifier optimization step
            C_loss = source_C_adv_loss + source_C_clf_loss + target_C_adv_loss + prior_C_adv_loss
            C_loss.backward(retain_graph=True)
            C_optim.step()
        
            '''
            Update D
            '''
            D.zero_grad()
            
            # compute decoder losses
            source_dec = D(source_embedding_cat)
            source_D_rec_loss = criterion_rec(source_dec, source_images)
            
            # perform D optimization step
            D_loss = source_D_rec_loss
            D_loss.backward(retain_graph=True)
            D_optim.step()

            '''
            Update E
            '''
            E.zero_grad()
            
            # compute encoder loss from classifier
            source_adv, source_clf = C(source_embedding)
            source_E_adv_loss = criterion_adv(source_adv, real_labels)
            source_E_clf_loss = criterion_clf(source_clf, source_labels)
            
            target_adv, target_clf = C(target_embedding)
            target_E_adv_loss = criterion_adv(target_adv, real_labels)
            
            # perform encoder optimization step
            E_loss = args.dec_weight * D_loss + (source_E_adv_loss + target_E_adv_loss + source_E_clf_loss)
            E_loss.backward(retain_graph=True)
            E_optim.step()
            
            '''
            Bookkeeping
            '''
            losses['D'].append(D_loss.item())
            losses['E'].append(E_loss.item())
            losses['C'].append(C_loss.item())
            
            if it % 10 == 0 and it != 0:
                D_avg = np.mean(losses['D'][-10:])
                E_avg = np.mean(losses['E'][-10:])
                C_avg = np.mean(losses['C'][-10:])
                print(f'{it:3d}\t{E_avg:3f}\t{C_avg:3f}\t{D_avg:3f}', flush=True, file=log_file)
        
        E.eval()
        C.eval()
        correct, total = [], []
        for it, (images, labels) in enumerate(source_validation_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            embeddings = E(images).squeeze()
            predictions = C(embeddings)
            correct.append(torch.sum(torch.argmax(predictions, 1) == labels).item())
            total.append(len(images))
        accuracy['train'].append(sum(correct)/sum(total))
        
        correct, total = [], []
        for it, (images, labels) in enumerate(target_test_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            embeddings = E(images).squeeze()
            predictions = C(embeddings)
            correct.append(torch.sum(torch.argmax(predictions, 1) == labels).item())
            total.append(len(images))
        accuracy['test'].append(sum(correct)/sum(total))
        
        print(f'Epoch: {epoch+1:3d} \t Train: {accuracy["train"][-1]:2.3f} \t Test: {accuracy["test"][-1]:2.3f}', flush=True, file=log_file)
        print(flush=True, file=log_file)
        
    save_model(E, 'E', experiment_name)
    save_model(C, 'C', experiment_name)
    save_model(D, 'D', experiment_name)
    save_data({
        'losses': losses,
        'accuracy': accuracy
    }, experiment_name)