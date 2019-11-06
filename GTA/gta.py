import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from common import save_model, save_data
from torchvision.datasets import MNIST, SVHN
from .utils import get_transform
from .models import (
    get_generator, get_discriminator,
    get_encoder, get_classifier
)

def gta(experiment_name, args):
    transform = get_transform(args.width, args.channels)
    source_train = SVHN(root='datasets/', transform=transform, split='train')
    source_validation = SVHN(root='datasets/', transform=transform, split='test')
    target_train = MNIST(root='datasets/', train=True, transform=transform)
    
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
    
    G, G_optim = get_generator(
        latent_dim=args.latent_dim,
        embedding_dim=args.embedding_dim,
        filters=args.filters, nclasses=nclasses,
        lr=args.lr, betas=(args.beta1, args.beta2),
        device=args.device
    )
    
    D, D_optim = get_discriminator(
        filters=args.filters,
        nclasses=nclasses,
        lr=args.lr, betas=(args.beta1, args.beta2),
        device=args.device
    )
    
    # loss functions
    criterion_clf = nn.CrossEntropyLoss().to(args.device)
    criterion_dis = nn.BCELoss().to(args.device)
    
    # targets for the GAN
    real_label_val = 1
    fake_label_val = 0
    real_labels = torch.FloatTensor(args.batch_size).fill_(real_label_val).to(args.device)
    fake_labels = torch.FloatTensor(args.batch_size).fill_(fake_label_val).to(args.device)
    
    losses = {'D': [], 'G': [], 'F': [], 'C': []}
    accuracy = []

    # prepare dataloaders
    source_train_loader = DataLoader(source_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    source_validation_loader = DataLoader(source_validation, batch_size=args.batch_size, shuffle=True, drop_last=False)
    target_train_loader = DataLoader(target_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    for epoch in range(args.epochs):
        F.train()
        C.train()
        print('Iter \t F_loss \t C_loss \t G_loss \t D_loss')
        for it, (source_batch, target_batch) in enumerate(zip(source_train_loader, target_train_loader)):
            '''
            Pre-processing
            '''
            source_images, source_labels = map(lambda v: v.to(args.device), source_batch)   
            target_images, target_labels = map(lambda v: v.to(args.device), target_batch)
            
            # compute one-hot vectors
            source_labels_oh = nn.functional.one_hot(source_labels, nclasses+1).float().to(args.device)
            target_labels_oh = source_labels_oh.scatter(1, torch.LongTensor(args.batch_size, 1).fill_(nclasses).to(args.device), 1)
            
            # sample latent noise batch
            latent_batch = torch.FloatTensor(args.batch_size, args.latent_dim, 1, 1).normal_(0, 1).to(args.device)
            
            # compute emeddings from source images
            source_embedding = F(source_images).view(args.batch_size, -1)
            source_embedding_cat = torch.cat((source_labels_oh, source_embedding), 1)
            source_embedding_cat = source_embedding_cat.view(args.batch_size, -1, 1, 1)
            source_embedding_cat = torch.cat((source_embedding_cat, latent_batch), 1)
            
            # compute emeddings from target images
            target_embedding = F(target_images).view(args.batch_size, -1)
            target_embedding_cat = torch.cat((target_labels_oh, target_embedding), 1)
            target_embedding_cat = target_embedding_cat.view(args.batch_size, -1, 1, 1)
            target_embedding_cat = torch.cat((target_embedding_cat, latent_batch), 1)
            
            # generate samples from concatentated embeddings
            source_generated_samples = G(source_embedding_cat)
            target_generated_samples = G(target_embedding_cat)
        
            '''
            Update D
            '''
            D.zero_grad()
    
            # compute discriminator losses on real source images
            source_real_dis, source_real_clf = D(source_images)
            source_D_dis_loss_real = criterion_dis(source_real_dis, real_labels)
            source_D_clf_loss_real = criterion_clf(source_real_clf, source_labels)
            
            # compute discriminator losses on fake source images
            source_fake_dis, source_fake_clf = D(source_generated_samples)
            source_D_dis_loss_fake = criterion_dis(source_fake_dis, fake_labels)
            
            # compute discriminator losses on fake target images
            target_fake_dis, target_fake_clf = D(target_generated_samples)
            target_D_d_loss_fake = criterion_dis(target_fake_dis, fake_labels)
            
            # perform D optimization step
            D_loss = source_D_dis_loss_real + source_D_clf_loss_real + source_D_dis_loss_fake + target_D_d_loss_fake
            D_loss.backward(retain_graph=True)
            D_optim.step()
        
            '''
            Update G
            '''
            G.zero_grad()
            
            # compute generator losses
            source_fake_dis, source_fake_clf = D(source_generated_samples)
            source_D_dis_loss_fake = criterion_dis(source_fake_dis, real_labels)
            source_D_clf_loss_fake = criterion_clf(source_fake_clf, source_labels)
            
            # perform G optimization step
            G_loss = source_D_dis_loss_fake + source_D_clf_loss_fake
            G_loss.backward(retain_graph=True)
            G_optim.step()
        
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
            
            # re-generate samples using updated generator
            source_generated_samples = G(source_embedding_cat)
            target_generated_samples = G(target_embedding_cat)
    
            # compute encoder loss from GAN
            source_fake_dis, source_fake_clf = D(source_generated_samples)
            source_D_clf_loss_fake = criterion_clf(source_fake_clf, source_labels)
            
            target_fake_dis, target_fake_clf = D(target_generated_samples)
            target_D_dis_loss_fake = criterion_dis(target_fake_dis, real_labels)
            
            # perform F optimization step
            F_loss = C_loss + args.adv_weight * (source_D_clf_loss_fake + args.alpha * target_D_dis_loss_fake)
            F_loss.backward(retain_graph=True)
            F_optim.step()
            
            '''
            Bookkeeping
            '''
            losses['D'].append(D_loss.item())
            losses['G'].append(G_loss.item())
            losses['F'].append(F_loss.item())
            losses['C'].append(C_loss.item())
            
            if it % 10 == 0 and it != 0:
                D_avg = np.mean(losses['D'][-10:])
                G_avg = np.mean(losses['G'][-10:])
                F_avg = np.mean(losses['F'][-10:])
                C_avg = np.mean(losses['C'][-10:])
                print(f'{it:3d}\t{F_avg:3f}\t{C_avg:3f}\t{D_avg:3f}\t{G_avg:3f}')
        
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
        print(f'Epoch: {epoch+1:3d} \t Accuracy: {accuracy[-1]:2.3f}')
        print()
        
    save_model(F, 'F', experiment_name)
    save_model(C, 'C', experiment_name)
    save_model(G, 'G', experiment_name)
    save_model(D, 'D', experiment_name)
    save_data({
        'losses': losses,
        'accuracy': accuracy
    }, experiment_name)