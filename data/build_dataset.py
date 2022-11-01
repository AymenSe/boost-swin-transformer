import torch
import torchvision.transforms as transforms
from torch.utils import DataLoader, SubsetRandomSampler
import numpy as np


def get_nih(config, NIH_Dataset, gender=None):
    
    train_transforms = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Resize(config.IMG_SIZE),
        # transforms.RandomAffine(45, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomAffine(45, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

#     val_transforms = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         transforms.Resize(config.IMG_SIZE),
#         transforms.RandomAffine(45, translate=(0.15, 0.15), scale=(0.85, 1.15)),
#         transforms.RandomHorizontalFlip(),   
        
#     ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(config.IMG_SIZE),
    ])

    train_dataset = NIH_Dataset(imgpath="images",
                            views=["PA","AP"],
                            split="train",
                            unique_patients=False,
                            gender=gender,
                            transform=train_transforms)
    
    test_dataset = NIH_Dataset(imgpath="images",
                            views=["PA","AP"],
                            split="test",
                            gender=gender,
                            unique_patients=False,
                            transform=test_transforms)



    if config.VAL_SPLIT is not None:
        dataset_size = len(train_dataset)
        dataset_indices = list(range(dataset_size))
        np.random.shuffle(dataset_indices)
        split_index = int(config.VAL_SPLIT * dataset_size)
        valid_indices = dataset_indices[:split_index]
        train_indices = dataset_indices[split_index:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        
    
#     val_ds_indices = list(range(len(valid_dataset)))
#     np.random.shuffle(val_ds_indices)
#     val_indices = val_ds_indices[:200]
#     valid_sampler = SubsetRandomSampler(val_indices)
    
#     test_ds_indices = list(range(len(test_dataset)))
#     np.random.shuffle(test_ds_indices)
#     test_indices = test_ds_indices[:200]
#     test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=config.BATCH_SIZE, 
                            shuffle=False if config.VAL_SPLIT else True,
                            sampler = train_sampler if config.VAL_SPLIT else None, 
                            num_workers=config.NUM_WORKERS)
    

    if config.VAL_SPLIT is not None:
        valid_loader = DataLoader(train_dataset, 
                                  batch_size=config.BATCH_SIZE, 
                                  shuffle=False, 
                                  sampler=valid_sampler,
                                  num_workers=config.NUM_WORKERS)
    else:
        valid_loader = None
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False) # , sampler=test_sampler)

    return train_loader, valid_loader, test_loader 


def get_chexpert(config, CheX_Dataset):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomAffine(45, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = CheX_Dataset(imgpath="C:\\Users\\marouane.tliba\\MedicalImaging\\swinCXR\\CheXpert-v1.0-small",
                                 csvpath="train.csv",
                                 views="Frontal",
                                 uncertain='zeros',
                                 transform=train_transforms)
    
    test_dataset = CheX_Dataset(imgpath="C:\\Users\\marouane.tliba\\MedicalImaging\\swinCXR\\CheXpert-v1.0-small",
                                 csvpath="valid.csv",
                                 views="Frontal",
                                 uncertain='zeros',
                                 transform=test_transforms)
    
    
    if config.VAL_SPLIT is not None:
        dataset_size = len(train_dataset)
        dataset_indices = list(range(dataset_size))
        np.random.shuffle(dataset_indices)
        split_index = int(config.VAL_SPLIT * dataset_size)
        valid_indices = dataset_indices[:split_index]
        train_indices = dataset_indices[split_index:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
    
    
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=config.BATCH_SIZE, 
                            shuffle=False if config.VAL_SPLIT else True,
                            sampler = train_sampler if config.VAL_SPLIT else None, 
                            num_workers=config.NUM_WORKERS)
    

    if config.VAL_SPLIT is not None:
        valid_loader = DataLoader(train_dataset, 
                                  batch_size=config.BATCH_SIZE, 
                                  shuffle=False, 
                                  sampler=valid_sampler,
                                  num_workers=config.NUM_WORKERS)
    else:
        valid_loader = None
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, valid_loader, test_loader