
import time
import torch
from utils import AverageMeter, ProgressMeter
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm 

def validate(config, model, loader, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    acc = AverageMeter("Acc", ":6.2f")
    auc = AverageMeter("AUC", ":6.2f")
    
    progress = ProgressMeter(
        len(loader),
        [losses, batch_time, acc, auc],
        prefix="Test:",
    )

    # switch to evaluate mode
    model.eval()
    y = []
    y_pred = []
    aucs = []
    end = time.time()
    loader = tqdm(loader, total=len(loader), desc="Validating", position=0, leave=True)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            # Move data to GPU config.DEVICE
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            for i in range(len(images)):
                labels[i] = labels[i].to(config.DEVICE)
            labels = labels.to(torch.long)
    
            # compute output
            predictions, _, _ = model(images)
            loss = criterion(predictions[0], labels[:, 0])
            for j in range(1, labels.shape[1]):
                loss += criterion(predictions[j], labels[:, j])
        
            losses.update(loss.item(), images.size(0))

            for i  in range(len(predictions)):
                y.append(labels[i].cpu().numpy())
                y_pred.append(torch.sigmoid(predictions[i]).detach().cpu().numpy())
            
            y_pred = np.array(y_pred)
            y = np.array(y)

            for i in range(config.NUM_CLASSES):
                try:
                    auc.update(roc_auc_score(y[:, i], y_pred[:, i], multi_class='ovr'), images.size(0))
                    aucs.append(roc_auc_score(y[:,i], y_pred[:,i], multi_class='ovr'))
                except ValueError:
                    aucs.append(0)

            auc_list = np.mean(aucs)
            auc_avg =  auc.avg 
            print(auc_list)
            print(auc_avg)

            batch_time.update(time.time() - end)
            # measure accuracy and record loss
            end = time.time()
        
            if config.PRINT_FREQ > 0 and idx % config.PRINT_FREQ == 0:
                progress.display(idx)

            

            