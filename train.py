import wandb
import torch
from .utils import *
from torch.cuda.amp import autocast
import time
from tqdm import tqdm

def train(config, train_loader, model, criterion,optimizer, scaler, epoch):
    """Train for one epoch on the training set"""
    # Tracking metrics
    
    kl_loss = nn.KLDivLoss(reduction='batchmean').to(config.DEVICE)

    loss_meter = AverageMeter("Loss", ":.6f")
    pred_loss_meter = AverageMeter("Pred Loss", ":.6f")
    boost_loss_meter = AverageMeter("Boost Loss", ":.6f")
    time_meter = AverageMeter("Time", ":6.3f")
    progress = ProgressMeter(
        len(train_loader),
        [loss_meter, time_meter],
        prefix="Epoch: [{}]".format(config.START_EPOCH),
    )   
    start= epoch * len(train_loader)
    model.train()
    end = time.time()
    train_loader = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{config.TRAIN.EPOCHS}")
    for idx, (images, labels) in enumerate(train_loader):
        step = start + idx
        # Move data to GPU device
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        for i in range(len(images)):
            labels[i] = labels[i].to(config.DEVICE)
        labels = labels.to(torch.long)

        # adjust_learning_rate(config, optimizer, train_loader, step)
        
        with autocast(enabled=config.AMP_ENABLE):
            predictions, global_embed, local_embed = model(images)
            pred_loss = criterion(predictions[0], labels[:, 0])
            for j in range(1, labels.shape[1]):
                pred_loss += criterion(predictions[j], labels[:, j])
            boost_loss = kl_loss(global_embed, local_embed)
            
            loss = pred_loss + config.LAMBDA * boost_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ### TODO: update the loss meter and time meter
        loss_meter.update(loss.item(), images.size(0))
        pred_loss_meter.update(pred_loss.item(), images.size(0))
        boost_loss_meter.update(boost_loss.item(), images.size(0))
        time_meter.update(time.time() - end)

        if step % config.PRINT_FREQ == 0:
            progress.display(idx)
            stats = {
                "loss": loss_meter.avg,
                "pred_loss": pred_loss_meter.avg,
                "boost_loss": boost_loss_meter.avg,
                "lr": optimizer.param_groups[0]["lr"],
                "step": step,
                "epoch": epoch,
            }
            wandb.log(stats)

