import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil

def load_pretrained(config, model):
    print(f"==============> start form {config.RESUME} ....................")

    pre_model = torch.load(config.RESUME, map_location='cpu')
    sd = model.state_dict()
    for key, value in pre_model.items():
        if key in sd.keys():  
            sd[key] = value
    model.load_state_dict(sd)
    print("Pretrained model loaded successfully!")


def save_checkpoint(config, model, optimizer, auc, scheduler=None, scaler=None, epoch=0, is_best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'auc': auc,
    }
    filename = os.path.join(config.OUTPUT, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(config.OUTPUT, 'model_best.pth'))

def load_checkpoint(config, model, optimizer, scheduler=None, scaler=None):
    print(f"==============> start form {config.RESUME} ....................")
    checkpoint = torch.load(config.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    start_epoch = checkpoint['epoch'] + 1
    auc = checkpoint['auc']
    print(f"==============> resume from epoch {start_epoch} ....................")
    return start_epoch, auc

def adjust_learning_rate(config, optimizer, loader, step):
    max_steps = config.TRAIN.EPOCHS * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = config.BATCH_SIZE / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * config.LAMBDA
    optimizer.param_groups[1]['lr'] = lr * config.LR_BIASES

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def bt_loss(config, z_a, z_b):
    # Number of batch size
    N, D = z_a.shape
    bn = nn.BatchNorm1d(D, affine=False, device=config.DEVICE)
    # empirical cross-correlation matrix
    c = bn(z_a).T @ bn(z_b)
    c = c.to(config.DEVICE)
    # sum the cross-correlation matrix between all gpus
    c.div_(config.BATCH_SIZE)
    # torch.distributed.all_reduce(c)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + config.LAMBDA * off_diag
    return loss

def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'