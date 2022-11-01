from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

def get_scheduler(config, optimizer):
    
    if config.SCHEDULER_NAME == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config.FACTOR, patience=config.PATIENCE, verbose=True)
    else:
        raise ValueError("Scheduler not supported")

    return scheduler