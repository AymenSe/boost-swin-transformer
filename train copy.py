import wandb
import torch
from data.build_dataset import get_nih, get_chexpert
import os
from data.dataset import NIH_Dataset, CheX_Dataset
from model.build_swin_vit import build_swin_vit
from .utils import *
from torch.cuda.amp import GradScaler, autocast
import time


def train(params):
    with wandb.init(project="swin-twins", entity="asekhri", job_type="train", config=params) as run:
        config = wandb.config
        os.makedirs(config.OUTPUT , exist_ok=True)

        if config.DATASET == "NIH":
            train_loader, valid_loader, test_loader = get_nih(config, NIH_Dataset)
        elif config.DATASET == "CheX":
            train_loader, valid_loader, test_loader = get_chexpert(config, CheX_Dataset)
        else:
            raise ValueError("Dataset not supported")

        
        model = build_swin_vit(config)
        model = model.to(config.DEVICE)
        if config.PRETRAINED:
            load_pretrained(config, model)
        model.train()
        
        scaler = GradScaler(enabled=config.AMP_ENABLE)
        
        start_time = time.time()
        artifact = wandb.Artifact("proposed-method", type="model", description="boost swin T with SSL", metadata=dict(config))
        
        for epoch in range(config.EPOCHS):              
            for step, ((x1, x2), _) in enumerate(train_loader, start= epoch * len(train_loader)):
                # Move data to GPU device
                x1 = x1.to(config.DEVICE)
                x2 = x2.to(config.DEVICE)

                adjust_learning_rate(config, optimizer, train_loader, step)
                with autocast(enabled=True):
                    em1, em2 = model(x1, x2)
                    # loss1 = bt_loss(config, em1[0], em2[0])
                    # loss2 = bt_loss(config, em1[1], em2[1])
                    # loss3 = bt_loss(config, em1[2], em2[2])
                    loss = bt_loss(config, em1[3], em2[3])
                    
                    # loss = loss4 + 0.0 * loss3 + 0.0 * loss2 + 0.0 * loss1

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # if step % config.PRINT_FREQ == 0:
                stats = dict(epoch=epoch, step=step,
                            lr_weights=optimizer.param_groups[0]['lr'],
                            lr_biases=optimizer.param_groups[1]['lr'],
                            loss=loss.item(),
                            # embd1_loss=loss1.item(),
                            # embd2_loss=loss2.item(),
                            # embd3_loss=loss3.item(),
                            # embd4_loss=loss4.item(),
                            time=int(time.time() - start_time))
                wandb.log(stats)
                
            state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
            torch.save(state, config.OUTPUT + '/checkpoint.pth')
            artifact.add_file(os.path.join(config.OUTPUT, 'checkpoint.pth'), name="ckp.pth")

            # if epoch == 2:
            #     break

        # save final model
        torch.save(model.backbone.state_dict(), config.OUTPUT + '/backbone.pth')
        artifact.add_file(os.path.join(config.OUTPUT, 'backbone.pth'))

        run.log_artifact(artifact)
    wandb.finish()