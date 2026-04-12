import torch
from .utils import load_checkpoint, CheckPoint
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import shutil
import os

class Trainer:
    def __init__(self, dataloader, model, model_name, head, optimizer, criterion, device, epochs, gradient_accumulation_step=0):
        self.dataloader = dataloader
        self.model = model
        self.model_name = model_name
        self.head = head
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epoch = 0
        self.epochs = epochs
        self.gradient_accumulation_step = gradient_accumulation_step
        
        self.checkpoint = CheckPoint(self.model, self.head, self.optimizer)
        
            
    def _load_checkpoint(self, file):
        self.epoch = load_checkpoint(file, self.model, self.head, self.optimizer)
        
        
    def train(self, save_dir=None):
        self.model.train()
        scaler = GradScaler()
        train_losses = []
        count = 0
        
        for it in range(self.epoch, self.epochs + 1):
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {it}/{self.epochs}")
            for images, labels in pbar:
                
                count += 1
                if count >= self.gradient_accumulation_step:
                    self.optimizer.zero_grad()
                    count = 0
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    embedings, norm = self.model(images)
                    cos_theta = self.head(embedings, norm, labels)
                    loss = self.criterion(cos_theta, labels)
                    
                train_losses.append(loss.item())
                    
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                # update tqdm
                pbar.set_postfix({
                    "loss": f"{np.mean(train_losses):.4f}",
                    "lr": self.optimizer.param_groups[0]["lr"]
                })
                
            # save checkpoint
            file = self.checkpoint.save(f'{self.model_name}_checkpoint_{it}.pth', it)
            if save_dir:
                shutil.move(file, os.path.join(save_dir))
        
        
    