import torch
from .utils import load_checkpoint, CheckPoint
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, dataloader, model, model_name, head, optimizer, criterion, device, epochs):
        self.dataloader = dataloader
        self.model = model
        self.model_name = model_name
        self.head = head
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epoch = 0
        self.epochs = epochs
        
        self.checkpoint = CheckPoint(self.model, self.head, self.optimizer)
        
            
    def _load_checkpoint(self, file):
        self.epoch = load_checkpoint(file, self.model, self.head, self.optimizer)
        
        
    def train(self):
        self.model.train()
        scaler = GradScaler()
        train_losses = []
        
        for it in range(self.epoch, self.epochs + 1):
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {it}/{self.epochs}")
            for images, labels in pbar:
                
                self.optimizer.zero_grad()
                
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
            self.checkpoint.save(f'{self.model_name}_checkpoint_{it}.pth', it)
        
        
    