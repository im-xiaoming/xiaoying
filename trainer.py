import torch
from .utils import load_checkpoint
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, dataloader, model, head, optimizer, criterion, device, epochs):
        self.dataloader = dataloader
        self.model = model
        self.head = head
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epoch = 0
        self.epochs = epochs
        
            
    def _load_checkpoint(self, file):
        self.epoch = load_checkpoint(file, self.model, self.head, self.optimizer)
        
        
    def train(self):
        self.model.train()
        scaler = GradScaler()
        train_losses = []
        
        for it in range(self.epoch, self.epochs):
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {it+1}/{self.epochs}")
            for images, labels in pbar:
                
                self.optimizer.zero_grad()
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.autocast():
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
        
        
    