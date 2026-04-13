import torch
from .utils import load_checkpoint, CheckPoint
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import shutil
import os


class TrainingArgs:
    def __init__(self, model_name, device, epochs, gradient_accumulation_step=0, eval_per_epoch=1):
        self.model_name = model_name
        self.device = device
        self.epochs = epochs
        self.gradient_accumulation_step = gradient_accumulation_step
        self.eval_per_epoch = eval_per_epoch


class Trainer:
    def __init__(self, args, dataloader, model, head, optimizer, criterion):
        
        self.dataloader = dataloader
        self.model = model
        self.head = head
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = 0
        
        self.device = args.device
        self.model_name = args.model_name
        self.epochs = args.epochs
        self.gradient_accumulation_step = args.gradient_accumulation_step
        self.eval_per_epoch = args.eval_per_epoch
        
        self.checkpoint = CheckPoint(self.model, self.head, self.optimizer)
        
            
    def _load_checkpoint(self, file):
        self.epoch = load_checkpoint(file, self.model, self.head, self.optimizer)
        
        
    def train(self, save_dir=None, eval_loader=None, metrics=[{}]):
        """
        Metrics receives list dict includes: metric_name: str, metric: function
        """
        
        self.model.train()
        
        scaler = GradScaler()
        train_losses = []
        
        grad_count = 0
        epoch_count = 0
        
        for it in range(self.epoch, self.epochs + 1):
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {it}/{self.epochs}")
            for images, labels in pbar:
                
                grad_count += 1
                if grad_count >= self.gradient_accumulation_step:
                    self.optimizer.zero_grad()
                    grad_count = 0
                
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
                    "loss": f"{train_losses[-1]:.4f}",
                    "mean loss": f"{np.mean(train_losses):.4f}",
                    "lr": self.optimizer.param_groups[0]["lr"]
                })
                
            # save checkpoint
            file = self.checkpoint.save(f'{self.model_name}_checkpoint_{it}.pth', it)
            if save_dir:
                if os.path.exists(os.path.join(save_dir, file)):
                    os.remove(os.path.join(save_dir, file))
                shutil.move(file, os.path.join(save_dir))
            
            # evaluate
            epoch_count += 1
            if epoch_count >= self.eval_per_epoch:
                if len(metrics) > 0: 
                    self._eval(eval_loader, metrics)
                epoch_count = 0
                
    
    def _eval(self, eval_loader, metrics=[{}]):
        self.model.eval()
        
        assert eval_loader != None
        
        for metric in metrics:
            if metric.get('metric_name').lower() == 'accuracy':
                metric.get('metric')(self.model, eval_loader, self.device)



from .validation import evaluate

def get_metrics(metric_type='accuracy'):
    if metric_type.lower() == 'accuracy':
        return evaluate.evaluate1
    else:
        raise ValueError("metrics_name is invalid.")
    