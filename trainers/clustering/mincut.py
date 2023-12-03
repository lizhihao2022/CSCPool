import os

import torch
from torch.nn import functional as F

import wandb
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score, completeness_score

from ..base import BaseTrainer
from utils import LossRecord
from models import MinCutGCN


class MinCutTrainer(BaseTrainer):
    """
    Trainer class for Graph Neural Networks.

    Args:
    - args (dict): dictionary containing the following keys:
        - model_name (str): name of the model
        - device (str): device to run the model on
        - epochs (int): number of epochs to train the model for
        - eval_freq (int): frequency of evaluation during training
        - patience (int): number of epochs to wait before early stopping
        - verbose (bool): whether to print progress updates during training
        - wandb_log (bool): whether to log training progress to Weights & Biases
        - logger (function): function to use for logging progress updates
        - saving_best (bool): whether to save the best model during training
        - saving_checkpoint (bool): whether to save model checkpoints during training
        - saving_path (str): path to save model checkpoints and best model
    """
    def __init__(self, args):
        super().__init__(model_name=args['model_name'], device=args['device'], epochs=args['epochs'], 
                         eval_freq=args['eval_freq'], patience=args['patience'], verbose=args['verbose'], 
                         wandb_log=args['wandb'], logger=args['log'], saving_best=args['saving_best'], 
                         saving_checkpoint=args['saving_checkpoint'], saving_path=args['saving_path'])

    def build_model(self, args):
        model = MinCutGCN(
            node_dim=args['node_dim'],
            edge_dim=args['edge_dim'],
            node_hidden_dim=args['node_hidden_dim'],
            edge_hidden_dim=args['edge_hidden_dim'],
            output_dim=args['output_dim'],
            num_layers=args['num_layers'],
            dropout=args['dropout'],
            )
        return model
    
    def train(self, model, train_loader, optimizer, criterion=None, scheduler=None, **kwargs):
        loss_record = LossRecord(["train_loss", "ce_loss", "mc_loss", "o_loss"])
        model.cuda()
        model.train()
                
        graph = train_loader.to(self.device)
        x = graph.x
        edge_weight = graph.edge_weight
        edge_index = graph.edge_index
        y = graph.y
        mask = graph.train_mask
        
        out, mc_loss, o_loss = model(x, edge_index, edge_weight)
        ce_loss = F.cross_entropy(out[mask], y[mask])
        loss = mc_loss + o_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_record.update({
            "train_loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "mc_loss": mc_loss.item(),
            "o_loss": o_loss.item(),
        })
            
        if scheduler is not None:
            scheduler.step()
        return loss_record
    
    def evaluate(self, model, eval_loader, criterion=None, split="valid", **kwargs):
        loss_record = LossRecord(["{}_loss".format(split), "nmi", "cs"])
        model.eval()
        with torch.no_grad():
            graph = eval_loader.to(self.device)
            x = graph.x
            edge_weight = graph.edge_weight
            edge_index = graph.edge_index
            y = graph.y
            mask = graph.val_mask if split == "valid" else graph.test_mask
            
            out, mc_loss, o_loss = model(x, edge_index, edge_weight, batch=graph.batch)
            nmi = normalized_mutual_info_score(out.max(1)[1].cpu(), y.cpu())
            cs = completeness_score(out.max(1)[1].cpu(), y.cpu())
            
            ce_loss = F.cross_entropy(out[mask], y[mask])
            loss = mc_loss + o_loss
            
            loss_record.update({
                "{}_loss".format(split): loss.item(),
                "nmi": nmi,
                "cs": cs,
                })
            
        return loss_record
    
    def process(self, model, train_loader, valid_loader, test_loader, optimizer, 
                criterion, regularizer=None, scheduler=None, **kwargs):
        if self.verbose:
            self.logger("Start training")

        best_epoch = 0
        best_metrics = None
        counter = 0
        
        for epoch in tqdm(range(self.epochs)):
            train_loss_record = self.train(model, train_loader, optimizer, criterion, scheduler)
            # if self.verbose:
            #     self.logger("Epoch {} | {} | lr: {:.6f}".format(epoch, train_loss_record, optimizer.param_groups[0]["lr"]))
            if self.wandb:
                wandb.log(train_loss_record.to_dict())
            
            if self.saving_checkpoint and (epoch + 1) % self.checkpoint_freq == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.cpu().state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss_record': train_loss_record.to_dict(),
                    }, os.path.join(self.saving_path, "checkpoint_{}.pth".format(epoch)))
                model.cuda()
                if self.verbose:
                    self.logger("Epoch {} | save checkpoint in {}".format(epoch, self.saving_path))
                
            if (epoch + 1) % self.eval_freq == 0:
                valid_loss_record = self.evaluate(model, valid_loader, criterion, split="valid")
                if self.verbose:
                    self.logger("Epoch {} | {}".format(epoch, valid_loss_record))
                valid_metrics = valid_loss_record.to_dict()
                
                if self.wandb:
                    wandb.log(valid_loss_record.to_dict())
                
                if not best_metrics or valid_metrics['nmi'] > best_metrics['nmi']:
                    counter = 0
                    best_epoch = epoch
                    best_metrics = valid_metrics
                    torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
                    model.cuda()
                    if self.verbose:
                        self.logger("Epoch {} | save best models in {}".format(epoch, self.saving_path))
                elif self.patience != -1:
                    counter += 1
                    if counter >= self.patience:
                        if self.verbose:
                            self.logger("Early stop at epoch {}".format(epoch))
                        break

        self.logger("Optimization Finished!")
        
        # load best model
        if not best_metrics:
            torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
        else:
            model.load_state_dict(torch.load(os.path.join(self.saving_path, "best_model.pth")))
            self.logger("Load best models at epoch {} from {}".format(best_epoch, self.saving_path))        
        model.cuda()
        
        valid_loss_record = self.evaluate(model, valid_loader, criterion, split="valid")
        self.logger("Valid metrics: {}".format(valid_loss_record))
        test_loss_record = self.evaluate(model, test_loader, criterion, split="test")
        self.logger("Test metrics: {}".format(test_loss_record))
        
        if self.wandb:
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary.update(test_loss_record.to_dict())
