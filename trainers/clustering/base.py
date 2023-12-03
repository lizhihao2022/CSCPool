import logging
import wandb
import torch
from time import time 

from datasets import ClusterDataset
from .mincut import MinCutTrainer
from .csc import CSCTrainer
from .diff import DiffTrainer

TRAINER_DICT = {
    'MinCutGCN': MinCutTrainer,
    'CSCGCN': CSCTrainer,
    'DiffGCN': DiffTrainer,
}


def clustering_procedure(args):
    if args['model_name'] not in TRAINER_DICT.keys():
        raise NotImplementedError("Model {} not implemented".format(args['model_name']))
    
    if args['verbose']:
        logger = logging.info if args['log'] else print

    if args['wandb']:
        wandb.init(
            project=args['wandb_project'], 
            name=args['saving_name'],
            tags=[args['model_name'], args['dataset']],
            config=args)
    
    # load data
    if args['verbose']:
        logger("Loading {} dataset, subset is {}".format(args['dataset'], args['subset']))
    start = time()
    dataset = ClusterDataset(
        name=args['dataset'],
        data_dir=args['data_dir'],
    )
    if args['verbose']:
        logger("Loading data costs {: .2f}s".format(time() - start))
    
    # build model
    if args['verbose']:
        logger("Building models")
    start = time()
    trainer = TRAINER_DICT[args['model_name']](args)
    args['node_dim'] = dataset.num_features
    args['output_dim'] = dataset.num_classes
    model = trainer.build_model(args)
    model = model.to(args['device'])
    optimizer = torch.optim.Adam(
        model.parameters(), 
        betas=(0.9, 0.999),
        lr=args['lr'],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args['milestones'],
        gamma=args['gamma'],
    )
    
    if args['verbose']:
        logger("Model: {}".format(model))
        logger("Optimizer: {}".format(optimizer))
        logger("Scheduler: {}".format(scheduler))
        logger("Building models costs {: .2f}s".format(time() - start))

    trainer.process(
        model=model,
        train_loader=dataset.data,
        valid_loader=dataset.data,
        test_loader=dataset.data,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=None,
    )
