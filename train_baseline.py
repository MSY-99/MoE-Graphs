import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import f1_score

from models import NodeGCN
from models import NodeGIN
from models import NodeGAT

from utils import *
from torch_geometric.transforms import Compose, NormalizeFeatures, ToUndirected
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB

from torch.optim.lr_scheduler import (
    SequentialLR, LinearLR, CosineAnnealingLR,
    ReduceLROnPlateau, StepLR
)

import plotext as plt

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--dataset_name', default='bace', type=str, required=True)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--gpu', default='cuda:2', type=str)
parser.add_argument('--model', default='GCN', type=str, required= True)
parser.add_argument('--model_name', default='GCN', type=str)

parser.add_argument('--sched', default='cosine', choices=['cosine','plateau','step','none'])
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--lr_min', type=float, default=1e-5)
parser.add_argument('--step_size', type=int, default=30)   # for step
parser.add_argument('--gamma', type=float, default=0.1) 

def pick_mask(mask, idx):
    return mask[:, idx] if mask.dim() == 2 else mask

def main():
    start = time.time()
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')

    # ----- Data ----- #
    name_lower = args.dataset_name.lower()

    if name_lower in ['cora', 'pubmed', 'citeseer']:
        dataset = Planetoid(
            root=f'./data/{name_lower}',
            name=name_lower.capitalize(),
            transform=Compose([NormalizeFeatures(), ToUndirected()])
        )
        data = dataset[0]

    elif name_lower in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(
            root=f'./data/{name_lower}',
            name=name_lower,
            geom_gcn_preprocess=True,
            transform=Compose([NormalizeFeatures(), ToUndirected()])
        )
        data = dataset[0]
        
    elif name_lower in ['texas','wisconsin','cornell']:
        webkb_name = 'Cornell' if name_lower in ['cornell'] else name_lower.capitalize()
        dataset = WebKB(
            root=f'./data/{webkb_name.lower()}',
            name=webkb_name,
            transform=Compose([NormalizeFeatures(), ToUndirected()])
        )
        data = dataset[0]

    data = data.to(device)

    in_dim = data.x.size(-1)
    out_dim = int(data.y.max().item() + 1)
    hidden_dim = 64

    # ----- Data masking ----- #
    train_mask = pick_mask(data.train_mask, 0)
    val_mask   = pick_mask(data.val_mask, 0)
    test_mask  = pick_mask(data.test_mask, 0)

    print(f'dataset {args.dataset_name} | task node | num_classes {out_dim} | '
            f'model {args.model} | seed {args.seed} | split_idx 0')

    # ----- Model ----- #
    if args.model == 'GCN':
        model = NodeGCN(in_dim, hidden_dim, out_dim, dropout=0.2).to(device)

    elif args.model == 'GIN':
        model = NodeGIN(in_dim, hidden_dim, out_dim, dropout=0.2, learn_eps=True).to(device)

    elif args.model == 'GAT':
        model = NodeGAT(in_dim, hidden_dim, out_dim, heads1=1, dropout=0.2, feat_dropout=0.5, negative_slope=0.2).to(device)
    
    else:
        raise ValueError(f'Unknown model {args.model} for Planetoid node classification')
    
    # ----- Optimizer/ Loss ----- #
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    criterion   = nn.CrossEntropyLoss()

    # ----- Learning rate scheduler ----- #
    if args.sched == 'cosine':
        warm = max(0, min(args.warmup_epochs, args.epochs-1))
        main = max(1, args.epochs - warm)
        sched_warm = LinearLR(optimizer, start_factor=0.1, total_iters=warm)        # 워밍업
        sched_main = CosineAnnealingLR(optimizer, T_max=main, eta_min=args.lr_min)  # 코사인
        scheduler = SequentialLR(optimizer, schedulers=[sched_warm, sched_main], milestones=[warm])

    elif args.sched == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                    patience=20, min_lr=args.lr_min, verbose=False)

    elif args.sched == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    else:
        scheduler = None

    # ----- Train ----- #  
    best_val, best_state = -1.0, None
    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # ----- Validation ----- #
        model.eval()
        with torch.no_grad():
            logits = model(data)
            pred = logits.argmax(dim=-1)

            val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
            val_f1 = f1_score(data.y[val_mask].cpu(), pred[val_mask].cpu(), average='macro')

            val_loss  = criterion(logits[val_mask], data.y[val_mask])
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if scheduler is not None:
            if args.sched == 'plateau':
                scheduler.step(val_loss.item())
            else:
                scheduler.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f'\rEpoch {epoch}/{args.epochs} | lr {cur_lr:.2e} | loss {loss:.4f} | '
                    f'val {val_acc:.4f}/{val_f1:.4f} | ',end='')
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    
    save_dir = './best_model'
    model_dir = os.path.join(save_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)
    fname = f"{args.model}_{args.dataset_name}_{args.seed}.pt"
    save_path = os.path.join(model_dir, fname)
    torch.save(model.state_dict(), save_path)

    # ----- Test ----- #
    model.eval()
    with torch.no_grad():
        logits = model(data)
        pred = logits.argmax(dim=-1)
        test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
        test_f1  = f1_score(data.y[test_mask].cpu(), pred[test_mask].cpu(), average='macro')

    print(f'\nTest Acc {test_acc:.4f} | F1 {test_f1:.4f}')

    # ----- Save results ----- #
    result_dir = f'./results/original'
    os.makedirs(result_dir, exist_ok=True)
    result_csv_path = os.path.join(result_dir, f'{args.model}_results.csv')
    if not os.path.exists(result_csv_path):
        pd.DataFrame(columns=['model','params','seed','dataset','acc','f1']).to_csv(result_csv_path, index=False)
    
    result = {
        'model': args.model_name,
        'params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'seed': args.seed,
        'dataset': args.dataset_name,
        'acc': test_acc,
        'f1': test_f1,
    }

    results = pd.read_csv(result_csv_path) if os.path.exists(result_csv_path) else pd.DataFrame(columns=result.keys())
    results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
    results.to_csv(result_csv_path, index=False)

    # ----- Show epoch loss ----- #
    # epochs = list(range(1, len(train_losses) + 1))

    # plt.clear_figure()
    # plt.clc()

    # plt.canvas_color("black")
    # plt.axes_color("black")

    # plt.plot(epochs, train_losses, label="train", color="blue")
    # plt.plot(epochs, val_losses,   label="valid", color="hot pink")

    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.title(f"{args.model} loss curve")

    # plt.show()

    end = time.time()
    total_time = end - start
    print(f'Result save complete: {result_csv_path}')
    print(f'Time: {total_time//60:.0f}m {total_time%60:.0f}s\n')


if __name__ == '__main__': main()