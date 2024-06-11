#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import torch
import torch.optim as optim
import numpy as np
import csv

from terminaltables import AsciiTable
from torchsummary import summary

from model import load_model
from dataloader import _create_data_loader
from utils import to_cpu, print_environment_info, provide_determinism
from loss import compute_loss


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="./models/trained_best.pth")
    parser.add_argument("-fp", "--file_path", type=str, help="Path to data")
    parser.add_argument("--model_save", type=str, default='./models', help="Model save path")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-m", "--model", type=str, default=None, help="Train from checkpoint.")
    parser.add_argument("--name", type=str, default='ASCAD', help="Name of network model")
    parser.add_argument("--lr", type=float, default=1e-4, help="")
    parser.add_argument("-bs", "--batch_size", type=int, default=256, help="")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="")

    parser.add_argument("--seed", type=int, default=1, help="Makes results reproducable. Set -1 to disable.")
    parser.add_argument("-v","--verbose", type=str, default=True, help="")

    parser.add_argument("-tl", "--trace_length", type=int, default=1400, help="")
    parser.add_argument("--nb_train", type=int, default=90000, help="")
    parser.add_argument("--nb_valid", type=int, default=10000, help="")
    parser.add_argument("--nb_class", type=int, default=256, help="")
    
    parser.add_argument("-s", "--stride", type=int, default=32, help="window stride")


    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # fix random seed
    if args.seed != -1:
        provide_determinism(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # #################
    # Create Dataloader
    # #################
    X_train = np.load(args.file_path + '/X_train.npy')
    Y_train = np.load(args.file_path + '/Y_train.npy')
    X_valid = np.load(args.file_path + '/X_valid.npy')
    Y_valid = np.load(args.file_path + '/Y_valid.npy')
    
    # print(X_train[args.nb_train:args.nb_train+args.nb_valid].shape, Y_train[args.nb_train:args.nb_train+args.nb_valid].shape)

    kwargs_train = {
        'trs': X_train[0:args.nb_train,:],
        'label': Y_train[0:args.nb_train],
        'trace_num':args.nb_train
    }
    kwargs_valid = {
        'trs': X_valid[0:args.nb_valid,:],
        'label': Y_valid[0:args.nb_valid],
        'trace_num':args.nb_valid,
    }
    train_loader = _create_data_loader(args.batch_size, kwargs_train)
    valid_loader = _create_data_loader(args.batch_size, kwargs_valid)


    # ############
    # Create model
    # ############
    model = load_model(nbclass=args.nb_class, model_path=args.model, name=args.name)
    summary(model, input_size=(1, args.trace_length))
    

    # ################
    # Create optimizer
    # ################
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)


    # ###########
    # Train Model
    # ###########
    min_loss = 1000

    for epoch in range(1, args.epochs+1):

        print("\n---- Training Model ----")

        model.train()  # Set model to training mode
        epoch_losses_train = {}
        epoch_losses_valid = {}

        for batch_i, (trs, targets) in enumerate(tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            # batches_done = len(train_loader) * epoch + batch_i
            trs, targets = trs.to(device), targets.to(device)
            # print(trs.shape, targets.shape)

            optimizer.zero_grad()
            outputs = model(trs)

            loss, loss_dict = compute_loss(predictions=outputs, targets=targets, stride=args.stride)

            loss.backward()
            optimizer.step()

            if batch_i==0:
                epoch_losses_train = loss_dict
            else:
                for k in epoch_losses_train:
                    epoch_losses_train[k] += loss_dict[k]

        for k in epoch_losses_train:
            epoch_losses_train[k] /= batch_i+1

        # ########
        # Evaluate
        # ########
        print("\n---- Evaluating Model ----")
        model.eval()
        with torch.no_grad():
            for batch_i, (trs, targets) in enumerate(tqdm.tqdm(valid_loader, desc=f"Validation Epoch {epoch}")):
                # Evaluate the model on the validation set
                trs, targets = trs.to(device), targets.to(device)
                outputs = model(trs)

                loss, loss_dict = compute_loss(predictions=outputs, 
                                               targets=targets, 
                                               stride=args.stride)

                if batch_i==0:
                    epoch_losses_valid = loss_dict
                else:
                    for k in epoch_losses_valid:
                        epoch_losses_valid[k] += loss_dict[k]

        for k in epoch_losses_valid:
            epoch_losses_valid[k] /= batch_i+1

        # ############
        # Log progress
        # ############
        if args.verbose:
            print(f'Epoch-{epoch}')
            print(AsciiTable(
                [
                    ["Loss&Metric",             "Train",                                    "Validation"],
                    ["leak loss",     epoch_losses_train['loss_conf_leak'],  epoch_losses_valid['loss_conf_leak']],
                    ["no_leak loss",  epoch_losses_train['loss_conf_noleak'],epoch_losses_valid['loss_conf_noleak']],
                    ["IoU loss",      epoch_losses_train['loss_iou'],        epoch_losses_valid['loss_iou']],
                    ["class loss",    epoch_losses_train['loss_cls'],        epoch_losses_valid['loss_cls']],
                    ["total loss",    epoch_losses_train['loss'],            epoch_losses_valid['loss']]
                ]).table)

        if epoch == 1:
            with open('./output/train_loss_'+args.name+'.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=epoch_losses_train.keys())
                writer.writeheader()
            with open('./output/valid_loss_'+args.name+'.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=epoch_losses_valid.keys())
                writer.writeheader()

        with open('./output/train_loss_'+args.name+'.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=epoch_losses_train.keys())
            writer.writerow(epoch_losses_train)
        with open('./output/valid_loss_'+args.name+'.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=epoch_losses_valid.keys())
            writer.writerow(epoch_losses_valid)

        # #############
        # Save model
        # #############
        if epoch_losses_valid['loss'] < min_loss:
            min_loss = epoch_losses_valid['loss']
            checkpoint_path = args.model_save
            print(f"---- Saving checkpoint to: '{checkpoint_path+f'/{args.name}_best.pth'}' ----")
            torch.save({"epoch": epoch, 
                        "model_parameter": model.state_dict()}, 
                       checkpoint_path+f'/{args.name}_best.pth')


if __name__ == "__main__":
    run()
