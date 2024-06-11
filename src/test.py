#! /usr/bin/env python3

from __future__ import division

import argparse
import tqdm
import numpy as np
import torch

from model import load_model
from utils import solution_filtering, select_cls_pred, print_environment_info
from utils import plot_guessing_entropy, plot_instance_result, plot_pred_AOI_freq
from dataloader import _create_data_loader
import csv
import os


def _evaluate(model, dataloader, thres_conf, thres_nms, max_det):
    """Evaluate model.
    -------------
    Input:
    - model: Model to evaluate
    - dataloader: Dataloader provides the batches of traces (drop_last=False, shuffle=False)
    - thres_conf:
    - thres_nms:
    - max_det:
    --------------
    Output: ndarray format.
    - preds: Regression solutions (nbattack, nl, 3+nbclass)
    - preds_mask: the solution mask (nbattack, nl)
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set model to evaluation mode
    preds = torch.tensor([]).to(device)
    preds_mask = torch.tensor([]).bool().to(device)
    with torch.no_grad():
        for trs in tqdm.tqdm(dataloader, desc="Testing"):
            trs = trs.to(device)
            outputs = model(trs)
            solution_mask = solution_filtering(prediction=outputs, 
                                               thres_conf=thres_conf, 
                                               thres_nms=thres_nms, 
                                               max_det=max_det)

            preds = torch.cat([preds, outputs], dim=0)
            preds_mask = torch.cat([preds_mask, solution_mask], dim=0)

    return preds.cpu().numpy(), preds_mask.cpu().numpy()



def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Testing.")
    parser.add_argument("-fp", "--file_path", type=str, help="Path to data")
    parser.add_argument("-m", "--model", type=str, help="Path to model")
    parser.add_argument("--name", type=str, default='ASCAD', help="Name of network model")
    parser.add_argument("-bs", "--batch_size", type=int, default=2000, help="Size of each image batch")
    
    parser.add_argument("-tl", "--trace_length", type=int, default=700, help="")
    parser.add_argument("-nc", "--nb_class", type=int, default=9, help="")
    parser.add_argument("-na", "--nb_attack", type=int, default=200, help="")
    parser.add_argument("-to", "--trace_offset", type=int, default=45400, help="")

    parser.add_argument("--thres_nms", type=float, default=0.5, help="")
    parser.add_argument("--thres_conf", type=float, default=0.75, help="")
    parser.add_argument("--max_det", type=int, default=10, help="")

    parser.add_argument("-k", "--key", type=int, default=0x1, help="")
    parser.add_argument("-ws", "--stride", type=int, default=32, help="window stride")
    parser.add_argument("-b", "--byte", type=int, default=None, help="target byte in case contain more than one.")

    parser.add_argument("--frequency", type=str, default='False', help="")
    parser.add_argument("--showinstance", type=int, default=None, help="")
    parser.add_argument("--GE", type=str, default='False', help="")
    parser.add_argument("--save_plot", type=str, default='True', help="")

    args = parser.parse_args()
    print(f"Command line arguments: {args}")


    X_attack = np.load(args.file_path + '/X_attack.npy')
    X_attack = X_attack[0:args.nb_attack]

    kwargs_test = {
        'trs': X_attack,
        'label': None,
        'trace_num': args.nb_attack
    }

    test_loader = _create_data_loader(args.batch_size, kwargs_test, shuffle=False, drop_last=False)

    model = load_model(nbclass=args.nb_class, model_path=args.model, name=args.name)

    preds, preds_mask = _evaluate(model=model,
                                  dataloader=test_loader,
                                  thres_conf=args.thres_conf,
                                  thres_nms=args.thres_nms,
                                  max_det=args.max_det)
    
    preds = preds*args.stride # rescale
    savefile = args.name+f'_{args.trace_length}' if args.save_plot=='True' else None
    
    if args.showinstance is not None:
        assert args.showinstance < args.nb_attack, \
            print("Demonstrated insatnces could not excess the total number of attack instances.") 
        label = np.load(args.file_path + '/Y_attack.npy')[:args.showinstance]
        
        plot_instance_result(traces = X_attack[:args.showinstance],
                            AOI_pred = preds[:args.showinstance],
                            AOI_mask = preds_mask[:args.showinstance],
                            AOI_target = label,  
                            startat=args.trace_offset,
                            savefile=savefile)
        
    if args.frequency=='True':
        freq = plot_pred_AOI_freq(AOIs=preds[...,:2], 
                           mask=preds_mask, 
                           trace_length=args.trace_length, 
                           startat=args.trace_offset, 
                           savefile=savefile)
        if args.save_plot != 'True':
            with open('./output/freq_'+args.name+'.csv', 'a') as f:
                np.savetxt(f, [freq], delimiter=",", newline="\n")

        
    if args.GE=='True':
        label = np.load(args.file_path + '/Y_attack.npy')[:args.nb_attack]
        plaintext = np.load(args.file_path + '/P_attack.npy')[:args.nb_attack]
        mask = None
        if os.path.exists(args.file_path + '/R_attack.npy'):
            mask = np.load(args.file_path + '/R_attack.npy')[:args.nb_attack]
        label_method = 'HW' if args.nb_class==9 else 'ID'

        pred_solution, idx_solution = select_cls_pred(AOI_target=label[:,:2],
                                                      pred=preds,
                                                      pred_mask=preds_mask,
                                                      thres_overlap=0.8)
        
        if len(idx_solution)>0:
            if mask is not None:
                mask = mask[idx_solution, args.byte]
            rst = plot_guessing_entropy(preds=pred_solution[:, 3:],
                                        plaintext=plaintext[idx_solution, args.byte],
                                        mask=mask,
                                        real_key=args.key,
                                        label_method=label_method,
                                        savefile=savefile)
            if args.save_plot != 'True':
                with open('./output/GE_'+args.name+'.csv', 'a') as f:
                    np.savetxt(f, [rst['GE']], delimiter=",", newline="\n")
        else:
            print('No AOI has been detected.')



if __name__ == "__main__":
    run()
