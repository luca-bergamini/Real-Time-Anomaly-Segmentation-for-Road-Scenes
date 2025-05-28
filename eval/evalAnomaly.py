# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
import copy
import torch.nn.utils.prune as prune
import time
from fvcore.nn import FlopCountAnalysis
import torch.nn as nn

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# --- PRUNE MODEL ---
def prune_model(model, amount):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model

# --- COUNT NONZERO PARAMETERS ---
def count_nonzero_parameters(model):
    total_params = 0
    nonzero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += param.nonzero().size(0)
    return total_params, nonzero_params

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default='MSP',  choices=['MSP', 'MaxLogit', 'MaxEntropy', 'Void'],
                    help="Choose OOD scoring method: MSP, MaxLogit, or MaxEntropy")
    parser.add_argument('--temperature', type=float, default=1.0,
                    help="Temperature scaling for softmax/logit OOD scoring")
    parser.add_argument('--pruning', type=float, default=0.0 )
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    #print ("Loading model: " + modelpath)
    #print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    #print ("Model and weights LOADED successfully")

    # --- PRUNE ---
    if args.pruning > 0.0:
        model = prune_model(model, amount=args.pruning)

    # --- DEEP COPY FOR SAFE ANALYSIS ---
    model_for_analysis = copy.deepcopy(model.module if isinstance(model, torch.nn.DataParallel) else model)
    model_for_analysis.eval()

    # --- FLOPs ANALYSIS ---
    dummy_input = torch.randn(1, 3, 512, 1024).to(next(model.parameters()).device)
    with torch.no_grad():
        flop_analyzer = FlopCountAnalysis(model_for_analysis, dummy_input)
        total_flops = flop_analyzer.total()

    # --- EFFECTIVE FLOPs ---
    total_params, nonzero_params = count_nonzero_parameters(model_for_analysis)
    sparsity_ratio = nonzero_params / total_params
    effective_flops = total_flops * sparsity_ratio

    print(f"Total Pruned FLOPs: {effective_flops / 1e9:.2f} GFLOPs")

    # --- THEORETICAL INFERENCE TIME ---
    t4_flops_per_sec = 641.19e9  # T4 throughput in FLOPs/s
    theoretical_time_sec = effective_flops / t4_flops_per_sec
    print(f"Estimated theoretical inference time: {theoretical_time_sec:.6f} seconds")

    model.eval()

    image_transform = Compose([Resize((512, 1024), Image.BILINEAR), ToTensor()])
    target_transform = Compose([Resize((512, 1024), Image.NEAREST)])
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):

        images = image_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()

        with torch.no_grad():
            start_time = time.time()
            result = model(images) / args.temperature
            end_time = time.time()
            print(f"Real inference time: {(end_time - start_time):.6f} seconds")

        if args.method == 'Void':
            anomaly_result = torch.nn.functional.softmax(result, dim=1)[:, 19, :, :]
            anomaly_result = anomaly_result.data.cpu().numpy().squeeze()

        if args.method == 'MSP':
            softmax_probs = torch.nn.functional.softmax(result, dim=1)
            msp = torch.max(softmax_probs, dim=1)[0].cpu().numpy().squeeze()
            anomaly_result = 1.0 - msp

        elif args.method == 'MaxLogit':
            max_logit = torch.max(result, dim=1)[0]
            anomaly_result = -max_logit.cpu().numpy().squeeze()

        elif args.method == 'MaxEntropy':
            probs = torch.nn.functional.softmax(result, dim=1)
            log_probs = torch.log(probs + 1e-8)
            entropy = -torch.sum(probs * log_probs, dim=1)
            anomaly_result = entropy.cpu().numpy().squeeze()   

        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        ood_gts = np.array(target_transform(mask))

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()