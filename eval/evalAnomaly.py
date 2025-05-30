# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
import importlib
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F
from torch.ao.quantization import QConfigMapping, QConfig, get_default_qat_qconfig
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver, FixedQParamsObserver
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import sys
import time
import torch.nn.utils.prune as prune
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
def prune_model(model, amount=0.3):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=amount)
                # Optional: remove pruning reparameterization so weights are actually pruned
                prune.remove(module, 'weight')
        return model

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
    parser.add_argument('--method', default='MSP', choices=['MSP', 'MaxLogit', 'MaxEntropy', 'Void'],
                    help="Choose OOD scoring method: MSP, MaxLogit, or MaxEntropy")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="Temperature scaling for softmax/logit OOD scoring")
    parser.add_argument('--quantize', action='store_true')
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

    #model_file = importlib.import_module(args.loadModel[:-3])
    #model = ERFNet(NUM_CLASSES)
    
    if os.path.splitext(os.path.basename(args.loadModel))[0] == "bisenet":
        # Add the `train/` directory to sys.path
        train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train"))
        if train_dir not in sys.path:
            sys.path.insert(0, train_dir)
        
        model_path = args.loadModel
        model_name = "bisenet"

        if not os.path.isabs(model_path):
            # Convert to absolute path relative to current working directory
            model_path = os.path.abspath(model_path)

        spec = importlib.util.spec_from_file_location(model_name, model_path)
        model_file = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_file)
    else:
        model_file = importlib.import_module(args.loadModel[:-3])
    
    model = model_file.Net(NUM_CLASSES)

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

    model.eval()

    image_transform = Compose([Resize((512, 1024), Image.BILINEAR), ToTensor()])
    target_transform = Compose([Resize((512, 1024), Image.NEAREST)])
    image_paths = glob.glob(os.path.expanduser(str(args.input[0])))
    
    # ---------------- QUANTIZATION ----------------
    if args.quantize:
        print("Preparing FX Graph Mode quantization...")

        # Use CPU for quantization
        model = model.cpu()
        model.eval()

        # Static quantization configuration
        custom_qconfig = QConfig(
            activation=MinMaxObserver.with_args(quant_min=0, quant_max=255, dtype=torch.quint8),
            weight=PerChannelMinMaxObserver.with_args(quant_min=-128, quant_max=127, dtype=torch.qint8)
        )

        qconfig_mapping = QConfigMapping().set_global(custom_qconfig)

        # Optional: add default mapping for fixed-qparams ops
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping("fbgemm")
        qconfig_mapping.set_global(custom_qconfig)
        example_inputs = torch.randn(1, 3, 512, 1024)

        # Prepare the model for calibration
        model_prepared = prepare_fx(model, qconfig_mapping, example_inputs)

        print("Calibrating model...")

        with torch.no_grad():
            for i, path in enumerate(image_paths):
                image = image_transform((Image.open(path).convert('RGB'))).unsqueeze(0)
                model_prepared(image)
                if i >= 10:  # Use a few images for calibration
                    break

        # Convert to quantized model
        model_quantized = convert_fx(model_prepared)

        print("Model quantized.")
        model = model_quantized
        torch.save(model.state_dict(), "quantized_model_anomaly.pth")
    # ---------------- ENDING QUANTIZATION ----------------
    
    total_inference_time = 0.0
    num_images = 0
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):

        images_tensor = image_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()

        if args.cpu:
            images = images_tensor.cpu()
        else:
            images = images_tensor.cuda()
            
        if not args.cpu:
            torch.cuda.synchronize()  # Ensure all GPU ops are done before timing
            
        start_infer = time.time()
        
        with torch.no_grad():
            if not os.path.splitext(os.path.basename(args.loadModel))[0] == "bisenet":
                result = model(images) / args.temperature
            else:
                result = model(images)

        if not args.cpu:
            torch.cuda.synchronize()  # Wait for GPU ops to finish

        end_infer = time.time()
        total_inference_time += (end_infer - start_infer)
        num_images += 1

        if os.path.splitext(os.path.basename(args.loadModel))[0] == "bisenet":
            result = result[0]
            result = result / args.temperature

        if args.method == 'Void':
            anomaly_result = F.softmax(result, dim=1)[:, 19, :, :]
            anomaly_result = anomaly_result.data.cpu().numpy().squeeze()
        elif args.method == 'MSP':
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
        
    if num_images > 0:
        avg_infer_time = total_inference_time / num_images
        print("=======================================")
        print(f"Avg inference time per image: {avg_infer_time:.4f} seconds")
        print(f"Total inference time (model only): {total_inference_time:.2f} seconds for {num_images} images")
        print("=======================================")

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