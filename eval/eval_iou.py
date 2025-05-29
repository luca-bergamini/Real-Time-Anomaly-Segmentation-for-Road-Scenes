# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time
import sys
from tqdm import tqdm

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.utils.prune as prune
from torch.quantization import get_default_qconfig
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from dataset import cityscapes
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry
from fvcore.nn import FlopCountAnalysis

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    #print ("Loading model: " + modelpath)
    #print ("Loading weights: " + weightspath)

    model_path = args.loadModel
    
    if os.path.splitext(os.path.basename(args.loadModel))[0] == "bisenet":
        # Add the `train/` directory to sys.path
        train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train"))
        if train_dir not in sys.path:
            sys.path.insert(0, train_dir)
        model_name = "bisenet"

        if not os.path.isabs(model_path):
            # Convert to absolute path relative to current working directory
            model_path = os.path.abspath(model_path)

        spec = importlib.util.spec_from_file_location(model_name, model_path)
        model_file = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_file)
    elif os.path.splitext(os.path.basename(args.loadModel))[0] == "enet":
        # Add the `train/` directory to sys.path
        train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train"))
        if train_dir not in sys.path:
            sys.path.insert(0, train_dir)
        model_name = "enet"

        if not os.path.isabs(model_path):
            # Convert to absolute path relative to current working directory
            model_path = os.path.abspath(model_path)

        spec = importlib.util.spec_from_file_location(model_name, model_path)
        model_file = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_file)
    else:
        model_file = importlib.import_module(args.loadModel[:-3])
        
    model = model_file.Net(NUM_CLASSES)

    #model = torch.nn.DataParallel(model)
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

     # ---------------- PRUNING ----------------

    def prune_model(model, amount=0.3):
        print(f"Applying unstructured L1 pruning with {amount * 100}% sparsity to Conv2d layers...")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=amount)
                # Optional: remove pruning reparameterization so weights are actually pruned
                prune.remove(module, 'weight')
        return model

    if args.pruning> 0:
        model = prune_model(model, amount=args.pruning)

    def count_nonzero_parameters(model):
        total_params = 0
        nonzero_params = 0
        for param in model.parameters():
            total_params += param.numel()
            nonzero_params += param.nonzero().size(0)
        return total_params, nonzero_params

    total, nonzero = count_nonzero_parameters(model)
    print(f"Total parameters: {total} | Non-zero (effective) parameters after pruning: {nonzero}")
    print(f"Pruned percentage: {(1 - nonzero / total) * 100:.2f}%")

        # === FLOPs and theoretical time estimation ===
    dummy_input = torch.randn(1, 3, 512, 1024).to(next(model.parameters()).device)
    model_for_flops = model.module if isinstance(model, torch.nn.DataParallel) else model

    flop_analyzer = FlopCountAnalysis(model_for_flops, dummy_input)
    total_flops = flop_analyzer.total()

    total_params, nonzero_params = count_nonzero_parameters(model)
    sparsity_ratio = nonzero_params / total_params

    effective_flops = total_flops * sparsity_ratio
    print(f"Total FLOPs: {effective_flops / 1e9:.2f} GFLOPs")

    t4_flops_per_sec = 641.19e9  # 641 GFLOPS/s to match real inference time
    theoretical_time_sec = effective_flops / t4_flops_per_sec
    print(f"Estimated time: {theoretical_time_sec:.6f} seconds")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    if args.void:
        iouEvalVal = iouEval(NUM_CLASSES, 20)
    else:
        iouEvalVal = iouEval(NUM_CLASSES)

    total_inference_time = 0.0
    num_images = 0

    for step, (images, labels, filename, filenameGt) in enumerate(tqdm(loader)):
        if not args.cpu:
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)

        if not args.cpu:
            torch.cuda.synchronize()  # Make sure all CUDA ops are done before timing
        start_infer = time.time()

        with torch.no_grad():
            outputs = model(inputs)

        if not args.cpu:
            torch.cuda.synchronize()  # Wait for GPU ops to finish

        end_infer = time.time()
        total_inference_time += (end_infer - start_infer)
        num_images += images.size(0)  # count per-image even with batch_size > 1

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("=======================================")
    print(f"Avg inference time per image: {total_inference_time / num_images:.4f} seconds")
    print(f"Total inference time (model only): {total_inference_time:.2f} seconds for {num_images} images")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    if args.void:
        print(iou_classes_str[19], "void")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--void', action='store_true')
    parser.add_argument('--pruning', type=float, default=0.0, help="Amount of structured pruning (0 to disable)")
    parser.add_argument('--quantize', action='store_true')

    main(parser.parse_args())