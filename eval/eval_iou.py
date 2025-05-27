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
#import torch.quantization
from torch.quantization import get_default_qconfig
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

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

    #model = ERFNet(NUM_CLASSES)
    #model_file = importlib.import_module(args.loadModel[:-3])
    
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
                elif args.loadModel != "erfnet.py":
                    own_state["module."+name].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    #print ("Model and weights LOADED successfully")
    
    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # ---------------- QUANTIZATION ----------------
    if args.quantize:
        print("Preparing FX Graph Mode quantization...")

        # Must run on CPU for quantization
        model = model.cpu()
        model.eval()

        # Specify quantization config (static)
        qconfig_mapping = get_default_qconfig_mapping("fbgemm")

        # Dummy input (shape must match your training input)
        example_inputs = torch.randn(1, 3, 512, 1024)

        # FX prepare step: insert observers for calibration
        model_prepared = prepare_fx(model, qconfig_mapping, example_inputs)

        print("Calibrating model...")
        # Calibrate the model using a few samples
        with torch.no_grad():
            for i, (images, labels, _, _) in enumerate(loader):
                model_prepared(images.cpu())
                if i >= 10:
                    break

        # Convert to quantized model
        model_quantized = convert_fx(model_prepared)

        print("Model quantized.")
        model = model_quantized  # Replace model with quantized version
        model.eval()
        torch.save(model.state_dict(), "quantized_model.pth")
    # ---------------- ENDING QUANTIZATION ----------------

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    #iouEvalVal = iouEval(NUM_CLASSES)
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
    for i in tqdm(range(iou_classes.size(0)), desc="Processing IoU classes"):
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
    parser.add_argument('--quantize', action='store_true')

    main(parser.parse_args())