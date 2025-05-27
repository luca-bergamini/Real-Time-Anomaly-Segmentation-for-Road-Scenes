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
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig

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
    
    model = model_file.Net(NUM_CLASSES, aux_model="eval")

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
        # 2. Set up qconfig - you can use default per-tensor symmetric quantization for weights and activations:
        qconfig = get_default_qconfig('fbgemm')  # good backend for x86 CPUs

        # 3. Prepare your model for quantization - this inserts observers
        qconfig_dict = {"": qconfig}  # applies to all layers by default
        model_prepared = prepare_fx(model, qconfig_dict)

        # 4. Calibration step: run some data through the model to collect stats for quantization
        # Use a small calibration dataset or some batches from your training/validation set
        calibration_data = torch.randn(8, 3, 640, 480).cuda()  # example dummy data
        with torch.no_grad():
            model_prepared(calibration_data)

        # 5. Convert the model to a quantized model
        model_quantized = convert_fx(model_prepared)
        
        model = model_quantized
        model.eval()
    # ---------------- ENDING QUANTIZATION ----------------

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    #iouEvalVal = iouEval(NUM_CLASSES)
    if args.void:
        iouEvalVal = iouEval(NUM_CLASSES, 20)
    else:
        iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(tqdm(loader, desc="Evaluating")):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()
        else:
            images = images.cpu()
            labels = labels.cpu()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)
        
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[1] if len(outputs) > 1 else outputs[0]

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 

        #print (step, filenameSave)
        
    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in tqdm(range(iou_classes.size(0)), desc="Processing IoU classes"):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
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