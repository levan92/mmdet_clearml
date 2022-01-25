#!/usr/bin/env python3
import torch
from timm.models import create_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Torchscript Converter')
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                        help='timm model_name to link model-arch')
    parser.add_argument('--checkpoint', default='*.pth.tar', type=str, metavar='WILDCARD',
                        help='load the best checkpoint')
    parser.add_argument('--num-classes', type=int, default=5, metavar='N',
                    help='number of label classes (default = 5 for topk)')
    parser.add_argument('--output', default='weights/resnet_best.pt', type=str, metavar='WILDCARD',
                        help='output file name (default = weights/resnet_best.pt')

    args = parser.parse_args()

    model = create_model(model_name=args.model, num_classes=args.num_classes,
                     in_chans=3, checkpoint_path=args.checkpoint)
    sm = torch.jit.script(model)
    sm.save(args.output)
