


from __future__ import print_function, absolute_import

import argparse
import os.path as osp
import shutil
import os
import numpy as np
import sys
import torch
import itertools
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


sys.path.append(os.getcwd())


from reid.loss import TripletLoss, SoftEntropy, SoftTripletLoss
from reid.loss.entropy_regularization import SoftEntropy

from reid import datasets
from reid import models
from reid.trainers import IntraCameraSelfKDTnormTrainer
from reid.trainers import InterCameraSelfKDTNormTrainer
from reid.evaluators_cos import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.cluster_utils import get_intra_cam_cluster_result
from reid.cluster_utils import get_inter_cam_cluster_result_tnorm,get_inter_cam_cluster_result,get_inter_cam_cluster_result_without_tnorm
from reid.utils.data.sampler import RandomIdentitySampler

def get_data(
            name,
            split_id,
            data_dir,
            height,
            width,
            batch_size,
            workers,
        ):
            root = osp.join(data_dir, name)

            dataset = datasets.create(name, root, split_id=split_id)

            normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            
            train_set = dataset.trainval
            num_classes = dataset.num_trainval_ids

            train_transformer = T.Compose([
                T.Resize((height, width), interpolation=3),
                T.ToTensor(),
                normalizer,
            ])

            test_transformer = T.Compose([
                T.Resize((height, width), interpolation=3),
                T.ToTensor(),
                normalizer,
            ])

            train_loader = DataLoader(Preprocessor(train_set,
                                                root=dataset.images_dir,
                                                transform=train_transformer),
                                    batch_size=batch_size,
                                    num_workers=workers,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)

            val_loader = DataLoader(Preprocessor(dataset.val,
                                                root=dataset.images_dir,
                                                transform=test_transformer),
                                    batch_size=batch_size,
                                    num_workers=workers,
                                    shuffle=False,
                                    pin_memory=False)

            test_loader = DataLoader(Preprocessor(
                list(set(dataset.query) | set(dataset.gallery)),
                root=dataset.images_dir,
                transform=test_transformer),
                                    batch_size=batch_size,
                                    num_workers=workers,
                                    shuffle=False,
                                    pin_memory=False)

            return dataset, num_classes, train_loader, val_loader, test_loader

def make_params(model, lr, weight_decay):
    params = []
    for key, value in model.model.named_parameters():
        if not value.requires_grad:
            continue

        params += [{
            "params": [value],
            "lr": lr * 0.1,
            "weight_decay": weight_decay
        }]
    for key, value in model.classifier.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    return params

def get_mix_rate(mix_rate, epoch, num_epoch, power=0.6):
    return mix_rate * (1 - epoch / num_epoch)**power

    #batch-size=8
    #batch_size_stage2=64
    #cluster_epochs=40
    #epoch_stage1=3
    #epoch_stage2=2


#show the one batch of dataloader
def show_batch_dataloader(dataloader):
    #returns a tuple of (image_pixels, filenames, person_ids, camera_ids).
    for batch_images in dataloader:
        print("batch_images",batch_images)
        exit(0)
    
def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
 
    shutil.copy(sys.argv[0], osp.join(args.logs_dir,
                                      osp.basename(sys.argv[0])))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (256, 128)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size * 8, args.workers,
                 )
    camera_number = {"market1501": 6, "dukemtmc": 8, "msmt17": 15, "viper": 2}
    
   
    model = models.create("ft_net_inter_convnext",
                          domain_number=camera_number[args.dataset],
                          num_classes=num_classes,
                          pretrained=True)#the output of the model is feature vector
    
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    model=model.to(device)
 
    metric = None
    evaluator = Evaluator(model, use_cpu=args.use_cpu)
    if args.evaluate:
        evaluator.evaluate_tnorm(
            test_loader,
            dataset.query,
            dataset.gallery,
            metric,
            return_mAP=True,
            camera_number=camera_number[args.dataset],
        )
        evaluator.evaluate(
            test_loader,
            dataset.query,
            dataset.gallery,
            metric,
        )
        return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('--checkpoint', type=str, metavar='PATH')
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        default='market1501',
                        choices=datasets.names())
    parser.add_argument('--eph_stage1', type=float, default=0.0025)
    parser.add_argument('--eph_stage2', type=float, default=0.0017)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--init_weight', type=float, default=0.1)
    parser.add_argument('--mix_rate',
                        type=float,
                        default=0.01,
                        help="mu in Eq (5)")
    parser.add_argument('--decay_factor', type=float, default=0.6)
#default=8
    parser.add_argument('-b', '--batch-size', type=int, default=8 )#default is 8
    
#default=64
    parser.add_argument('-b2', '--batch-size-stage2', type=int, default=32)
    parser.add_argument('--instances', default=4)
    #default=4
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height',
                        type=int,
                        help="input height, default: 256 for resnet*, "
                        "144 for inception")
    parser.add_argument('--width',
                        type=int,
                        help="input width, default: 128 for resnet*, "
                        "56 for inception")
    # optimizer
    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help="learning rate of new parameters, for pretrained "
                        "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--evaluate',
                        action='store_true',
                        help="evaluation only")
    parser.add_argument(
        '--use_cpu',
        action='store_true',
        help='use cpu to calculate dist to prevent from GPU OOM')
    parser.add_argument('--epochs_stage1', type=int, default=3)
    parser.add_argument('--epochs_stage2', type=int, default=2)
    parser.add_argument('--cluster_epochs', type=int, default=40)
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--start_save',
                        type=int,
                        default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--linkage', type=str, default="average")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--multi_task_weight', type=float, default=1.)
    
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir',
                        type=str,
                        metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir',
                        type=str,
                        metavar='PATH',
                        default=osp.join(working_dir, './logs'))
    
    main(parser.parse_args())
   
