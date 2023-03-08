


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
from reid.cluster_utils import get_inter_cam_cluster_result_tnorm
from reid.utils.data.sampler import RandomIdentitySampler

import debugpy





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

def show_batch_dataloader(dataloader):
    #returns a tuple of (image_pixels, filenames, person_ids, camera_ids).
    for batch_images in dataloader:
        #print(batch_images)
        
        print(next(iter(batch_images)))
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
   
    
    # print("trainloader one batch")
    # show_batch_dataloader(train_loader)
    # print("valloader one")
    # show_batch_dataloader(val_loader)
    
    
    
    # Create model
    #ft_net_inter_Tnorm=backbone+classifier
    model = models.create("ft_net_inter_TNorm",
                          domain_number=camera_number[args.dataset],
                          num_classes=num_classes,
                          stride=args.stride,
                          init_weight=args.init_weight)#the output of the model is feature vector
    
    # .....(classifier): Sequential(
    #(0): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #(1): Linear(in_features=2048, out_features=751, bias=False)
    #print("model in inter ",model)
    
    # Load from checkpoint
    start_epoch = 0
    best_top1 = 0
    top1 = 0
    is_best = False
    
      #if we use pretrained model
    if args.checkpoint is not None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        #if mode=evaluation, we have market traiend model
        if args.evaluate:

            checkpoint = load_checkpoint(args.checkpoint)
            param_dict = model.state_dict()
            for k, v in checkpoint['state_dict'].items():
                if 'model' in k and k in param_dict.keys():
                    param_dict[k] = v
            model.load_state_dict(param_dict)
            
        #otherwise, we have resnet pretraiend mdoel
        else:
            model.model.load_param(args.checkpoint)
         
    model=model.to(device)
      #model = model.cuda()

    # Distance metric
    metric = None

    # Evaluator
    
    evaluator = Evaluator(model, use_cpu=args.use_cpu)
      ##################################################################  
    #if mode=evaluation
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
            #############################################################################

    train_transformer = [
        T.Resize((args.height, args.width), interpolation=3),
        T.RandomHorizontalFlip(),
        T.Pad(10),
        T.RandomCrop((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=0.5),
    ]
    train_transformer = T.Compose(train_transformer)
                #########################repeat 40 times for clustering 
                #and in each round, 3 times for training intra
                #and 2 times for training inter model
                    #cluster-epochs=40
    for cluster_epoch in range(args.cluster_epochs):
                    # -------------------------Stage 1 intra camera training--------------------------
                    # Cluster and generate new dataset and model
        cluster_result = get_intra_cam_cluster_result(model, train_loader,
                                                      args.eph_stage1,
                                                      args.linkage
                                                
                                                      )
    
        #len of cluster result for first time is 6- in total we have 116 images and 5 images are ignored in cluster result
        #another finding is that , the total batches are processing not just one batch
        #another finding is that, beacuse we have 6 cameras in realted to images, we have 6 dic.key and each dic has soem images and labels
        #one dict.key[0]=21
        #one dict.key[1]=19
        #one dict.key[2]=26
        #one dict.key[3]=4
        #one dict.key[4]=18
        #one dict.key[5]=27
        #another finding is that each item is ('00000179_05_0000.jpg', 0)#filenale,label
        #another finding is about how to cluster in cluster function, in the function, we have a loop for each camera because clustering is done for each camera
        
       
        #cluster_result[3] is OrderedDict([('00001368_03_0001.jpg', 0), ('00001368_03_0002.jpg', 0), ('00001017_03_0000.jpg', 1), ('00001017_03_0001.jpg', 1)])
        
       
       
        
        
        
        cluster_datasets = [
            datasets.create("cluster", osp.join(args.data_dir, args.dataset),
                            cluster_result[cam_id], cam_id)
            for cam_id in sorted(cluster_result.keys())
        ]
        
        
        
        cluster_dataloaders = [
            DataLoader(
                Preprocessor(dataset.train_set,
                                    root=dataset.images_dir,
                                    transform=train_transformer,
                                    mutual=True),
                
                       batch_size=args.batch_size,
                       num_workers=args.workers,
                       shuffle=True,
                       pin_memory=False,
                       drop_last=True) for dataset in cluster_datasets
        ]
        
        
        #my finding is that as we know the batch has (image_pixels, filenames, person_ids, camera_ids) so in cluster_dataloaders[4], the list of camera_ids is 4 beacsue the key in cluster_result 
        #has 6 dicts and each dict has seveal images, so all images in one cluster has teh same cam
        #another finding is that when we have 4 images in one dataset, dataloader is empty because batch_size is 8 and the author throw way the batches with different sizes, when we change batch_size 
        #to 2, so batch size shows 4 images
        #another finding is that dataloader was a tuple of (image_pixels, filenames, person_ids, camera_ids). but cluster_dataloader that was created after clustering has one extra item
        #another fiding is that we add one transformed imag with respect to each image, these identical images may have different value pixels because some transformation
        #is applied to the copied image or if the copied image is loaded from file, the pixel values may have diferent values because of compression and encoding 
        
        # show_batch_dataloader(cluster_dataloaders[3])
        # exit(0)
        
        #print(len(cluster_dataloaders))=6
        
     
     
     
        param_dict = model.model.state_dict()
        
        #num_classes are a list containing the number of different labels/classes in each dataset
        #the number of noerons in the last layer in classifiers is the same as number of classes
        model = models.create(
            "ft_net_intra_TNorm",
            num_classes=[dt.classes_num for dt in cluster_datasets],
            stride=args.stride,
            init_weight=args.init_weight)
        
        """model consists of a backbone and 
            (classifier): ModuleList(
        (0): Sequential(
        (0): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=2048, out_features=3, bias=False)
        )
        (1): Sequential(
        (0): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=2048, out_features=3, bias=False)
        )
        (2): Sequential(
        (0): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=2048, out_features=3, bias=False)
        )
        (3): Sequential(
        (0): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=2048, out_features=2, bias=False)
        )
        (4): Sequential(
        (0): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=2048, out_features=4, bias=False)
        )
        (5): Sequential(
        (0): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Linear(in_features=2048, out_features=4, bias=False)
        )"""
            
        #creates a dictionary of the current state of the model's parameters, model_param_dict, using the state_dict() method.     
        model_param_dict = model.model.state_dict()
        
        for k, v in model_param_dict.items():#The code then loops through each key-value pair in model_param_dict using the items() method.
            #If k is present in param_dict.keys(), the corresponding value of model_param_dict[k] is replaced with the value of param_dict[k].
            if k in param_dict.keys():
                model_param_dict[k] = param_dict[k]
        #the updated model_param_dict is loaded back into the model using the load_state_dict() method.
        model.model.load_state_dict(model_param_dict)
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(device)
        #model = model.cuda()
        criterion = nn.CrossEntropyLoss().to(device)
        #criterion = nn.CrossEntropyLoss().cuda()
        soft_criterion = SoftEntropy().to(device)
        #soft_criterion = SoftEntropy().cuda()
        # Optimizer
        param_groups = make_params(model, args.lr, args.weight_decay)
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
        # Trainer
        trainer = IntraCameraSelfKDTnormTrainer(
                                                model,
                                                criterion,
                                                soft_criterion,
                                                warm_up_epoch=args.warm_up,
                                                multi_task_weight=args.multi_task_weight,)
        
        # Start training
        #epochs_stage1=3
        #we have 6 cluster_dataloaders, so the first for is run 6 times
        # for i, inputs in enumerate(zip(*cluster_dataloaders)):
        #     for domain, domain_input in enumerate(inputs):
        #             imgs1, imgs2, _, pids, _ = domain_input
        #             imgs1 = imgs1.cuda()
        #             imgs2 = imgs2.cuda()
        #             targets = pids.cuda()#use pseudo labels
        #             print(domain)
        
        
        
        
        
        for epoch in range(0, args.epochs_stage1):
            trainer.train(
                cluster_epoch,
                epoch,
                cluster_dataloaders,
                optimizer,
                
                print_freq=5
            )
      

            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)

            save_checkpoint(
                {
                    'state_dict': model.state_dict(),
                    'epoch': cluster_epoch + 1,
                    'best_top1': best_top1,
                    'cluster_epoch': cluster_epoch + 1,
                },
                is_best,
                fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        if cluster_epoch == (args.cluster_epochs - 1):
            save_checkpoint(
                {
                    'state_dict': model.state_dict(),
                    'epoch': cluster_epoch + 1,
                    'best_top1': best_top1,
                    'cluster_epoch': cluster_epoch + 1,
                },
                False,
                fpath=osp.join(args.logs_dir, 'latest.pth.tar'))

        print('\n * cluster_epoch: {:3d} top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(cluster_epoch, top1, best_top1, ' *' if is_best else ''))



    print("Final test for evaluation trained model on test set")
   
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'latest.pth.tar'))#change the name of checkpont to load
    new_state_dict = {}
    for k in checkpoint['state_dict'].keys():
        if 'model' in k:
            new_state_dict[k] = checkpoint['state_dict'][k]
    model.load_state_dict(new_state_dict, strict=False)
    
    best_rank1, mAP = evaluator.evaluate_tnorm(
        test_loader,
        dataset.query,
        dataset.gallery,
        metric,
        return_mAP=True,
        camera_number=camera_number[args.dataset])
    
    best_rank1_2, mAP2 = evaluator.evaluate(
        test_loader,
        dataset.query,
        dataset.gallery,
        metric,
        return_mAP=True,
    )
    print("Tnorm: Rank1: {} mAP: {}\t Normal: Rank1: {} mAP: {}".format(
        best_rank1, mAP, best_rank1_2, mAP2))







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
    parser.add_argument('-b', '--batch-size', type=int, default=2 )#default is 8
    
#default=64
    parser.add_argument('-b2', '--batch-size-stage2', type=int, default=32)
    parser.add_argument('--instances', default=8)
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
    torch.cuda.set_device(1)
