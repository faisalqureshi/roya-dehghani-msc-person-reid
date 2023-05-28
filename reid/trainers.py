from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .evaluation_metrics import accuracy
from .loss import TripletLoss
from .utils.meters import AverageMeter
import numpy as np



class IntraCameraSelfKDTnormTrainer(object):
    def __init__(
        self,
        model_1,
        entropy_criterion,
        soft_entropy_criterion,
        warm_up_epoch=-1,# a warm-up epoch parameter (warm_up_epoch) that specifies the number of epochs during which only the cross-entropy loss is used for training, before introducing the soft target loss.
        multi_task_weight=1.0
    ):
        super(IntraCameraSelfKDTnormTrainer, self).__init__()
        #model1=ft_net_intra_TNorm
        self.model_1 = model_1
        self.T = 1.
        self.entropy_criterion = entropy_criterion
        self.soft_entropy_criterion = soft_entropy_criterion
        self.warm_up_epoch = warm_up_epoch
        self.multi_task_weight = multi_task_weight
    #The goal of the trainer is to minimize the cross-entropy loss between the predicted outputs of the model and the true targets.
    #The "_forward" function takes in input data, targets, and a domain index, and calculates the cross-entropy loss, the precision 
    # (accuracy), and the soft cross-entropy loss between the predicted outputs of the model and the true targets
    def _forward(self, inputs1, inputs2, targets, i):
        convert = np.random.rand() > 0.5
        outputs1 = self.model_1(inputs1, i)
        # outputs2 = self.model_1(inputs2, i, convert=convert)
        outputs2 = self.model_1(inputs2, i)#new

        loss_ce1 = self.entropy_criterion(outputs1, targets)
        prec1, = accuracy(outputs1.data, targets.data)
        prec1 = prec1[0]
        #soft_loss1 is the soft cross-entropy loss between the outputs2 and outputs1 divided by the temperature squared.
        #The soft cross-entropy loss is calculated between the outputs of the model on two randomly augmented versions of the same input.
       
        soft_loss1 = self.soft_entropy_criterion(outputs2 / self.T, (outputs1 / self.T).detach()) * self.T * self.T

        return loss_ce1, prec1, soft_loss1

    def train(
        self,
        cluster_epoch,
        epoch,
        data_loader,
        optimizer,
        print_freq=1,
    ):
        #model1=classifier in module list
        self.model_1.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce1 = AverageMeter()
        precisions_1 = AverageMeter()
        losses_soft1 = AverageMeter()

        end = time.time()
        data_loader_size = min([len(l) for l in data_loader])

        for i, inputs in enumerate(zip(*data_loader)):
            data_time.update(time.time() - end)
            #we have 6 data-loaders and domain is the cam_num/domain and domain_input is imgs,copied_imgs,fnames,pids,cam_ids
            for domain, domain_input in enumerate(inputs):
                imgs1, imgs2, _, pids, _ = domain_input
                imgs1 = imgs1.cuda()
                imgs2 = imgs2.cuda()
                targets = pids.cuda()#use pseudo labels

                loss1, prec1, soft_loss1 = self._forward(imgs1, imgs2, targets, domain)
                if domain == 0:
                    loss1_sum = loss1
                    soft_loss1_sum = soft_loss1
                else:
                    loss1_sum = loss1_sum + loss1
                    soft_loss1_sum = soft_loss1_sum + soft_loss1

                losses_ce1.update(loss1.item(), targets.size(0))
                precisions_1.update(prec1, targets.size(0))
                losses_soft1.update(soft_loss1.item(), targets.size(0))
            #a multi-task weight parameter used for scaling the soft cross-entropy loss
            final_loss = loss1_sum + soft_loss1_sum * self.multi_task_weight
            # If the current epoch is less than the warm-up epoch, the final loss is multiplied by a factor of 0.1
            if cluster_epoch < self.warm_up_epoch:
                final_loss = final_loss * 0.1
            # Finally, the gradients are zeroed, the backward pass is calculated, and the optimizer is updated.
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
        
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Cluster_Epoch: [{}]\t'
                      'Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce1 {:.3f} ({:.3f})\t'
                      'Prec1 {:.2%} ({:.2%})\t'
                      'Loss_soft1 {:.3f} ({:.3f})\t'.format(
                          cluster_epoch, epoch, i + 1, data_loader_size,
                          batch_time.val, batch_time.avg, data_time.val,
                          data_time.avg, losses_ce1.val, losses_ce1.avg,
                          precisions_1.val, precisions_1.avg,
                          losses_soft1.val, losses_soft1.avg))


class InterCameraSelfKDTNormTrainer(object):
    def __init__(
        self,
        model_1,
        entropy_criterion,
        triplet_criterion,
        soft_entropy_criterion,
        triple_soft_criterion,
        warm_up_epoch=-1,
        multi_task_weight=1.,
    ):
        super(InterCameraSelfKDTNormTrainer, self).__init__()
        self.model_1 = model_1
        
        self.entropy_criterion = entropy_criterion
        self.triplet_criterion = triplet_criterion
        self.soft_entropy_criterion = soft_entropy_criterion
        self.triple_soft_criterion = triple_soft_criterion
        self.warm_up_epoch = warm_up_epoch
        self.multi_task_weight = multi_task_weight
        self.T = 1.

    def _forward(self, inputs1, inputs2, targets):
        convert = np.random.rand() > 0.5
        prob1, distance1 = self.model_1(inputs1)
        # prob2, distance2 = self.model_1(inputs2, convert=convert)
        prob2, distance2 = self.model_1(inputs2)#new

        loss_ce1 = self.entropy_criterion(prob1, targets)
        prec1, = accuracy(prob1.data, targets.data)
        prec1 = prec1[0]

        soft_loss1 = self.soft_entropy_criterion(prob2 / self.T, (prob1 / self.T).detach()) * self.T * self.T

        loss_triple1, prec_triple1 = self.triplet_criterion(distance1, targets)

        loss_triple1_soft = self.triple_soft_criterion(distance2, distance1.detach(), targets)

        return loss_ce1, prec1, loss_triple1, prec_triple1, \
               soft_loss1, loss_triple1_soft

    def train(
        self,
        cluster_epoch,
        epoch,
        data_loader,
        optimizer,
        print_freq=1,
    ):
        self.model_1.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce1 = AverageMeter()
        precisions_1 = AverageMeter()
        losses_triple1 = AverageMeter()
        precisions_triple1 = AverageMeter()
        losses_soft1 = AverageMeter()
        losses_triple_soft = AverageMeter()

        end = time.time()
        data_loader_size = len(data_loader)

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            imgs1, imgs2, _, pids, _ = inputs

            imgs1 = imgs1.cuda()
            imgs2 = imgs2.cuda()
            targets = pids.cuda()

            # TODO: whether use soft triple loss
            loss_ce1, prec1, loss_triple1, prec_triple1, \
            loss_soft1, loss_triple1_soft = self._forward(
                imgs1, imgs2, targets)

            losses_ce1.update(loss_ce1.item(), targets.size(0))
            precisions_1.update(prec1, targets.size(0))
            losses_triple1.update(loss_triple1.item(), targets.size(0))
            precisions_triple1.update(prec_triple1, targets.size(0))
            losses_soft1.update(loss_soft1, targets.size(0))
            # Note: add soft triple
            losses_triple_soft.update(loss_triple1_soft, targets.size(0))

            final_loss = loss_ce1 + loss_triple1 + loss_soft1 + loss_triple1_soft
            
            if cluster_epoch < self.warm_up_epoch:
                final_loss = final_loss * 0.1

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Cluster_Epoch: [{}]\t'
                      'Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce1 {:.3f} ({:.3f})\t'
                      'Prec1 {:.2%} ({:.2%})\t'
                      'Loss_triple1 {:.3f} ({:.3f})\t'
                      'Prec_triple1 {:.2%} ({:.2%})\t'
                      'Loss_soft1 {:.3f} ({:.3f})\t'
                      'Loss_triple_soft {:.3f} ({:.3f})\t'.format(
                          cluster_epoch, epoch, i + 1, data_loader_size,
                          batch_time.val, batch_time.avg, data_time.val,
                          data_time.avg, losses_ce1.val, losses_ce1.avg,
                          precisions_1.val, precisions_1.avg,
                          losses_triple1.val, losses_triple1.avg,
                          precisions_triple1.val, precisions_triple1.avg,
                          losses_soft1.val, losses_soft1.avg,
                          losses_triple_soft.val, losses_triple_soft.avg))
