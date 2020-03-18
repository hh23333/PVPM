from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch

import torchreid
from torchreid.engine import engine
from torchreid.losses import CrossEntropyLoss, Isolate_loss, Part_similarity_constrain
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics


class ImageSoftmaxEngine(engine.Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_cpu (bool, optional): use cpu. Default is False.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torch
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(self, datamanager, model, optimizer, scheduler=None, use_cpu=False,
                 label_smooth=True):
        super(ImageSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_cpu)
        
        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch+1)<=fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
            
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self._compute_loss(self.criterion, outputs, pids)
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            losses.update(loss.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                num_batches = len(trainloader)
                eta_seconds = batch_time.avg * (num_batches-(batch_idx+1) + (max_epoch-(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'Eta {eta}'.format(
                      epoch+1, max_epoch, batch_idx+1, len(trainloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accs,
                      lr=self.optimizer.param_groups[0]['lr'],
                      eta=eta_str
                    )
                )
            
            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()


class PoseSoftmaxEngine(engine.Engine):
    def __init__(self, datamanager, model, optimizer, scheduler=None, use_cpu=False,
                 label_smooth=True):
        super(PoseSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_cpu)

        # TODO modify the criterion for pairwise comparison
        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.att_criterion = Isolate_loss()

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        att_loss = AverageMeter()

        self.model.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch + 1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        end = time.time()
        # TODO modify pose_dataloader
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)

            imgs, pids, pose_heatmaps = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
                pose_heatmaps = pose_heatmaps.cuda()

            self.optimizer.zero_grad()
            # TODO add pose subnet to model
            outputs, attmaps = self.model(imgs, pose_heatmaps)
            loss_att = self.att_criterion(attmaps)
            loss = self._compute_loss(self.criterion, outputs, pids)
            loss = loss+loss_att
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            att_loss.update(loss_att.item(), pids.size(0))
            losses.update(loss.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

            if (batch_idx + 1) % print_freq == 0:
                # estimate remaining time
                num_batches = len(trainloader)
                eta_seconds = batch_time.avg * (num_batches - (batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'attLoss {attloss.val:.4f} ({attloss.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'Eta {eta}'.format(
                    epoch + 1, max_epoch, batch_idx + 1, len(trainloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    attloss=att_loss,
                    acc=accs,
                    lr=self.optimizer.param_groups[0]['lr'],
                    eta=eta_str
                )
                )

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        pose_heatmaps = data[4]
        return imgs, pids, pose_heatmaps

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        pose_heatmaps = data[4]
        return imgs, pids, camids, pose_heatmaps
    def _extract_features(self, img, pose):
        self.model.eval()
        return self.model(img, pose)

class PoseSoftmaxEngine_wscorereg(engine.Engine):
    '''
    Pose guided softmaxEngine with visibility score regression
    '''
    def __init__(self, datamanager, model, optimizer, scheduler=None, use_cpu=False,
                 label_smooth=True, use_att_loss=True, reg_matching_score_epoch=0, num_att=6):
        super(PoseSoftmaxEngine_wscorereg, self).__init__(datamanager, model, optimizer, scheduler, use_cpu)

        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.part_c_criterion = Part_similarity_constrain(part_num=num_att).cuda()
        self.use_att_loss = use_att_loss
        if self.use_att_loss:
            self.att_criterion = Isolate_loss()
        self.reg_matching_score_epoch = reg_matching_score_epoch

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        use_matching_loss=False
        if epoch>=self.reg_matching_score_epoch:
            use_matching_loss=True
        losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        att_losses = AverageMeter()
        part_losses = AverageMeter()
        matching_losses = AverageMeter()

        self.model.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch + 1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)
        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)

            imgs, pids, pose_heatmaps = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
                pose_heatmaps = pose_heatmaps.cuda()

            self.optimizer.zero_grad()
            outputs, attmaps, part_score, v_g = self.model(imgs, pose_heatmaps)
            #classification loss
            loss_class = self._compute_loss(self.criterion, outputs, pids)
            # using for weighting each part with visibility
            # loss_class = self._compute_loss(self.criterion, outputs, pids, part_score.detach())
            loss_matching,loss_partconstr = self.part_c_criterion(v_g, pids, part_score, use_matching_loss)
            # add matching loss
            loss = loss_class+loss_partconstr
            # visibility verification loss
            if use_matching_loss:
                loss=loss+loss_matching
                matching_losses.update(loss_matching.item(), pids.size(0))
            if self.use_att_loss:
                loss_att = self.att_criterion(attmaps)
                loss=loss+loss_att
                att_losses.update(loss_att.item(), pids.size(0))
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            losses.update(loss.item(), pids.size(0))
            part_losses.update(loss_partconstr.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

            if (batch_idx + 1) % print_freq == 0:
                # estimate remaining time
                num_batches = len(trainloader)
                eta_seconds = batch_time.avg * (num_batches - (batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'part_Loss {loss_part.val:.4f} ({loss_part.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'Eta {eta}'.format(
                    epoch + 1, max_epoch, batch_idx + 1, len(trainloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_part=part_losses,
                    acc=accs,
                    lr=self.optimizer.param_groups[0]['lr'],
                    eta=eta_str
                )
                ,end='\t'
                )
                if self.use_att_loss:
                    print('attLoss {attloss.val:.4f} ({attloss.avg:.4f})'.format(attloss=att_losses),end='\t')
                if use_matching_loss:
                    print('matchLoss {match_loss.val:.4f} ({match_loss.avg:.4f})'.format(match_loss=matching_losses),end='\t')
                print('\n')

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        pose_heatmaps = data[4]
        return imgs, pids, pose_heatmaps

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        pose_heatmaps = data[4]
        return imgs, pids, camids, pose_heatmaps
    def _extract_features(self, img, pose):
        self.model.eval()
        return self.model(img, pose)