# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

'''
from http://www.360doc.com/content/20/1212/12/99071_951075725.shtml

该部分做的是确定正负样本，是在anchor维度上的。也就是确定所有的anchor哪些是正样本，哪些是负样本。
划分为正样本的anchor意味着负责gt box的预测，训练的时候就会计算gt box的loss。
而负样本表明该anchor没有负责任何物体，当然也需要计算loss，但是只计算confidence loss，因为没有目标，所以无法计算box loss 和类别loss。
Yolo还有一个设置就是忽略样本，也就是anchor和gt box有较大的iou，但是又不负责预测它，就忽略掉，不计算任何loss。
防止有错误的梯度更新到网络，也是为了提高网络的召回率。这里总结如下：

    正样本：负责预测gt box的anchor。loss计算box loss(包括中心点+宽高)+confidence loss + 类别loss。
    负样本：不负责预测gt box的anchor。loss只计算confidence loss。
    忽略样本：和gt box的iou大于一定阈值，但又不负责该gt box的anchor，一般指中心点grid cell附近的其他grid cell 里的anchor。不计算任何loss。

下面看具体实现。代码是同时确定gt box是分配在哪一层的哪一个或几个anchor上。
具体的类为GridAssigner，其中输入参数为：Bboxes,所有的anchor,box_responsible_flags,gt 
第一步分配的anchor flags，主要是记录在候选anchor中分配。和gt_bboxes。
该类遍历batch，维护一个assigned_gt_inds，类似mask的概念，元素值会被分配为-1：忽略样本，0：负样本，正整数：正样本，
同时数字代表负责的gt box的索引。具体步骤如下：

第一步，将所有的assigned_gt_inds设置为-1，默认为忽略样本。

第二步，将所有iou小于一定值例如0.5（或者在一定区间的），设置为0，置为负样本。
gt box和全部anchor计算iou，这里的boxes为anchor，是带有位置信息的。
获取的overlaps 尺度为gt box个数*全部anchor个数（这里为300+1200+4800=6300）。

overlaps = self.iou_calculator(gt_bboxes, bboxes) # 获取全部iou，size为gt个数X6300
max_overlaps, argmax_overlaps = overlaps.max(dim=0) # 找和所有gtbox最大的iou，size为6300，也就是看看每一个anchor，
和所有gt box最大的iou有无大过阈值
assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps <= self.neg_iou_thr)] = 0 #如果小于阈值，例如0.5，设置为负样本，不负责任何gt的预测。

第三步，将全部iou中，非负责gt的（记录在box_responsible_flags，非中心点grid cell的anchor）置为-1，
该步骤首先排除掉非中心点grid cell的anchor。因为排除掉的部分肯定不是正样本。
#获取和哪一个gt最大的iou，size为6300，和上一步类似，不过获取的都是负责gt box的grid cell里的anchor
max_overlaps, argmax_overlaps = overlaps.max(dim=0)
# 获取的iou和一定阈值对比，例如0.5，大于该值，设置为正样本。
## 可见这一步是将gt box对应的grid cell 里面大于一定阈值的anchor设置为正样本，可能是多个anchor。
pos_inds = (max_overlaps > self.pos_iou_thr) & box_responsible_flags.type(torch.bool)
assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#获取全部gt和哪一个anchor最大的iou，尺度为gt的数目，例如有2个gt，那么size就是2
gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)
# 遍历gt box，找到其最大的anchor，且在负责的grid cell中，设置为正样本。
# 因为上一步，有些gt box并找不到iou大于阈值的anchor，这部分也是要预测的，所以退而求其次，找最大iou的anchor负责它，
当然也是在gt box自己的grid cell里的anchor中寻找。
for i in range(num_gts):
if gt_max_overlaps[i] > self.min_pos_iou:
if self.gt_max_assign_all:
max_iou_inds = (overlaps[i, :] == gt_max_overlaps[i]) & \
box_responsible_flags.type(torch.bool)
assigned_gt_inds[max_iou_inds] = i + 1

至此，全部anchor全部分配完成，总结一下：

    全部anchor，和gt box的iou小于阈值的，设置为负样本；
    正样本来自两部分：第一是gt box对应的grid cell里的anchor，iou大于阈值的。
    第二部分是gt box对应grid cell里的anchor，和gt box iou 最大的那一个；
    其余部分，设置为忽略样本；

可以看出，上面2中的第二部分的正样本是最后计算了，因此理论上所有gt box都会分配一个和自己iou最大的anchor。
如果预先被2中第一部分分配了，有可能会被其他gt box挤走，也就是标签重写现象。这个以后可以重点分析一下。
还可以看出，一个gt box 可以有多个anchor，但是一个anchor只能负责一个gt box。
可以理解为，正负样本的分配在训练该样本之前已经做好了，和训练的好坏以及预测的结果并无关系。
当然还有另外一种实现方式是：忽略样本由训练过程中的真实预测的box和gt box算iou，较大的且没有被分配到的为忽略样本，是一种动态的分配方式
'''

@BBOX_ASSIGNERS.register_module()
class GridAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxes, box_responsible_flags, gt_bboxes, gt_labels=None):
        """Assign gt to bboxes. The process is very much like the max iou
        assigner, except that positive samples are constrained within the cell
        that the gt boxes fell in.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts <= neg_iou_thr to 0
        3. for each bbox within a cell, if the iou with its nearest gt >
            pos_iou_thr and the center of that gt falls inside the cell,
            assign it to that bbox
        4. for each gt bbox, assign its nearest proposals within the cell the
            gt bbox falls in to itself.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            box_responsible_flags (Tensor): flag to indicate whether box is
                responsible for prediction, shape(n, )
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all gt and bboxes
        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # 2. assign negative: below
        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        # shape of max_overlaps == argmax_overlaps == num_bboxes
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps <= self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, (tuple, list)):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps > self.neg_iou_thr[0])
                             & (max_overlaps <= self.neg_iou_thr[1])] = 0

        # 3. assign positive: falls into responsible cell and above
        # positive IOU threshold, the order matters.
        # the prior condition of comparision is to filter out all
        # unrelated anchors, i.e. not box_responsible_flags
        overlaps[:, ~box_responsible_flags.type(torch.bool)] = -1.

        # calculate max_overlaps again, but this time we only consider IOUs
        # for anchors responsible for prediction
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        # shape of gt_max_overlaps == gt_argmax_overlaps == num_gts
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        pos_inds = (max_overlaps >
                    self.pos_iou_thr) & box_responsible_flags.type(torch.bool)
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign positive to max overlapped anchors within responsible cell
        for i in range(num_gts):
            if gt_max_overlaps[i] > self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = (overlaps[i, :] == gt_max_overlaps[i]) & \
                         box_responsible_flags.type(torch.bool)
                    assigned_gt_inds[max_iou_inds] = i + 1
                elif box_responsible_flags[gt_argmax_overlaps[i]]:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        # assign labels of positive anchors
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]

        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
