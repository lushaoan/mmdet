# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class YOLOBBoxCoder(BaseBBoxCoder):
    """YOLO BBox coder.

    Following `YOLO <https://arxiv.org/abs/1506.02640>`_, this coder divide
    image into grids, and encode bbox (x1, y1, x2, y2) into (cx, cy, dw, dh).
    cx, cy in [0., 1.], denotes relative center position w.r.t the center of
    bboxes. dw, dh are the same as :obj:`DeltaXYWHBBoxCoder`.

    Args:
        eps (float): Min value of cx, cy when encoding.
    """

    def __init__(self, eps=1e-6):
        super(BaseBBoxCoder, self).__init__()
        self.eps = eps

    @mmcv.jit(coderize=True)
    def encode(self, bboxes, gt_bboxes, stride):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        作用是将gt box利用grid cell和anchor编码成网络输出的形式
        bboxes：由head中的sampler产生pos_bboxes，其实就是anchor
        gt_bboxes：原图上的gt_bbox
        两者的格式都是(x1, y1, x2, y2)，值都是原图大小

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., anchors.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
            stride (torch.Tensor | int): Stride of bboxes.

        Returns:
            torch.Tensor: Box transformation deltas
            返回的是center的delta
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
        y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        # 获取anchor的中心点和宽高
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        ''' 对应回笔记上的公式，在implement上，有出入，这里是使用center来写的，论文里是使用左上角来描述cx,cy
            以左上角来做，本质上是为了pred出来的东西不要超过一整个grid cell，用center做也一样
            笔记里写的是在featuremap size上来做，但这里的实现是使用原图大小（所有数据经过resizer）来做，其实也一样
         bx = sigmoid(tx) + cx
         by = sigmoid(ty) + cy
         bw = Pw * exp(tw)
         bh = Ph * exp(th)
         
         bx, by, bw, bh 是gt的数据
         tx, ty, tw, th 是pred的数据
         cx, cy, Pw, Ph 是anchor的数据
         
         那么，下面的代码就能对应上了
         w_target=tw, w_gt=bx, w=Pw
         x_center_target=tx, x_center_gt=bx, x_center=cx
        '''
        w_target = torch.log((w_gt / w).clamp(min=self.eps))
        h_target = torch.log((h_gt / h).clamp(min=self.eps))
        '''
        关于+0.5的个人理解
        x_center_gt 与 x_center位于同一个grid cell内，而x_center就在cell的正中心
        因此  -0.5 <= (x_center_gt - x_center) / stride <= 0.5
        回到上面的公式， bx-cx 应该是处于 (0, 1)范围内的，故而需要 +0.5 才能保持形式上的一致
        '''
        x_center_target = ((x_center_gt - x_center) / stride + 0.5).clamp(
            self.eps, 1 - self.eps)
        y_center_target = ((y_center_gt - y_center) / stride + 0.5).clamp(
            self.eps, 1 - self.eps)
        encoded_bboxes = torch.stack(
            [x_center_target, y_center_target, w_target, h_target], dim=-1)
        return encoded_bboxes

    @mmcv.jit(coderize=True)
    def decode(self, bboxes, pred_bboxes, stride):
        """Apply transformation `pred_bboxes` to `boxes`.

        bboxes: 这个是anchor_generator生成出来，其实就是anchor
        pred_bboxes: tx ty经过了sigmoid的结果

        Args:
            boxes (torch.Tensor): Basic boxes, e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            torch.Tensor: Decoded boxes.
            返回(x1, y1, x2, y2)格式
        """
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        # x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        # y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        # w = bboxes[..., 2] - bboxes[..., 0]
        # h = bboxes[..., 3] - bboxes[..., 1]
        # # Get outputs x, y
        # '''
        # 与上面encoder的公式对应，pred_bboxes是已经经过 sigmoid的输出，
        # 那么 -0.5 <= pred_bboxes[..., 0] - 0.5 <= 0.5
        # *stride + x_center后，就是得到他自己的center
        # '''
        # x_center_pred = (pred_bboxes[..., 0] - 0.5) * stride + x_center
        # y_center_pred = (pred_bboxes[..., 1] - 0.5) * stride + y_center
        # w_pred = torch.exp(pred_bboxes[..., 2]) * w
        # h_pred = torch.exp(pred_bboxes[..., 3]) * h

        xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (
            pred_bboxes[..., :2] - 0.5) * stride
        whs = (bboxes[..., 2:] -
               bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp()

        decoded_bboxes = torch.stack(
            (xy_centers[..., 0] - whs[..., 0], xy_centers[..., 1] -
             whs[..., 1], xy_centers[..., 0] + whs[..., 0],
             xy_centers[..., 1] + whs[..., 1]),
            dim=-1)
        return decoded_bboxes
