#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.07.09'
__copyright__ = 'Copyright 2021, PI'


import numpy as np
import cv2
import onnxruntime
from einops import rearrange
from pymvcam import MVCam
import copy


yolo_thre = 0.7
kpt_input_size = [192, 256]
colors = [(0, 255, 255),
          (0, 255, 0),
          (255, 0, 0),
          (0, 0, 255),
          ]


def box2cs(box, input_size):
    x, y, w, h = box[:4]
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def get_affine_transform(center, scale, rot, output_size, shift=(0., 0.), inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray[N, K, H, W]: Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        heatmap height: H
        heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
            heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
            heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
            heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        batch size: B
        num keypoints: K
        num persons: N
        hight of heatmaps: H
        width of heatmaps: W
        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        res (np.ndarray[N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)
    batch_heatmaps = np.transpose(batch_heatmaps,
                                  (2, 3, 0, 1)).reshape(H, W, -1)
    batch_heatmaps_pad = cv2.copyMakeBorder(
        batch_heatmaps, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
    batch_heatmaps_pad = np.transpose(
        batch_heatmaps_pad.reshape(H + 2, W + 2, B, K),
        (2, 3, 0, 1)).flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(np.int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords


def keypoints_from_heatmaps(heatmaps, center, scale, unbiased=False, post_process='default',
                            kernel=11, valid_radius_factor=0.0546875, use_udp=False, target_type='GaussianHeatMap'):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        batch size: N
        num keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (str/None): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP.
        use_udp (bool): Use unbiased data processing.
        target_type (str): 'GaussianHeatMap' or 'CombinedTarget'.
            GaussianHeatMap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    # normalize configs
    if post_process is False:
        post_process = None
    elif post_process is True:
        if unbiased is True:
            post_process = 'unbiased'
        else:
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            post_process = 'unbiased'

    # start processing
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        assert target_type in ['GaussianHeatMap', 'CombinedTarget']
        if target_type == 'GaussianHeatMap':
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type == 'CombinedTarget':
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(np.int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == 'unbiased':  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(
                np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals



if __name__ == '__main2__':
    yolo_onnx_file = '/media/pi/ssdMobileDisk/open-mmlab/mmdetection/work_dir/line5/line.onnx'
    kpt_onnx_file = '/media/pi/ssdMobileDisk/open-mmlab/mmpose/work_dir/hrnet/line2/line.onnx'

    yolo_session = onnxruntime.InferenceSession(yolo_onnx_file)
    kpt_session = onnxruntime.InferenceSession(kpt_onnx_file)
    yolo_input_name = yolo_session.get_inputs()[0].name
    yolo_output_name = yolo_session.get_outputs()[0].name
    kpt_input_name = kpt_session.get_inputs()[0].name
    kpt_output_name = kpt_session.get_outputs()[0].name

    # img_file = '/media/lsa/MobileDisk3/dataset/PieProject/task_labeled/0_L.bmp'
    img_file = '/media/pi/ssdMobileDisk/open-mmlab/mmdetection/mycode/right_4.bmp'
    raw_img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(src=raw_img, dsize=(608, 512))
    img_np = np.array([np.float32(img) / 255.]).transpose(0, 3, 1, 2)

    yolo_res = yolo_session.run([yolo_output_name], {yolo_input_name: img_np})[0][0]
    valid_obj_id = yolo_res[:, 4] > yolo_thre
    found = yolo_res[valid_obj_id]  # format: xyxy

    raw_height, raw_width = raw_img.shape[0:2]
    for obj in found:
        px1 = int(raw_width * obj[0] / 608)
        py1 = int(raw_height * obj[1] / 512)
        px2 = int(raw_width * obj[2] / 608)
        py2 = int(raw_height * obj[3] / 512)
        bbox = [px1, py1, px2-px1, py2-py1]
        center, scale = box2cs(box=bbox, input_size=kpt_input_size)
        trans = get_affine_transform(center=center, scale=scale, rot=0, output_size=kpt_input_size)
        img = cv2.warpAffine(raw_img, trans, (kpt_input_size[0], kpt_input_size[1]), flags=cv2.INTER_LINEAR)
        img = img / 255.
        img = img.astype(np.float32)
        input_img = rearrange(img, 'h w c -> 1 c h w')
        output_heatmap = kpt_session.run([kpt_output_name], {kpt_input_name: input_img})[0]
        center = rearrange(center, 'c -> 1 c')
        scale = rearrange(scale, 'c -> 1 c')
        preds, maxvals = keypoints_from_heatmaps(heatmaps=output_heatmap, center=center, scale=scale)
        pts = preds[0]
        print(pts)
        np.savetxt('right_4.txt', pts)
        cv2.rectangle(img=raw_img, pt1=(px1, py1), pt2=(px2, py2), color=(0,0,255), thickness=2)
        for i, p in enumerate(pts):
            cv2.circle(img=raw_img, center=(int(p[0]), int(p[1])), radius=5, thickness=5, color=colors[i])

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', raw_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    yolo_onnx_file = '/media/pi/ssdMobileDisk/open-mmlab/mmdetection/work_dir/line/line.onnx'
    kpt_onnx_file = '/media/pi/ssdMobileDisk/open-mmlab/mmpose/work_dir/hrnet/line/kpt.onnx'

    yolo_session = onnxruntime.InferenceSession(yolo_onnx_file)
    kpt_session = onnxruntime.InferenceSession(kpt_onnx_file)
    yolo_input_name = yolo_session.get_inputs()[0].name
    yolo_output_name = yolo_session.get_outputs()[0].name
    kpt_input_name = kpt_session.get_inputs()[0].name
    kpt_output_name = kpt_session.get_outputs()[0].name

    cam_left = MVCam(acSn='058090410009')
    cam_left.start()
    cam_left.setAeState(False)
    cam_left.setAnalogGain(64)
    cam_left.setExposureTime(10000)
    cam_left.setContrast(100)
    cam_left.setGamma(100)

    cam_right = MVCam(acSn='058112110139')
    cam_right.start()
    cam_right.setAeState(False)
    cam_right.setAnalogGain(64)
    cam_right.setExposureTime(10000)
    cam_right.setContrast(100)
    cam_right.setGamma(100)

    cv2.namedWindow('img_left', cv2.WINDOW_NORMAL)
    cv2.namedWindow('img_right', cv2.WINDOW_NORMAL)
    stop = False
    count = 0
    while not stop:
        raw_img_left = cam_left.readImage()
        raw_img_left_save = copy.deepcopy(raw_img_left)
        raw_img_left = cv2.cvtColor(src=raw_img_left, code=cv2.COLOR_GRAY2BGR)
        img_left = cv2.resize(src=raw_img_left, dsize=(608, 512))
        img_left_np = np.array([np.float32(img_left) / 255.]).transpose(0, 3, 1, 2)

        raw_img_right = cam_right.readImage()
        raw_img_right_save = copy.deepcopy(raw_img_right)
        raw_img_right = cv2.cvtColor(src=raw_img_right, code=cv2.COLOR_GRAY2BGR)
        img_right = cv2.resize(src=raw_img_right, dsize=(608, 512))
        img_right_np = np.array([np.float32(img_right) / 255.]).transpose(0, 3, 1, 2)

        # yolo_res_left = yolo_session.run([yolo_output_name], {yolo_input_name: img_left_np})[0][0]
        # valid_obj_id_left = yolo_res_left[:, 4] > yolo_thre
        # found_left = yolo_res_left[valid_obj_id_left]  # format: xyxy
        #
        # yolo_res_right = yolo_session.run([yolo_output_name], {yolo_input_name: img_right_np})[0][0]
        # valid_obj_id_right = yolo_res_right[:, 4] > yolo_thre
        # found_right = yolo_res_right[valid_obj_id_right]  # format: xyxy
        #
        # raw_height, raw_width = raw_img_left.shape[0:2]
        # left_found_pts = None
        # right_found_pts = None
        #
        # for obj in found_left:
        #     px1 = int(raw_width * obj[0] / 608)
        #     py1 = int(raw_height * obj[1] / 512)
        #     px2 = int(raw_width * obj[2] / 608)
        #     py2 = int(raw_height * obj[3] / 512)
        #     bbox = [px1, py1, px2-px1, py2-py1]
        #     center, scale = box2cs(box=bbox, input_size=kpt_input_size)
        #     trans = get_affine_transform(center=center, scale=scale, rot=0, output_size=kpt_input_size)
        #     img = cv2.warpAffine(raw_img_left, trans, (kpt_input_size[0], kpt_input_size[1]), flags=cv2.INTER_LINEAR)
        #     img = img / 255.
        #     img = img.astype(np.float32)
        #     input_img = rearrange(img, 'h w c -> 1 c h w')
        #     output_heatmap = kpt_session.run([kpt_output_name], {kpt_input_name: input_img})[0]
        #     center = rearrange(center, 'c -> 1 c')
        #     scale = rearrange(scale, 'c -> 1 c')
        #     preds, maxvals = keypoints_from_heatmaps(heatmaps=output_heatmap, center=center, scale=scale)
        #     pts = preds[0]
        #     left_found_pts = pts
        #
        #     cv2.rectangle(img=raw_img_left, pt1=(px1, py1), pt2=(px2, py2), color=(0,0,255), thickness=2)
        #     for i, p in enumerate(pts):
        #         cv2.circle(img=raw_img_left, center=(int(p[0]), int(p[1])), radius=5, thickness=5, color=colors[i])
        #
        # for obj in found_right:
        #     px1 = int(raw_width * obj[0] / 608)
        #     py1 = int(raw_height * obj[1] / 512)
        #     px2 = int(raw_width * obj[2] / 608)
        #     py2 = int(raw_height * obj[3] / 512)
        #     bbox = [px1, py1, px2-px1, py2-py1]
        #     center, scale = box2cs(box=bbox, input_size=kpt_input_size)
        #     trans = get_affine_transform(center=center, scale=scale, rot=0, output_size=kpt_input_size)
        #     img = cv2.warpAffine(raw_img_right, trans, (kpt_input_size[0], kpt_input_size[1]), flags=cv2.INTER_LINEAR)
        #     img = img / 255.
        #     img = img.astype(np.float32)
        #     input_img = rearrange(img, 'h w c -> 1 c h w')
        #     output_heatmap = kpt_session.run([kpt_output_name], {kpt_input_name: input_img})[0]
        #     center = rearrange(center, 'c -> 1 c')
        #     scale = rearrange(scale, 'c -> 1 c')
        #     preds, maxvals = keypoints_from_heatmaps(heatmaps=output_heatmap, center=center, scale=scale)
        #     pts = preds[0]
        #     right_found_pts = pts
        #
        #     cv2.rectangle(img=raw_img_right, pt1=(px1, py1), pt2=(px2, py2), color=(0,0,255), thickness=2)
        #     for i, p in enumerate(pts):
        #         cv2.circle(img=raw_img_right, center=(int(p[0]), int(p[1])), radius=5, thickness=5, color=colors[i])

        cv2.imshow('img_left', raw_img_left)
        cv2.imshow('img_right', raw_img_right)
        key = cv2.waitKey(50)
        if key == 113:
            stop = True
        elif key == 116:
            cv2.imwrite(f'left_cur.bmp', raw_img_left_save)
            cv2.imwrite(f'right_cur.bmp', raw_img_right_save)
            # np.savetxt(f'left_cur.txt', left_found_pts)
            # np.savetxt(f'right_cur.txt', right_found_pts)
            count += 1

    cv2.destroyAllWindows()