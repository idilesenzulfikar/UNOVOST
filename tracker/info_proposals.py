#!/usr/bin/env python3

import os
import numpy as np
import json
from numpy import array as arr
from pycocotools.mask import encode, area, decode, merge
from numpy.linalg import norm
from copy import deepcopy as copy
import glob
from PIL import Image

from .alternative_ReID import extract_features


def iom(dt, gt):
    area1 = area([dt])
    area2 = area([gt])
    min_area = np.min([area1, area2])
    intersection_area = area(merge([dt, gt], intersect=True))
    return intersection_area / min_area


def get_props_with_iom(props, props_ind):
    props_iom_ratio = []
    if len(props['id']) > 1:
        for i in props_ind:
            ratio = area([merge(
                [merge([props['seg'][i], props['seg'][j]], intersect=True) for j in props_ind if i != j],
                intersect=False)])[0] / area([props['seg'][i]])[0]
            props_iom_ratio.append(ratio)
    else:
        props_iom_ratio.append(0)

    return props_iom_ratio


# get proposals info from json files
def get_proposals_info(prop_dir, name, image_dir=None, arch='vgg16'):
    files = sorted(glob.glob(os.path.join(prop_dir, name, '*.json')))
    if image_dir:
        images = sorted(glob.glob(os.path.join(image_dir, name, '*.jpg')))
        if len(files) != len(images):
            raise ValueError(
                "There are %d proposal files %d images, they do not match.".format(len(files), len(images)))
    all_props = []
    img_size = ()
    for i, file in enumerate(files):
        with open(file, "r") as f:
            proposals = json.load(f)

        curr_prop = dict()
        curr_prop['seg'] = [prop["segmentation"] for prop in proposals]
        curr_prop['fwd'] = [prop["forward_segmentation"] if "forward_segmentation" in prop.keys() else None for prop in
                            proposals]
        if image_dir:
            img = Image.open(images[i])
            curr_prop['reid'] = extract_features(img, proposals, arch)
        else:
            curr_prop['reid'] = [arr(prop["ReID"]) if "ReID" in prop.keys() else np.inf * np.ones((128)) for prop in
                                 proposals]
        curr_prop['score'] = arr([prop["score"] for prop in proposals])
        curr_prop['id'] = np.arange(0, len(proposals))
        curr_prop['mask'] = [decode(seg) for seg in curr_prop['seg']]

        if len(curr_prop['seg']) > 0 and len(img_size) == 0:
            img_size = curr_prop['mask'][0].shape

        all_props.append(curr_prop)

    return all_props, img_size


def get_tracklets_temporal_info(tracklets):
    tracklets_info = []
    for i, ts in enumerate(tracklets):
        t_start = np.where(ts != -1)[0][0]
        r = np.where(ts == -1)[0]
        if len(r[
                   r > t_start]) > 0:  # if the tracklet sequence runs at the end of time line, then seq_end is null, in that case seq_end is the last index
            t_end = r[r > t_start][0] - 1
        else:
            t_end = len(ts) - 1
        tracklets_info.append({'tracklet_id': i, 'start': t_start, 'end': t_end, 'backward_merged_prop': -1})

    return tracklets_info


def get_tracklets_info(tracklets, all_tracklet_prop):
    tracklets_info = get_tracklets_temporal_info(tracklets)

    for i, ts in enumerate(tracklets):
        t_start = tracklets_info[i]['start']
        t_end = tracklets_info[i]['end']

        t_reid = arr([all_tracklet_prop[t][ind]['reid'] for t in range(t_start, t_end + 1) for ind in
                      range(len(all_tracklet_prop[t])) if all_tracklet_prop[t][ind]['id'] == ts[t]])

        t_avg_reid = np.mean(t_reid, axis=0)
        tracklets_info[i]['avg_reid_score'] = t_avg_reid

        tracklets_info[i]['start_prop_reid'] = t_reid[0]
        tracklets_info[i]['end_prop_reid'] = t_reid[-1]

        seq = range(t_start, t_end + 1)
        tracklets_info[i]['avg_area'] = np.mean(
            [all_tracklet_prop[t][ind]['area'] for t in seq for ind in range(len(all_tracklet_prop[t])) if
             all_tracklet_prop[t][ind]['id'] == ts[t]])

        t_score = [all_tracklet_prop[t][ind]['score'] for t in range(t_start, t_end + 1) for ind in
                   range(len(all_tracklet_prop[t])) if all_tracklet_prop[t][ind]['id'] == ts[t]]
        t_avg_score = np.mean(arr(t_score), axis=0) if len(t_score) != 0 else arr(
            [all_tracklet_prop[t_start][ind]['score'] for ind in range(len(all_tracklet_prop[t_start])) if
             all_tracklet_prop[t_start][ind]['id'] == ts[t_start]])
        tracklets_info[i]['avg_score'] = t_avg_score

    return tracklets_info
