#!/usr/bin/env python3

import numpy as np
from numpy import array as arr
from pycocotools.mask import iou, area


def remove_overlap_props(props, props_ids, threshold=0.2):
    final_props = []
    isOverlap = False
    if len(props_ids) > 0:
        for i in props_ids:
            if len(final_props) == 0:
                final_props.append(i)
                continue
            ratios = [iou([props['seg'][i]], [props['seg'][merged]], arr([0], np.uint8))[:, 0][0] for merged in
                      final_props]
            if ratios:
                if np.max(ratios) >= threshold:
                    isOverlap = True
                    if props['score'][i] > props['score'][final_props[np.argmax(ratios)]]:
                        final_props.pop(np.argmax(ratios))
                        final_props.append(i)
                else:
                    final_props.append(i)
        if isOverlap:
            final_props = remove_overlap_props(props, final_props, threshold)

    return final_props


# generate tracklet proposals removing overlaps with confidence score
def generate_tracklet_props(all_props, overlap_threshold=0.2):
    all_tracklet_props = []  # all possible tracklet proposals in video sequence -> list of dictionaries
    all_wrap_scores = []  # all wrap score matrices in video sequence
    frames_no_proposals = []  # frames that have no proposals available

    def props_info(props, key, props_ids):
        return [props[key][i] for i in props_ids]

    prev_props = remove_overlap_props(all_props[0], all_props[0]['id'], threshold=overlap_threshold)
    prev_next_segs = props_info(all_props[0], 'fwd', prev_props)
    prev_reid_score = props_info(all_props[0], 'reid', prev_props)
    prev_seg = props_info(all_props[0], 'seg', prev_props)
    prev_score = props_info(all_props[0], 'score', prev_props)

    for t, props in enumerate(all_props[1:]):

        final_props = remove_overlap_props(props, props['id'], threshold=overlap_threshold)

        if len(final_props) > 0 and len(prev_next_segs) > 0:
            wrap_scores = arr(
                [iou([props['seg'][fp] for fp in final_props], [prev_next_seg], arr([0], np.uint8))[:, 0] for
                 prev_next_seg in prev_next_segs])
        else:
            wrap_scores = None

        all_wrap_scores.append(wrap_scores)

        if len(props['seg']) == 0:
            frames_no_proposals.append(t + 1)

        all_tracklet_props.append(
            [{'id': pp, 'reid': prev_reid_score[i], 'score': prev_score[i], 'area': area(prev_seg[i])} for i, pp in
             enumerate(prev_props)])

        prev_props = final_props
        prev_next_segs = props_info(props, 'fwd', prev_props)
        prev_reid_score = props_info(props, 'reid', prev_props)
        prev_seg = props_info(props, 'seg', prev_props)
        prev_score = props_info(props, 'score', prev_props)

    all_tracklet_props.append(
        [{'id': pp, 'reid': prev_reid_score[i], 'score': prev_score[i], 'area': area(prev_seg[i])} for i, pp in
         enumerate(prev_props)])  # adding the proposals in the last frame

    return all_tracklet_props, frames_no_proposals, all_wrap_scores
