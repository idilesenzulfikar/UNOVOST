#!/usr/bin/env python3

import argparse
import yaml
import numpy as np
import os
import glob
from numpy import array as arr
import time

from tracker.info_proposals import get_proposals_info, get_tracklets_info
from tracker.generate_proposals import generate_tracklet_props
from tracker.generate_tracklets import resolve_tracklet_props
from tracker.merge_tracklets import merge_tracklets_ReID

from visualize import save_with_pascal_colormap

MAX_N_PROPOSALS = 20

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--proposal_dir', type=str, required=True,
                        help='Path to the directory containing json files with proposals, optical flow warping, reid vectors')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the directory to save results')
    parser.add_argument('--config', type=str, default='../configs/unovost.yaml',
                        help='Configuration file')

    tickle = time.time()

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set = config['split']
    overlap_threshold = config['threshold']['overlap']
    opt_flow_threshold = config['threshold']['opt_flow']
    optim_algorithm = config['optimization']
    use_iom = config['use_iom']
    no_merge = config['no_merge']

    proposal_dir = os.path.join(args.proposal_dir, set)
    output_dir = os.path.join(args.output_dir, set)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(proposal_dir,'*')))]

    for name in names:

        print('PROCESSING VIDEO ', name)

        all_props, img_size = get_proposals_info(prop_dir=proposal_dir, name=name,
                                                 image_dir=None, arch=None)

        all_tracklet_props, frames_no_proposals, all_wrap_scores = generate_tracklet_props(all_props,
                                                                                           overlap_threshold=overlap_threshold)

        if frames_no_proposals:
            print("Warning: There are some frames with no proposals !!!")

        tracklets = resolve_tracklet_props(all_tracklet_props, all_props, all_wrap_scores, threshold=opt_flow_threshold,
                                           optimization_algorithm=optim_algorithm, use_iom=use_iom)

        tracklets_info = get_tracklets_info(tracklets, all_tracklet_props)

        trj, trj_id = merge_tracklets_ReID(tracklets, tracklets_info, no_merge=no_merge)
        trajectories = arr(trj)

        assert trajectories.shape[0] <= MAX_N_PROPOSALS, 'Max 20 object results is allowed in evalution'

        timeline = trajectories.shape[1]

        labels_prop = np.arange(1, 10000)
        labels_tracklets = np.arange(1, len(tracklets_info) + 1)
        labels = np.arange(1, len(tracklets_info) + 1)
        tracklets = arr(tracklets)

        if not os.path.exists(os.path.join(output_dir, 'results', name)):
            os.makedirs(os.path.join(output_dir, 'results', name))

        for t, props in enumerate(all_props):

            ###Final Output ----> PRINT
            props_to_use = trajectories[:, t]
            props_to_use_ind = np.where(trajectories[:, t] != -1)[0].tolist()
            png = np.zeros(img_size)

            if len(props_to_use_ind) != 0:
                for i in props_to_use_ind:
                    png[props["mask"][props_to_use[i]].astype("bool")] = labels[i]

            if not os.path.exists(os.path.join(output_dir, 'results', name)):
                os.makedirs(os.path.join(output_dir, 'results', name))

            save_with_pascal_colormap(os.path.join(output_dir, 'results', name + '/' + str(t).zfill(5) + '.png'), png)

        print('DONE')

    print(time.time() - tickle)
