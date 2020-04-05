#!/usr/bin/env python3

import numpy as np
from numpy import array as arr
from scipy.optimize import linear_sum_assignment

from .info_proposals import get_props_with_iom, get_tracklets_temporal_info


# find the indices in a multi-dimensional array for a number of maximum unique values with a threshold
def max_ind_threshold(mat, num_track, threshold):
    row_ind = []
    col_ind = []
    for n in range(num_track):
        max_val = np.amax(mat)
        if max_val > threshold:  # if the max_value is higher the threshold, you have an assignment to current tracklets in time t
            i, j = np.where(mat == max_val)  # add to max_ind
            if i.size > 1:
                print('Warning : There are more than one same max_val,but always select first values of i and j as the '
                      'indices')
            row_ind.append(i[0])
            col_ind.append(j[0])
            mat[i[0], :] = -1
            mat[:, j[0]] = -1
        else:  # In case max_val is under the threshold, the rest of the values are also under the threshold, so call the return
            return row_ind, col_ind
    return row_ind, col_ind


def match_hungarian(dist_matrix, thresh):
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    row_ind_thr = []
    col_ind_thr = []
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r][c] < thresh:
            row_ind_thr.append(r)
            col_ind_thr.append(c)
    return row_ind_thr, col_ind_thr


# generate tracklets using optical flow warping between proposals in successive frames
def resolve_tracklet_props(all_tracklet_props, all_props, all_wrap_scores, threshold=0.05,
                           optimization_algorithm='greedy', use_iom=False):
    new_tracklet = np.negative(np.ones(len(all_tracklet_props), dtype=int))
    tracklets = []
    tracklets_iom_scores = []

    for t, track_props in enumerate(all_tracklet_props[:-1]):  # t -> time, but skip the ending time

        curr_all_props_id = [tp['id'] for tp in track_props]

        if len(track_props) > 0:
            ws = all_wrap_scores[t]

            if ws is not None:
                num_curr_props = ws.shape[0]
                num_next_props = ws.shape[1]

                if num_curr_props < num_next_props:
                    num_assign = num_curr_props
                else:
                    num_assign = num_next_props

                if optimization_algorithm == 'greedy':
                    row_ind, col_ind = max_ind_threshold(ws.copy(), num_assign, threshold)
                elif optimization_algorithm == 'hungarian':
                    row_ind, col_ind = match_hungarian(np.negative(ws), np.negative(threshold))
                else:
                    raise NotImplementedError('Optimization algorithm is not found.')

                curr_props_id = [all_tracklet_props[t][i]['id'] for i in row_ind]
                next_props_id = [all_tracklet_props[t + 1][j]['id'] for j in col_ind]

                curr_props_iom_ratio = get_props_with_iom(all_props[t], curr_props_id)
                next_props_iom_ratio = get_props_with_iom(all_props[t + 1], next_props_id)
            else:
                curr_props_id = curr_all_props_id
                curr_props_iom_ratio = get_props_with_iom(all_props[t], curr_props_id)

                next_props_id = next_props_iom_ratio = []
        else:
            curr_props_id = next_props_id = curr_props_iom_ratio = next_props_iom_ratio = []

        if len(tracklets) == 0:  # the first tracklet is generated
            if len(curr_props_id) == 0:
                continue
            else:
                for i, curr_id in enumerate(curr_props_id):
                    nw = new_tracklet.copy()
                    nw[t] = curr_id
                    tracklets_iom_scores.append(curr_props_iom_ratio[i])
                    tracklets.append(nw)

        if tracklets:
            active_tracklets = arr(tracklets)[:, t]

        if len(curr_props_id) > 0:
            if len(next_props_id) > 0:
                for i, (curr_id, next_id) in enumerate(zip(curr_props_id, next_props_id)):
                    if curr_id in active_tracklets:  # update the active tracklet
                        s = np.where(arr(tracklets)[:, t] == curr_id)[0][0]
                        tracklets[s][t + 1] = next_id
                        tracklets_iom_scores[s] += next_props_iom_ratio[i]
                    else:  # new tracklet
                        nw = new_tracklet.copy()
                        nw[t] = curr_id
                        nw[t + 1] = next_id
                        tracklets.append(nw)
                        tracklets_iom_scores.append(curr_props_iom_ratio[i] + next_props_iom_ratio[i])

        if len(curr_all_props_id) > 0:
            active_tracklets = arr(tracklets)[:, t]
            ratio = get_props_with_iom(all_props[t], curr_all_props_id)
            for i, curr_id in enumerate(curr_all_props_id):
                if curr_id not in active_tracklets:
                    nw = new_tracklet.copy()
                    nw[t] = curr_id
                    tracklets.append(nw)
                    tracklets_iom_scores.append(ratio[i])

    # Removing overlaps between proposals w/ iom scoring ----- > I did not use this in the challenge.
    # Since I already wrote that part, why I would remove it :D
    if use_iom:
        tracklets_info = get_tracklets_temporal_info(tracklets)
        for i, iom_score in reversed(list(enumerate(tracklets_iom_scores))):
            t_start = tracklets_info[i]['start']
            t_end = tracklets_info[i]['end']
            tracklets_iom_scores[i] = iom_score / (t_end - t_start + 1)
            if (iom_score / (t_end - t_start + 1)) < 0 or (iom_score / (t_end - t_start + 1)) > 1:
                print('YES , in time ' + str(i))
            if (iom_score / (t_end - t_start + 1)) > 0.7:
                tracklets.pop(i)

    return tracklets
