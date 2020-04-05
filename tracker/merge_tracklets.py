#!/usr/bin/env python3

import numpy as np
from numpy import array as arr
from numpy.linalg import norm

MAX_N_PROPOSALS = 20


def normalization_distance(tracklets_info):
    distances = []
    for i, trk in enumerate(tracklets_info):
        for j in range(i + 1, len(tracklets_info)):
            distances.append(norm(trk['avg_reid_score'] - tracklets_info[j]['avg_reid_score']))

    return max(distances)


def calculate_merge_score(tracklet, merge_props, norm_distance, time_weighting=False, alpha=0.6):
    if time_weighting:  # Average ReID L2 distance with time weighting
        score = arr(
            [norm(tracklet['avg_reid_score'] - prop['avg_reid_score']) + (alpha * abs(tracklet['start'] - prop['end']))
             for prop in merge_props])
    else:  # Average ReID L2 distance
        score = arr([norm(tracklet['avg_reid_score'] - prop['avg_reid_score']) / norm_distance for prop in
                     merge_props])

        if any(score > 1) or any(score < 0):
            print("Upss I did it again...")
    return [prop for _, prop in sorted(zip(score, merge_props), key=lambda t: t[0])]


def merge_tracklet(tracklets, tracklet_id, sequence, isVisited):
    sequence.append(tracklet_id)
    merge_id = tracklets[tracklet_id]['backward_merged_prop']

    if merge_id == -1:
        isVisited[tracklet_id] = True
        return

    isVisited[tracklet_id] = True
    merge_tracklet(tracklets, merge_id, sequence, isVisited)


def generate_refined_clusters(tracklet_seqs, tracklets_info, trajectories, time, norm_distance):
    if len(tracklet_seqs) == 0:
        return

    tracklet_seqs = sorted([sorted(ts) for ts in tracklet_seqs])
    clustered_tracklet_seqs = cluster_tracklet_sequences(tracklet_seqs)

    for clustered_seqs in clustered_tracklet_seqs:
        if clustered_seqs['refined'] == False:
            cluster = clustered_seqs['clustered_sequences']
            trj, cluster = get_best_trajectory_within_cluster(cluster, tracklets_info, time,
                                                              norm_distance=norm_distance)
            trajectories.append(trj)
            refined_cluster = refine_cluster(cluster, trj)
            generate_refined_clusters(refined_cluster, tracklets_info, trajectories, time=time,
                                      norm_distance=norm_distance)
            clustered_seqs['refined'] = True


def refine_cluster(clustered_seqs, sequence):
    for c_seq in clustered_seqs:
        for t_id in sequence:
            if t_id not in c_seq:
                break
            else:
                c_seq.remove(t_id)

    return clustered_seqs


def cluster_tracklet_sequences(tracklet_seqs):
    clustered_tracklet_seqs = []
    for ts in tracklet_seqs:
        clustered = False
        if not clustered_tracklet_seqs:
            clustered_tracklet_seqs.append({'cluster_id': ts[0], 'clustered_sequences': [], 'refined': False})
            clustered_tracklet_seqs[0]['clustered_sequences'].append(ts)
            continue
        for cluster in clustered_tracklet_seqs:
            if cluster['cluster_id'] == ts[0]:
                cluster['clustered_sequences'].append(ts)
                clustered = True
                break
        if not clustered:
            clustered_tracklet_seqs.append({'cluster_id': ts[0], 'clustered_sequences': [], 'refined': False})
            clustered_tracklet_seqs[-1]['clustered_sequences'].append(ts)
    return clustered_tracklet_seqs


def get_best_trajectory_within_cluster(clustered_seqs, tracklets_info, time, norm_distance):
    ends = []
    covers = []
    visual_similarity = []
    for tracklet_seq in clustered_seqs:
        total_gap = 0
        distance = []
        for i, t_id in enumerate(tracklet_seq):
            total_gap += tracklets_info[t_id]['end'] - tracklets_info[t_id]['start']
            distance += [
                1 - (norm(tracklets_info[t_id]['avg_reid_score'] - tracklets_info[t]['avg_reid_score']) / norm_distance)
                for t in tracklet_seq if t > t_id]
        covers.append(total_gap)
        if distance:
            visual_similarity.append(max(distance))
        else:
            visual_similarity.append(0)
        ends.append(tracklets_info[tracklet_seq[-1]]['end'])
    covers = np.array(covers) / time
    visual_similarity = np.array(visual_similarity)
    score = 0.9 * covers + 0.1 * visual_similarity

    if np.isnan(score).any() == True:
        nan_indices = np.where(np.isnan(score) == True)[0]
        for i in nan_indices:
            score[i] = 0

    max_score = np.max(score)
    best_ind = np.where(score == max_score)[0]
    if len(best_ind) > 1:
        ind = best_ind[np.argmax([ends[i] for i in best_ind])]
    else:
        ind = best_ind[0]

    sequence = clustered_seqs[ind]
    clustered_seqs.pop(ind)

    return sequence, clustered_seqs


def merge_compability(ts, end_before_tt_tracklet, norm_distance):
    merge_prop = calculate_merge_score(ts, end_before_tt_tracklet, norm_distance=norm_distance)
    if len(merge_prop) >= 2:
        first_best = merge_prop[0]
        second_best = merge_prop[1]
        if first_best['tracklet_id'] == second_best['backward_merged_prop']:
            potential_pred = second_best
            new_end_before_tt = [tt for tt in end_before_tt_tracklet if
                                 tt['start'] < potential_pred['end'] and tt['backward_merged_prop'] == potential_pred[
                                     'tracklet_id']]
            new_end_before_tt.append(potential_pred)
            merge_compability(ts, new_end_before_tt, norm_distance)
        else:
            ts['backward_merged_prop'] = first_best['tracklet_id']
    else:
        ts['backward_merged_prop'] = merge_prop[0]['tracklet_id']


# merge tracklets using ReID
def merge_tracklets_ReID(tracklets, tracklets_info, no_merge=False):
    tracklet_seqs = []
    trajectories = []

    trajectories_timeline = []
    trajectories_id = []

    if len(tracklets) > 0:

        trajectories_score = []
        trajectories_time = []
        timeline = np.arange(len(tracklets[0]))

        if no_merge is True:
            trajectories = [[i] for i in range(0, len(tracklets))]
        else:
            sorted_tracklets_info = sorted(tracklets_info, key=lambda i: (i['start'], i['end']))
            norm_distance = normalization_distance(sorted_tracklets_info)
            for t in timeline:
                start_tt_tracklet = list(filter(lambda sort_track: sort_track['start'] == t, sorted_tracklets_info))
                end_before_tt_tracklet = list(filter(lambda sort_track: sort_track['end'] < t, sorted_tracklets_info))
                if start_tt_tracklet and end_before_tt_tracklet:
                    for ts in start_tt_tracklet:
                        merge_compability(ts, end_before_tt_tracklet, norm_distance)

            isVisited = np.full(len(tracklets_info), False, dtype=bool)

            for i, t in enumerate(tracklets_info[::-1]):
                if not isVisited[t['tracklet_id']]:
                    sequence = []
                    merge_tracklet(tracklets_info, t['tracklet_id'], sequence, isVisited)
                    tracklet_seqs.append(sequence)

            if len(tracklet_seqs) > 0:
                generate_refined_clusters(tracklet_seqs, tracklets_info, trajectories, time=timeline.shape[0],
                                          norm_distance=norm_distance)
            else:
                print('Tracklets could not been merged...')

        for i, trj in enumerate(trajectories):
            seq_timeline = np.negative(np.ones_like(timeline, dtype=int))
            time = 0
            score = []
            area = []
            for t_id in trj:
                ss = tracklets_info[t_id]['start']
                ee = tracklets_info[t_id]['end'] + 1
                seq_timeline[ss:ee] = tracklets[t_id][ss:ee]
                time += ee - ss
                score.append(tracklets_info[t_id]['avg_score'] * time)
                area.append(tracklets_info[t_id]['avg_area'])
            trajectories_timeline.append(seq_timeline)
            trajectories_id.append(trajectories[i][0])
            trajectories_time.append(time)
            trajectories_score.append(np.mean(score))

        let_or_keep = [time * score for time, score in zip(trajectories_time, trajectories_score)]
        trajectories_timeline = arr([trajectory for _, trajectory in
                                     sorted(zip(trajectories_score, trajectories_timeline), key=lambda t: t[0],
                                            reverse=True)])[:MAX_N_PROPOSALS]
        trajectories_id = arr(
            [id for _, id in sorted(zip(trajectories_score, trajectories_id), key=lambda t: t[0], reverse=True)])[
                          :MAX_N_PROPOSALS]
    else:
        print('Tracklets list is empty...')

    return trajectories_timeline, trajectories_id
