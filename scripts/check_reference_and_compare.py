"""Diagnostic: compare referencing effects on per-channel peak stats for a subject.

Usage: python scripts/check_reference_and_compare.py --config config/phase1_config.yaml --subject S001

This script does NOT modify the pipeline. It prints:
 - `raw.info['description']`, `raw.info['projs']`, and `getattr(raw, '_orig_ref', None)`
 - top-10 channels by % epochs exceeding threshold (100 uV) for original raw and after applying average reference
 - summary distribution (min, 25/50/75/95 percentile, max) of per-epoch/channel peak amplitudes (uV) for both cases
"""
import argparse
import yaml
import numpy as np
import os
import sys
# allow running script from repo root so `src` package is importable
sys.path.insert(0, os.path.abspath('.'))
from src.data.loader import load_subject_raw
from src.data.events import events_from_raw
from src.data.preprocessing import Preprocessor


def percentiles(arr, ps=(25, 50, 75, 95)):
    return [np.percentile(arr, p) for p in ps]


def compute_stats(raw, cfg, subject_id, thresh_uv=100.0):
    # raw is an MNE Raw (preload True)
    events, event_id = events_from_raw(raw)
    if len(events) == 0:
        raise RuntimeError('No events found')
    pre = Preprocessor(cfg)
    # do not bandpass or ICA here; epoch directly as requested
    epochs = pre.epoch(raw, events, event_id)
    data = epochs.get_data()  # n_epochs, n_ch, n_times (Volts)
    n_epochs = data.shape[0]
    ch_names = epochs.ch_names
    # per-channel percent exceeding threshold
    over_counts = []
    peaks = []
    for ch_i, ch in enumerate(ch_names):
        exceeded = np.any(np.abs(data[:, ch_i, :]) * 1e6 > thresh_uv, axis=1)
        over_counts.append((ch, int(exceeded.sum())))
        # per-epoch peaks for this channel
        peaks.extend(np.max(np.abs(data[:, ch_i, :]) * 1e6, axis=1).tolist())
    over_pct = [(ch, cnt / max(1, n_epochs) * 100.0) for ch, cnt in over_counts]
    peaks = np.array(peaks)
    stats = {
        'n_epochs': n_epochs,
        'ch_names': ch_names,
        'over_counts': over_counts,
        'over_pct': over_pct,
        'peaks_uV': peaks,
    }
    return stats


def print_top10(over_pct, n_epochs):
    from operator import itemgetter
    top10 = sorted(over_pct, key=itemgetter(1), reverse=True)[:10]
    for ch, pct in top10:
        cnt = dict([(c, cnt) for c, cnt in [(k, v) for k, v in [(c, int(round(pct/100.0 * n_epochs)))]]])
    # print properly
    for ch, pct in top10:
        # find count from over_pct list
        cnt = dict(over_pct)[ch] if isinstance(over_pct, list) else None
        # find raw count using n_epochs
        est_cnt = int(round(pct / 100.0 * n_epochs))
        print(f" {ch}: {pct:.1f}% ({est_cnt}/{n_epochs})")


def print_peak_summary(peaks_uV):
    ps = percentiles(peaks_uV, (25, 50, 75, 95))
    print(f" min={peaks_uV.min():.3f} uV")
    print(f" 25%={ps[0]:.3f} uV 50%={ps[1]:.3f} uV 75%={ps[2]:.3f} uV 95%={ps[3]:.3f} uV")
    print(f" max={peaks_uV.max():.3f} uV")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--subject', required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    subject = args.subject

    raw, files = load_subject_raw(subject, preload=True)
    print('Run files:', files)

    # Print reference info available
    print('\nReference info from raw:')
    desc = raw.info.get('description', None)
    print(' raw.info["description"]:', desc)
    print(' raw.info["projs"]:', raw.info.get('projs', None))
    print(' raw._orig_ref (attr):', getattr(raw, '_orig_ref', None))
    print(' raw.info["bads"]:', raw.info.get('bads', None))

    # baseline stats without changing reference
    print('\n--- Original reference: computing stats (no filtering/ICA) ---')
    stats_orig = compute_stats(raw.copy(), cfg, subject)
    print(f"Total epochs = {stats_orig['n_epochs']}")
    print('\nPer-channel over-threshold percentage (top 10):')
    from operator import itemgetter
    top10_orig = sorted(stats_orig['over_pct'], key=itemgetter(1), reverse=True)[:10]
    for ch, pct in top10_orig:
        cnt = dict(stats_orig['over_counts'])[ch]
        print(f' {ch}: {pct:.1f}% ({cnt}/{stats_orig["n_epochs"]})')
    print('\nPeak amplitude distribution across channels/epochs (uV):')
    print_peak_summary(stats_orig['peaks_uV'])

    # Now apply average reference on a copy, before filtering/epoching
    print('\n--- After applying average reference: computing stats ---')
    raw_avg = raw.copy()
    try:
        raw_avg.set_eeg_reference('average', projection=False)
    except Exception as e:
        print(' set_eeg_reference failed:', e)
    stats_avg = compute_stats(raw_avg, cfg, subject)
    print(f"Total epochs = {stats_avg['n_epochs']}")
    print('\nPer-channel over-threshold percentage (top 10):')
    top10_avg = sorted(stats_avg['over_pct'], key=itemgetter(1), reverse=True)[:10]
    for ch, pct in top10_avg:
        cnt = dict(stats_avg['over_counts'])[ch]
        print(f' {ch}: {pct:.1f}% ({cnt}/{stats_avg["n_epochs"]})')
    print('\nPeak amplitude distribution across channels/epochs (uV):')
    print_peak_summary(stats_avg['peaks_uV'])

    # Also print a brief side-by-side comparison for top-10 channels
    print('\n=== Side-by-side top-10 comparison (orig vs avg-ref) ===')
    for i in range(10):
        ch_o, p_o = top10_orig[i]
        ch_a, p_a = top10_avg[i]
        print(f"{i+1:2d}. Orig: {ch_o:6s} {p_o:5.1f}%   | AvgRef: {ch_a:6s} {p_a:5.1f}%")


if __name__ == '__main__':
    main()
