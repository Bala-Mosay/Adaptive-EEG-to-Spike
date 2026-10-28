"""Inspect AutoReject per-epoch repair/drop behavior for a subject.

Usage: python scripts/print_autoreject_epoch_log.py --config config/phase1_config.yaml --subject S001
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import yaml
import numpy as np
from src.data.loader import load_subject_raw
from src.data.events import events_from_raw
from src.data.preprocessing import Preprocessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--subject', required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    subject = args.subject

    raw, files = load_subject_raw(subject, preload=True)
    pre = Preprocessor(cfg)
    # apply bandpass and average ref and ICA like pipeline
    raw = pre.bandpass(raw)
    try:
        raw.set_eeg_reference('average', projection=False)
    except Exception:
        pass
    raw = pre.run_ica(raw)

    events, event_id = events_from_raw(raw)
    epochs = pre.epoch(raw, events, event_id)
    n_epochs = len(epochs)
    print('Total epochs before autoreject:', n_epochs)

    try:
        from autoreject import AutoReject
    except Exception as e:
        print('autoreject not available:', e)
        return

    ar = AutoReject(n_jobs=1, verbose=False)
    print('Fitting AutoReject...')
    ar.fit(epochs)
    print('Fitted AutoReject')

    # inspect AutoReject attributes for available logs
    print('\nAutoReject attributes containing keywords:')
    attrs = dir(ar)
    keywords = ('thresh', 'consensus', 'repair', 'reject', 'log', 'bad')
    found = [a for a in attrs if any(k in a.lower() for k in keywords)]
    print(found)

    # get the authoritative reject log from AutoReject
    try:
        reject_log = ar.get_reject_log(epochs)
        labels_arr = reject_log.labels  # shape (n_epochs, n_channels)
    except Exception as e:
        print('Could not get reject_log from AutoReject:', e)
        labels_arr = None

    # transform to get cleaned epochs (may drop some)
    try:
        epochs_clean = ar.transform(epochs)
    except Exception:
        epochs_clean = ar.fit_transform(epochs)

    # determine dropped epochs using the cleaned selection (transform may drop)
    orig_idx = np.arange(n_epochs)
    try:
        kept = np.array(epochs_clean.selection)
    except Exception:
        kept = np.arange(len(epochs_clean))
    kept = kept[kept < n_epochs]
    dropped = np.setdiff1d(orig_idx, kept)
    print('Dropped epochs (count):', len(dropped))

    # If we have the reject_log.labels, use it to count interpolations per epoch
    if labels_arr is not None:
        # label==1 indicates interpolated channel; label==2 indicates bad/dropped channel
        interp_counts = np.sum(labels_arr == 1, axis=1)
        bad_counts = np.sum(labels_arr == 2, axis=1)
        # for kept epochs, report interpolated counts
        interp_kept = interp_counts[kept]
        bad_kept = bad_counts[kept]
        n_kept = len(kept)
        n_no_interp = int(np.sum(interp_kept == 0))
        n_with_interp = int(np.sum(interp_kept > 0))
        print('Kept epochs (count):', n_kept)
        print('Kept epochs with NO interpolated channels:', n_no_interp)
        print('Kept epochs with >=1 interpolated channel:', n_with_interp)
        print('Interpolated channels per-kept-epoch distribution (percentiles):')
        if n_with_interp > 0:
            print(' min', int(interp_kept.min()), '25%', float(np.percentile(interp_kept,25)), '50%', float(np.percentile(interp_kept,50)), '75%', float(np.percentile(interp_kept,75)), 'max', int(interp_kept.max()))
        else:
            print(' none')
        unique, counts = np.unique(interp_kept, return_counts=True)
        print('Histogram of interpolated channel counts (channels: epochs):')
        for u, c in zip(unique, counts):
            print(f' {u}: {c}')
        # also print unique channels ever interpolated
        chans_interpolated = np.where(np.any(labels_arr == 1, axis=0))[0]
        print('Unique channel indices interpolated across all epochs (count):', len(chans_interpolated))
    else:
        print('No reject_log.labels available; cannot report authoritative interpolation counts')

if __name__ == '__main__':
    main()
