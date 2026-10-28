"""Diagnostic: compare per-epoch filtering vs continuous filtering.

Usage: python scripts/diagnose_filter_edges.py --config config/phase1_config.yaml --subject S001
"""
import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import sys

# ensure project root is on sys.path so `src` imports work when running script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_subject_raw
from src.data.preprocessing import Preprocessor
from src.data.events import events_from_raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--subject", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pre = Preprocessor(cfg)

    raw, run_files = load_subject_raw(args.subject, preload=True)

    # follow same early steps as process_subject: bandpass + ICA
    raw_bp = pre.bandpass(raw.copy())
    raw_clean = pre.run_ica(raw_bp)

    events, event_id = events_from_raw(raw_clean)
    if len(events) == 0:
        raise RuntimeError("No events found")

    e_cfg = cfg.get("epoch", {})
    l = cfg.get("reject", {}).get("low", 8.0)
    h = cfg.get("reject", {}).get("high", 30.0)

    # epochs from cleaned raw (no 8-30Hz yet)
    epochs = pre.epoch(raw_clean, events, event_id)

    # per-epoch filtered (this mirrors reject_by_band behavior)
    epochs_perepoch_f = epochs.copy().filter(l_freq=l, h_freq=h, verbose=False)

    # continuous filtered raw then epoch
    raw_cont_f = raw_clean.copy().filter(l_freq=l, h_freq=h, verbose=False)
    epochs_cont_f = pre.epoch(raw_cont_f, events, event_id)

    sfreq = int(epochs.info['sfreq'])
    n_times = epochs.times.shape[0]
    half_sec = int(0.5 * sfreq)

    data_per = epochs_perepoch_f.get_data()
    data_cont = epochs_cont_f.get_data()

    # identify channel with peak amplitude in epoch 0 (per-epoch filtered)
    ep0 = data_per[0]  # channels x times
    ch_idx = int(np.argmax(np.max(np.abs(ep0), axis=1)))

    print(f"Run files: {run_files}")
    print(f"Using band {l}-{h} Hz; epoch tmin/tmax: {e_cfg.get('tmin')} / {e_cfg.get('tmax')}")
    print(f"Sample rate: {sfreq} Hz, samples per epoch: {n_times}")
    print(f"Channel with peak in epoch0 (per-epoch filter): index {ch_idx}, name {epochs.ch_names[ch_idx]}")

    def seg_str(arr, start, end):
        return np.round(arr[start:end], 6).tolist()

    # segments: start 0.5s, middle centered 0.5s, last 0.5s
    start0 = 0
    end0 = min(half_sec, n_times)
    mid_center = n_times // 2
    mid_start = max(0, mid_center - half_sec // 2)
    mid_end = min(n_times, mid_center + half_sec // 2)
    last_start = max(0, n_times - half_sec)
    last_end = n_times

    print('\nEpoch0 per-epoch-filtered samples (channel {})'.format(epochs.ch_names[ch_idx]))
    ch_ep0_per = ep0[ch_idx]
    print(' start 0.5s:', seg_str(ch_ep0_per, start0, end0))
    print(' middle 0.5s:', seg_str(ch_ep0_per, mid_start, mid_end))
    print(' last 0.5s:', seg_str(ch_ep0_per, last_start, last_end))

    # same for continuous-filtered epoch0
    ep0_cont = data_cont[0][ch_idx]
    print('\nEpoch0 continuous-filtered samples (channel {})'.format(epochs.ch_names[ch_idx]))
    print(' start 0.5s:', seg_str(ep0_cont, start0, end0))
    print(' middle 0.5s:', seg_str(ep0_cont, mid_start, mid_end))
    print(' last 0.5s:', seg_str(ep0_cont, last_start, last_end))

    # print peak_abs_uv for first 5 epochs side-by-side
    print('\nPeak absolute amplitudes (uV) for first 5 epochs:')
    for i in range(min(5, data_per.shape[0])):
        peak_per = float(np.max(np.abs(data_per[i]) * 1e6))
        peak_cont = float(np.max(np.abs(data_cont[i]) * 1e6))
        print(f' epoch {i}: per-epoch peak = {peak_per:.3f} uV; continuous-raw peak = {peak_cont:.3f} uV')

    # ---------- ICA effectiveness check ----------
    print('\nRunning ICA-effectiveness diagnostics...')
    ica_cfg = cfg.get('ica', {})
    n_components = ica_cfg.get('n_components', 20)
    random_state = ica_cfg.get('random_state', 42)
    import mne

    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, verbose=False)
    try:
        ica.fit(raw_bp)
    except Exception as e:
        print('ICA fit failed:', e)
        return

    # Explicit EOG proxy detection using Fp1. and Fp2. channels
    proxies = [ch for ch in ['Fp1.', 'Fp2.'] if ch in raw_bp.ch_names]
    print('EOG proxy channels available for detection:', proxies)
    src = None
    try:
        src = ica.get_sources(raw_bp).get_data()
    except Exception as e:
        print('Could not get ICA sources for correlation analysis:', e)

    detected_by_proxy = {}
    # thresholds to compare (permissive and stricter)
    thresholds = [0.5, 0.2, 0.1]
    proxy_scores = {}
    if src is not None and proxies:
        for proxy in proxies:
            proxy_data = raw_bp.get_data(picks=[proxy]).ravel()
            # compute correlation of each ICA component with proxy channel
            corrs = []
            for comp in range(src.shape[0]):
                try:
                    c = np.corrcoef(src[comp], proxy_data)[0, 1]
                except Exception:
                    c = 0.0
                corrs.append(float(c))
            abs_corrs = np.abs(np.array(corrs))
            proxy_scores[proxy] = abs_corrs
            for thr in thresholds:
                inds = list(np.where(abs_corrs >= thr)[0])
                detected_by_proxy.setdefault(thr, set()).update(inds)

    print('ICA n_components:', n_components)
    if proxy_scores:
        for proxy, scores_arr in proxy_scores.items():
            top_idx = list(np.argsort(-scores_arr)[:5])
            top_scores = [scores_arr[i] for i in top_idx]
            print(f' Top components correlated with {proxy}: {list(zip(top_idx, [float(s) for s in top_scores]))}')
        for thr in thresholds:
            ids = sorted(list(detected_by_proxy.get(thr, [])))
            print(f' Components flagged at threshold {thr}: {ids}')
    else:
        print('No proxy channels found or ICA sources unavailable; no components flagged')

    # Compute variance explained by components
    try:
        if src is None:
            src = ica.get_sources(raw_bp).get_data()
        var_per_comp = np.var(src, axis=1)
        total_var = var_per_comp.sum()
        print(f'Total variance in ICA sources: {total_var:.6e}')
    except Exception as e:
        print('Could not compute variance in ICA sources:', e)

    # Plot topographies and time courses for top candidate components (from combined proxy detection at thr=0.2)
    chosen_thr = 0.2
    chosen_comps = sorted(list(detected_by_proxy.get(chosen_thr, [])))[:3]
    out_dir = os.path.join('outputs', args.subject)
    os.makedirs(out_dir, exist_ok=True)
    if chosen_comps:
        try:
            fig_topo = ica.plot_components(picks=chosen_comps, show=False)
            # fig_topo can be a matplotlib.Figure or array; handle both
            try:
                fig_topo.savefig(os.path.join(out_dir, f'ica_topo_thr{chosen_thr}.png'))
            except Exception:
                # if multiple figures returned, save each
                if isinstance(fig_topo, (list, tuple)):
                    for idx, f in enumerate(fig_topo):
                        f.savefig(os.path.join(out_dir, f'ica_topo_thr{chosen_thr}_{idx}.png'))
        except Exception as e:
            print('Could not plot ICA topographies:', e)

        try:
            fig_src = ica.plot_sources(raw_bp, picks=chosen_comps, show=False)
            try:
                fig_src.savefig(os.path.join(out_dir, f'ica_sources_thr{chosen_thr}.png'))
            except Exception:
                if isinstance(fig_src, (list, tuple)):
                    for idx, f in enumerate(fig_src):
                        f.savefig(os.path.join(out_dir, f'ica_sources_thr{chosen_thr}_{idx}.png'))
        except Exception as e:
            print('Could not plot ICA source timecourses:', e)
    else:
        print(f'No components chosen at threshold {chosen_thr}; skipping plots')

    # ---------- Apply ICA exclusion (chosen components) and re-run epoch + rejection ----------
    if chosen_comps:
        try:
            ica.exclude = chosen_comps
            raw_after_ica = ica.apply(raw_bp.copy())
            epochs_after_ica = pre.epoch(raw_after_ica, events, event_id)
            before = len(epochs_after_ica)
            print(f'After ICA exclusion: epochs before rejection = {before}')
            epochs_after_reject = pre.reject_by_band(epochs_after_ica, diag=False)
            after = len(epochs_after_reject)
            print(f'After ICA exclusion + band rejection: epochs after rejection = {after}')
        except Exception as e:
            print('Could not apply ICA exclusion and recompute epochs/rejection:', e)
    else:
        print('No chosen components to exclude; skipping ICA application step')

    # ---------- Per-channel origin check across epochs (pre-rejection) ----------
    print('\nPer-epoch peak-origin channel counts (pre-rejection):')
    # use epochs before any rejection: epochs (from earlier, this is pre-rejection)
    data_pre = epochs.get_data()  # n_epochs, n_ch, n_times
    n_epochs = data_pre.shape[0]
    from collections import Counter

    culprit_counts = Counter()
    for i in range(n_epochs):
        ep = data_pre[i]
        # channel producing peak amplitude in this epoch
        ch_idx = int(np.argmax(np.max(np.abs(ep), axis=1)))
        culprit_counts[epochs.ch_names[ch_idx]] += 1

    # print sorted counts
    for ch, cnt in culprit_counts.most_common():
        print(f' {ch}: {cnt} / {n_epochs} epochs')


if __name__ == '__main__':
    main()
