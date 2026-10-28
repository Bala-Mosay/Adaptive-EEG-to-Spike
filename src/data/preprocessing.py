from typing import Tuple, List
import numpy as np
import mne


class Preprocessor:
    """Stage 1 signal conditioning pipeline.

    This class provides leakage-safe preprocessing building blocks.
    """

    def __init__(self, config: dict):
        self.config = config

    def bandpass(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        params = self.config.get("bandpass", {})
        raw.filter(l_freq=params.get("l_freq", 0.5), h_freq=params.get("h_freq", 40.0),
                   method=params.get("method", "iir"), iir_params=params.get("iir_params", {}),
                   verbose=False)
        return raw

    def run_ica(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        ica_cfg = self.config.get("ica", {})
        n_components = ica_cfg.get("n_components", 20)
        random_state = ica_cfg.get("random_state", 42)
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, verbose=False)
        ica.fit(raw)

        frontal = [ch for ch in raw.ch_names if ch.upper() in ("FP1", "FP2", "FPZ")]
        picks = []
        if frontal:
            try:
                picks, _ = ica.find_bads_eog(raw, ch_name=frontal[0])
            except Exception:
                picks = []

        if not picks:
            picks = []

        if picks:
            ica.exclude = picks
            raw = ica.apply(raw)
        return raw

    def reject_with_autoreject(self, epochs: mne.Epochs, diag: bool = False) -> mne.Epochs:
        """Use `autoreject.AutoReject` to repair/drop bad epochs and interpolate bad channels per epoch."""
        try:
            from autoreject import AutoReject
        except Exception as e:
            raise RuntimeError("autoreject is required for reject_with_autoreject: install autoreject") from e

        # fit autoreject on epochs (expects MNE Epochs)
        ar_cfg = self.config.get("autoreject", {})
        n_jobs = ar_cfg.get("n_jobs", 1)
        verbose = ar_cfg.get("verbose", False)
        ar = AutoReject(n_jobs=n_jobs, verbose=verbose)
        # AutoReject may raise for unexpected inputs; pass through
        ar.fit(epochs)
        if diag:
            print("Diagnostic: AutoReject computed per-channel thresholds")
            try:
                print(" AutoReject thresholds (first 10):", getattr(ar, 'thresh_', None)[:10])
            except Exception:
                pass
        # transform returns repaired epochs (with some epochs possibly dropped)
        try:
            epochs_clean = ar.transform(epochs)
        except Exception:
            # older API: fit_transform
            epochs_clean = ar.fit_transform(epochs)
        return epochs_clean

    def epoch(self, raw: mne.io.BaseRaw, events: np.ndarray, event_id: dict) -> mne.Epochs:
        e_cfg = self.config.get("epoch", {})
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=e_cfg.get("tmin", 0.0),
                            tmax=e_cfg.get("tmax", 4.0), baseline=e_cfg.get("baseline", None), preload=True, verbose=False)
        return epochs

    def reject_by_band(self, epochs: mne.Epochs, diag: bool = False) -> mne.Epochs:
        r = self.config.get("reject", {})
        l, h = r.get("low", 8.0), r.get("high", 30.0)
        thresh_uv = r.get("threshold_uv", 100.0)
        epochs_f = epochs.copy().filter(l_freq=l, h_freq=h, verbose=False)
        bad_idx = []
        for i, ep in enumerate(epochs_f.get_data()):
            # Reject if any sample exceeds the absolute threshold (±threshold_uv),
            # matching the project spec (peak amplitude), not peak-to-peak.
            # ep is in Volts (MNE uses Volts); convert to microvolts for threshold compare
            ep_uv = ep * 1e6
            peak_uv = np.max(np.abs(ep_uv))
            # diagnostic prints for first few epochs
            if diag and i < 5:
                # compute peak-to-peak as well for reference
                p2p_uv = np.ptp(ep_uv)
                print(f"Diagnostic(epoch={i}): peak_abs_uv={peak_uv:.3f} uV, peak_to_peak_uv={p2p_uv:.3f} uV, thresh_uv={thresh_uv} uV")
                print(f"Diagnostic(epoch={i}): ep dtype={ep.dtype}, ep min={ep.min():.4e} V, ep max={ep.max():.4e} V")
                print("Diagnostic: data units at comparison = Volts (converted to microvolts via *1e6 for comparison)")
            if peak_uv > thresh_uv:
                bad_idx.append(i)
        if bad_idx:
            epochs.drop(bad_idx)
        return epochs

    def process_subject(self, raw: mne.io.BaseRaw, output_dir: str = None, mapping: dict = None, diag: bool = False, run_files: List[str] = None, events: np.ndarray = None, event_id: dict = None) -> tuple:
        """Run Stage 1 end-to-end for a single subject.

        Steps: bandpass -> ICA -> extract events -> epoch -> reject -> baseline
        correction (via `baseline` param) -> return numpy epochs and metadata.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Concatenated raw recording for the subject.
        output_dir : str, optional
            If provided, saves `epochs.npz` and `metadata.csv` there.
        mapping : dict, optional
            Legacy global event-code mapping (only used when `events` is not
            provided). Prefer run-aware events for EEGMMIDB (see
            `src/data/events.py`).
        diag : bool
            Print diagnostic information.
        run_files : List[str], optional
            Run file paths, for diagnostics only.
        events : np.ndarray, optional
            Precomputed run-aware events on the concatenated timeline
            (from `load_subject_raw_with_events`). When provided, these are
            used instead of re-extracting events from the concatenated raw,
            which would mislabel task-2 runs.
        event_id : dict, optional
            Class-label -> code mapping accompanying `events`.

        Returns:
        - X: np.ndarray, shape (n_trials, n_channels, n_timepoints)
        - labels: List[str] label per trial
        - metadata: List[dict] with keys `subject_id`, `label`, `epoch_index`
        """
        from .events import events_from_raw, map_event_id_to_labels
        import os
        # bandpass
        raw = self.bandpass(raw)

        # set average EEG reference (applied after bandpass, before ICA)
        try:
            raw.set_eeg_reference('average', projection=False)
        except Exception:
            pass

        # ICA
        raw = self.run_ica(raw)

        # normalize channel names (strip trailing dots) and set a standard montage
        try:
            raw.rename_channels(lambda x: x.strip('.'))
            # set montage if possible; ignore missing channels
            try:
                raw.set_montage('standard_1020', on_missing='ignore')
            except Exception:
                pass
        except Exception:
            pass

        # extract events (use precomputed run-aware events when provided;
        # re-extracting from the concatenated raw would mislabel task-2 runs)
        if events is not None and event_id is not None:
            if diag:
                print("Diagnostic: using precomputed run-aware events")
                print("Diagnostic: event_id mapping:", event_id)
                print("Diagnostic: event count:", len(events))
        else:
            events, event_id = events_from_raw(raw)
            if diag:
                print("Diagnostic: run files:", run_files)
                # print annotation/event counts
                try:
                    ann = raw.annotations
                    print("Diagnostic: annotations count:", len(ann))
                except Exception:
                    pass
                # event_id mapping
                print("Diagnostic: event_id mapping:", event_id)
        if len(events) == 0:
            raise RuntimeError("No events found in raw object")

        # epoch
        epochs = self.epoch(raw, events, event_id)

        # Per-channel over-threshold diagnostic (percentage of epochs where channel exceeds threshold)
        r = self.config.get("reject", {})
        thresh_uv = r.get("threshold_uv", 100.0)
        data_pre = epochs.get_data()  # n_epochs, n_ch, n_times
        n_epochs = data_pre.shape[0]
        ch_names = epochs.ch_names
        over_counts = []
        for ch_i, ch in enumerate(ch_names):
            # boolean array for epochs where channel exceeded threshold
            exceeded = np.any(np.abs(data_pre[:, ch_i, :]) * 1e6 > thresh_uv, axis=1)
            over_counts.append((ch, int(exceeded.sum())))
        # compute percentage
        over_pct = [(ch, cnt / max(1, n_epochs) * 100.0) for ch, cnt in over_counts]
        # if diag, print top 10 channels by percentage
        if diag:
            from operator import itemgetter

            top10 = sorted(over_pct, key=itemgetter(1), reverse=True)[:10]
            print('\nPer-channel over-threshold percentage (top 10):')
            for ch, pct in top10:
                print(f' {ch}: {pct:.1f}% ({dict(over_counts)[ch]}/{n_epochs})')

        # Bad-channel detection and interpolation (config-driven)
        # Note on Methodology (Question 1 design choice): Global bad channel detection
        # runs per-subject during Stage 1 preprocessing (before CV fold splitting).
        # Rationale: Flagging electrodes with poor physical contact / disconnects across >=50%
        # of a subject's session reflects hardware recording quality, not label-dependent
        # information, so executing it once per subject before CV does not leak target information.
        bc_cfg = self.config.get('bad_channel_detection', {})
        if bc_cfg.get('enabled', True):
            pct_thr = float(bc_cfg.get('percent_threshold', 50.0))
            # pct_thr may be given as fraction (0-1) or percent (1-100)
            if pct_thr <= 1.0:
                pct_thr = pct_thr * 100.0
            bads = [ch for ch, pct in over_pct if pct >= pct_thr]
            if bads:
                if diag:
                    print('Detected bad channels (>= {:.1f}% epochs over threshold): {}'.format(pct_thr, bads))
                # set raw.info['bads'] with stripped names
                try:
                    raw.info['bads'] = bads
                    # interpolate bads on raw
                    try:
                        raw.interpolate_bads(reset_bads=False)
                    except Exception as e:
                        if diag:
                            print('Interpolation failed:', e)
                except Exception as e:
                    if diag:
                        print('Could not mark/interpolate bad channels:', e)
                # re-epoch from interpolated raw
                try:
                    epochs = self.epoch(raw, events, event_id)
                except Exception as e:
                    if diag:
                        print('Re-epoch after interpolation failed:', e)

        # rejection using autoreject (repairs and interpolates bad channels per-epoch)
        epochs_before = len(epochs)
        if diag:
            print(f"Diagnostic: total epochs before rejection = {epochs_before}")
        try:
            epochs = self.reject_with_autoreject(epochs, diag=diag)
        except Exception as e:
            if diag:
                print('AutoReject failed, falling back to legacy reject_by_band:', e)
            epochs = self.reject_by_band(epochs, diag=diag)
        epochs_after = len(epochs)
        if diag:
            print(f"Diagnostic: total epochs after rejection = {epochs_after}")
            # print drop_log summary
            drop_log = getattr(epochs, 'drop_log', None)
            if drop_log is not None:
                # summarize drop reasons counts; drop_log can be list of dicts or tuples
                reasons = []
                for r in drop_log:
                    if isinstance(r, dict):
                        reasons.append(r.get('reason', str(r)))
                    elif isinstance(r, (list, tuple)) and len(r) > 0:
                        # tuple like (reason, indices)
                        reasons.append(str(r[0]))
                    else:
                        reasons.append(str(r))
                from collections import Counter
                print('Diagnostic: epochs.drop_log counts:', Counter(reasons))

        # baseline correction already applied if baseline set in config during epoching
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)

        # map event codes to labels
        if events is not None and event_id is not None:
            code_map = {int(code): label for label, code in event_id.items()}
        else:
            code_map = map_event_id_to_labels(event_id, mapping=mapping)
        labels = [code_map[int(e)] for e in epochs.events[:, 2]]

        # build metadata
        subject_id = self.config.get('subject_id', 'unknown')
        metadata = []
        for idx, lab in enumerate(labels):
            metadata.append({'subject_id': subject_id, 'label': lab, 'epoch_index': int(epochs.selection[idx])})

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            np.savez_compressed(os.path.join(output_dir, 'epochs.npz'), X=data)
            # write simple CSV metadata
            meta_path = os.path.join(output_dir, 'metadata.csv')
            with open(meta_path, 'w', encoding='utf8') as fh:
                fh.write('subject_id,label,epoch_index\n')
                for row in metadata:
                    fh.write(f"{row['subject_id']},{row['label']},{row['epoch_index']}\n")

        return data, labels, metadata
