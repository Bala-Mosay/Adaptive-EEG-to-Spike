"""Simple CLI to run Phase 1 preprocessing for a single subject."""
from pathlib import Path
import argparse
import yaml
from src.data.loader import load_subject_raw_with_events
from src.data.preprocessing import Preprocessor
from src.data.events import events_from_raw, map_event_id_to_labels
from src.data.normalization import PerSubjectZScore
import os
import json


def run_subject(subject_id: str, config_path: str):
    cfg = yaml.safe_load(open(config_path))
    if subject_id in cfg.get("excluded_subjects", []):
        raise ValueError(f"Subject {subject_id} is excluded by config")
    raw, run_files, events, event_id = load_subject_raw_with_events(subject_id)
    # attach subject id to config for metadata
    cfg['subject_id'] = subject_id
    pre = Preprocessor(cfg)
    out_dir = cfg.get('output_dir', f'outputs/{subject_id}')
    data, labels, metadata = pre.process_subject(raw, output_dir=out_dir, mapping=cfg.get('event_mapping'), diag=cfg.get('diagnostic', False), run_files=run_files, events=events, event_id=event_id)

    print(f"Subject {subject_id}: epochs before/after rejection: {len(labels)}/{len(data)}")
    # save basic run info
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'run_info.json'), 'w') as fh:
        json.dump({'subject_id': subject_id, 'n_epochs': len(data)}, fh)


def main():
    p = argparse.ArgumentParser(description="Phase1 pipeline runner")
    p.add_argument("--config", required=True)
    p.add_argument("--subject", required=True)
    args = p.parse_args()
    run_subject(args.subject, args.config)


if __name__ == "__main__":
    main()
