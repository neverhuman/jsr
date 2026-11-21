# Training operations scripts

These are the scripts that ran the 2026-11 training campaign. They are committed here so the review can audit them. The copies that actually ran live on each host's data drive:
- H0 (xbabe3): `/mnt/xbabe3-external/jsr/runs/train/`
- xbabe1: `/mnt/amphora-backup/jsr-data/`
- xbabe4: `~/jsr-train/`

The scripts use absolute host paths, and each one starts with a comment saying what it does.

Two guard modules ship with the package itself and are not copied here:
- `../launch_guarded.sh`
- `../mac_guard.py` (and `ultra/fleet_guard.py`)

| Area | Scripts |
|---|---|
| Watching runs | `divergence_watch.py` (external loss/WER divergence stop for runs that predate the in-trainer guard), `sync_and_post.py` (pulls a remote run's log, posts evals), `dev_curve_remote.py`, `runs/D2_S_960h/dev_curve.*`, `qkv_norms.py` (per-block qkv norms from checkpoints) |
| Finishing runs | `finish_D2.sh`, `finish_remote.sh` (full dev-clean/dev-other scoring, copy plus sha256 check to H0), `r4t_finish.sh`, `runs/R4T_tc100/ref_eval.sh` |
| Checkpoint selection | `select_finalist.sh`, `select_D2MR.sh`, `post_selection.py` |
| Launch chains | `runs/D2_S_960h/launch_when_ready.sh`, `launch_R4T960.sh`, `xbabe1/abl.sh` (qkv ablations), `xbabe1/d3mrc.sh` and `sync_d3mrc.sh` (D3MRc), `xbabe4/d4m_relaunch.sh` |
| Reporting | `runs/D1d/post_evals.py` |
