# BAR-ATH ball possession — More than a Hack 2027, Phase 1

Per-frame possession labels (H / A / D / X) for the tactical video of Barcelona – Athletic Club,
produced by a three-state Viterbi segmenter over evidence from the provided automatic tracking and
Smart Tagging. Its four main parameters were tuned on **external** data (SkillCorner open data) and
frozen. Nothing is trained, fitted or tuned on the provided game, its video or its ground truth.

The technical write-up is in [`TECHNICAL_DOC.md`](TECHNICAL_DOC.md).

## Files

| File | What it does |
|---|---|
| `possession_model.py` | The whole pipeline: load tracking + Smart Tagging → per-frame evidence → Viterbi → submission CSV. All parameters are in `CONFIG` at the top. |
| `tune_skillcorner.py` | Grid search of the parameters on SkillCorner open-data matches, scored with the organisers' `possession_metrics.py`. Produced the frozen values in `CONFIG`. |
| `smoke_test.py` | End-to-end check on a synthetic match in the official file formats. Tests the pipeline, not performance. |
| `requirements.txt` | Pinned dependencies. |

The match data is **not** in this repository and must never be committed (see `.gitignore`).

## Setup (Windows, PowerShell)

Requires Git (with Git LFS, bundled with Git for Windows) and Python 3.12.

```powershell
git clone <this repository URL> hackathon
cd hackathon
python -m pip install -r requirements.txt
git clone https://github.com/metrica-sports/more-than-a-hack-2027 mtah
```

Unpack the organisers' data bundle into `mtah\data`, so that
`mtah\data\tactical\BAR-ATH_tactical_atd.txt` exists. (Or put it anywhere and set
`$env:MTAH_DATA_DIR` to that folder.)

## Regenerate the submission

```powershell
python possession_model.py --repo .\mtah --out <team_name>.csv
python .\mtah\evaluation\validate_submission.py <team_name>.csv
```

The pipeline is deterministic: same data bundle + same code → the same file. Runtime is a few
minutes on a laptop.

## Reproduce the tuning (optional)

```powershell
git clone https://github.com/SkillCorner/opendata
cd opendata
git lfs install
git lfs pull --include="data/matches/1874553/*,data/matches/1886347/*,data/matches/1899585/*"
cd ..
python tune_skillcorner.py --opendata .\opendata --repo .\mtah --matches 3 --out best_config.json
```

The grid in `tune_skillcorner.py` is the second (extended) search; the first search used
`switch_penalty` 10–32, `dead_penalty` 4–12, `prox_scale_m` 1.5–3.0 and `max_team_logodds` 2.0–3.5.

## Synthetic smoke test (optional)

```powershell
python smoke_test.py .\mtah
```

## Credits

Tuning data: [SkillCorner open data](https://github.com/SkillCorner/opendata) (MIT licence).
Scoring code: the organisers' `evaluation/possession_metrics.py`.
