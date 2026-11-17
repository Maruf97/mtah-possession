"""Tune possession_model.CONFIG on EXTERNAL data: SkillCorner open broadcast tracking
(github.com/SkillCorner/opendata, MIT, 10 matches of A-League 2024/25, 10 fps).

Why this is legal under the hackathon rules: nothing here touches BAR-ATH. The per-frame
`possession.group` field of the SkillCorner tracking is the tuning target, `phases_of_play.csv`
(defined only while the ball is in play) gives the alive/dead split, and the scorer is the
organisers' own possession_metrics.py so we optimise the real ranking metric (spellF1).

Frame-rate handling: the decoder accumulates evidence per frame, so every frame-denominated
constant (penalties, gap lengths, tails) is scaled by fps/25 before use. Tuned values are
reported in 25 fps units and can be pasted straight into CONFIG.

Usage (after `git lfs pull` in the opendata clone):
    python tune_skillcorner.py --opendata D:\\opendata --repo D:\\mtah --matches 3 --out best.json
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession_model as pm  # noqa: E402

FPS_REF = 25.0


def scale_cfg(cfg, fps):
    """Frame-denominated constants are tuned at 25 fps; rescale for another frame rate."""
    out = dict(cfg)
    r = fps / FPS_REF
    for k in ("max_interp_frames", "long_gap_frames", "setpiece_owner_tail_frames",
              "goal_dead_tail_frames", "min_possession_frames"):
        out[k] = int(round(cfg[k] * r))
    for k in ("switch_penalty", "dead_penalty"):
        out[k] = cfg[k] * r
    return out


def load_match(folder, fps=10):
    mid = os.path.basename(folder.rstrip("/\\"))
    meta = json.load(open(os.path.join(folder, "{}_match.json".format(mid))))
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    team_of_player = {p["id"]: (pm.HOME if p["team_id"] == home_id else pm.AWAY)
                      for p in meta["players"]}
    slot_of_player = {pid: i for i, pid in enumerate(sorted(team_of_player))}
    team_of_slot = np.array([team_of_player[pid] for pid in sorted(team_of_player)], dtype=np.int8)

    periods = {p["period"]: (p["start_frame"], p["end_frame"]) for p in meta["match_periods"]}
    n = max(e for _, e in periods.values()) + 1
    players = np.full((n, len(team_of_slot), 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    poss = np.full(n, -1, dtype=np.int8)      # -1 unknown, 0 H, 1 A
    period_of = np.zeros(n, dtype=np.int8)

    with open(os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(mid))) as fh:
        for line in fh:
            rec = json.loads(line)
            f = rec["frame"]
            if f >= n:
                continue
            period_of[f] = rec.get("period") or 0
            bd = rec.get("ball_data") or {}
            if bd.get("x") is not None and bd.get("y") is not None:
                ball[f, 0] = float(bd["x"]) + L / 2
                ball[f, 1] = float(bd["y"]) + W / 2
            for p in rec.get("player_data") or []:
                s = slot_of_player.get(p.get("player_id"))
                if s is None or p.get("x") is None:
                    continue
                players[f, s, 0] = float(p["x"]) + L / 2
                players[f, s, 1] = float(p["y"]) + W / 2
            # SkillCorner writes "home team" / "away team" (older files: "home" / "away")
            g = str((rec.get("possession") or {}).get("group") or "").strip().lower()
            if g.startswith("home"):
                poss[f] = pm.HOME
            elif g.startswith("away"):
                poss[f] = pm.AWAY

    # speeds from finite differences (SkillCorner gives no speed field)
    d = np.diff(players[:, :, :2], axis=0)
    sp = np.linalg.norm(d, axis=2) * fps
    players[1:, :, 2] = sp
    ball[1:, 2] = np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1) * fps

    # ground truth: alive inside phases of play, D elsewhere inside the periods, X outside
    alive = np.zeros(n, dtype=bool)
    import csv
    with open(os.path.join(folder, "{}_phases_of_play.csv".format(mid))) as fh:
        for row in csv.DictReader(fh):
            a, b = int(row["frame_start"]), int(row["frame_end"])
            alive[a:b + 1] = True
    gt = np.full(n, 3, dtype=np.int8)   # X
    for per, (a, b) in periods.items():
        gt[a:b + 1] = pm.DEAD
        seg = slice(a, b + 1)
        lab = poss[seg].copy()
        # loose-ball rule: last team keeps it -> forward-fill unknown possession within alive
        last = -1
        for i in range(len(lab)):
            if lab[i] >= 0:
                last = lab[i]
            elif alive[a + i] and last >= 0:
                lab[i] = last
        block = gt[seg]
        block[alive[seg] & (lab >= 0)] = lab[alive[seg] & (lab >= 0)]
        gt[seg] = block
    blocks = [(per, a, b) for per, (a, b) in sorted(periods.items())]
    graded = gt != 3
    print("  match {}: frames with team label {:.1f}%, ground truth H {:.1f}%  A {:.1f}%  D {:.1f}%"
          .format(mid, 100 * np.mean(poss >= 0),
                  *[100 * np.mean(gt[graded] == k) for k in (pm.HOME, pm.AWAY, pm.DEAD)]))
    if not np.any((gt == pm.HOME) | (gt == pm.AWAY)):
        sys.exit("no possession labels found in match {} -- the 'possession.group' values "
                 "were not recognised; send this output for a fix".format(mid))
    return dict(id=mid, players=players, ball=ball, team_of_slot=team_of_slot, pitch=(L, W),
                gt=gt, blocks=blocks, fps=fps)


def decode(match, cfg):
    c = scale_cfg(cfg, match["fps"])
    scores, _ = pm.build_evidence(match["players"], match["ball"], match["team_of_slot"], [],
                                  "home", "away", match["pitch"], match["fps"], c)
    pred = np.full(len(scores), 3, dtype=np.int8)
    for _, a, b in match["blocks"]:
        path = pm.viterbi(scores[a:b + 1], c["switch_penalty"], c["dead_penalty"])
        pred[a:b + 1] = pm.merge_short_possessions(path, c["min_possession_frames"])
    return pred


def score(match, pred, metrics):
    res = metrics.evaluate(match["gt"], pred)
    return res["spell_f1_mean"], res["macro_f1"], res["n_gt_spells"], res["n_pred_spells"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opendata", required=True, help="path to the SkillCorner/opendata clone")
    ap.add_argument("--repo", required=True, help="path to the more-than-a-hack-2027 clone")
    ap.add_argument("--matches", type=int, default=3, help="how many matches to use")
    ap.add_argument("--out", default="best_config.json")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.repo, "evaluation"))
    import possession_metrics as metrics  # noqa: E402

    folders = sorted(glob.glob(os.path.join(args.opendata, "data", "matches", "*")))
    matches = []
    for folder in folders[:args.matches]:
        trk = glob.glob(os.path.join(folder, "*_tracking_extrapolated.jsonl"))
        if not trk or os.path.getsize(trk[0]) < 10000:
            print("skip {} (tracking not pulled -- run `git lfs pull`)".format(folder))
            continue
        print("loading", folder)
        matches.append(load_match(folder))
    if not matches:
        sys.exit("no matches with tracking data")

    # default first, so you can see what the priors give before any tuning
    base = dict(pm.CONFIG)
    rows = [score(m, decode(m, base), metrics) for m in matches]
    print("priors: spellF1 {:.3f}  macroF1 {:.3f}   {}".format(
        np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows]),
        [(r[2], r[3]) for r in rows]))

    grid = dict(
        switch_penalty=[32, 40, 50, 64],
        dead_penalty=[12, 16, 24],
        prox_scale_m=[1.5, 2.0],
        max_team_logodds=[1.5, 2.0],
    )
    best = (-1, None)
    keys = list(grid)
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(base); cfg.update(zip(keys, combo))
        s = float(np.mean([score(m, decode(m, cfg), metrics)[0] for m in matches]))
        if s > best[0]:
            best = (s, cfg)
            print("  new best spellF1 {:.3f}  {}".format(s, dict(zip(keys, combo))))
    print("\nbest external spellF1 {:.3f}".format(best[0]))
    json.dump(best[1], open(args.out, "w"), indent=2)
    print("wrote", args.out, "-- paste these values into possession_model.CONFIG and freeze.")


if __name__ == "__main__":
    main()
