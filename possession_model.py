"""Possession inference for More than a Hack 2027, Phase 1 (BAR-ATH).

A three-state (H / A / D) Viterbi segmenter over per-frame evidence computed from the provided
automatic tracking (ATD) and Smart Tagging. No learning on the provided game: every number in
CONFIG is either a football prior (documented next to it) or was tuned on EXTERNAL data
(SkillCorner open data, three A-League matches; see tune_skillcorner.py) and then frozen.

Pipeline
--------
  1. Parse the raw ATD into arrays: players (n, 46, 3), ball (n, 3), normalised 0-1 -> metres.
  2. Bridge short ball gaps by linear interpolation (ball flight is smooth over < ~2 s).
  3. Per-frame evidence:
       - team proximity: how close the nearest H / A player is to the ball (control evidence);
       - dead-ball evidence: ball outside the pitch, long ball absence with slow players, few
         tracked objects (halftime, warm-up), Smart Tagging set pieces / goals;
       - Smart Tagging phase priors: attacking phases tagged with team T favour T, defending
         phases tagged with team T favour the other team, set pieces favour D then the owner.
  4. Viterbi decoding with switch penalties. The H<->A penalty is the operational definition
     of "control": a team has to accumulate roughly SWITCH_PENALTY / (per-frame log-odds) frames
     of being clearly closest to the ball before the decoder hands possession over. A single
     touch, a deflection or a won header that is not followed by sustained proximity never
     accumulates enough evidence, so possession stays with the previous team -- exactly the
     "last team keeps it" rule in CHALLENGE.md. Ball invisible => no team evidence => the state
     is carried forward for free.
  5. Frames outside the two halves are written as X (never graded).

Usage
-----
    export MTAH_DATA_DIR=/path/to/bundle    # or put files in <repo>/data
    python possession_model.py --repo /path/to/more-than-a-hack-2027 --out my_team.csv
    python /path/to/repo/evaluation/validate_submission.py my_team.csv
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices are expected

# ------------------------------------------------------------------------------------------
# CONFIG -- all tunables in one place. Values marked [tuned] were chosen by grid search on three
# SkillCorner open-data matches (see tune_skillcorner.py) and frozen; the rest are football priors.
# Nothing here was fitted on BAR-ATH, its video or its ground truth.
# ------------------------------------------------------------------------------------------
CONFIG = dict(
    # --- ball gap bridging ---
    max_interp_frames=50,        # bridge ball gaps up to 2 s by linear interpolation
    # --- proximity / control evidence ---
    prox_scale_m=1.5,            # e^(-d/scale): 1.5 m ~ playing reach  [tuned, SkillCorner]
    prox_floor=0.03,             # keeps log-odds finite when nobody is near the ball
    max_team_logodds=1.5,        # cap on per-frame |log P(H)/P(A)|  [tuned, SkillCorner]
    # --- dead-ball evidence (log-odds contributions, D vs alive) ---
    oob_margin_m=0.3,            # ball beyond the lines by this much -> out of play
    w_oob=4.0,
    w_ball_missing_long=1.2,     # ball unseen for > long_gap_frames AND players slow
    long_gap_frames=75,          # 3 s
    slow_player_speed=1.2,       # m/s: median tracked speed below this = nobody is playing
    w_slow_players=1.5,
    w_few_players=3.0,           # fewer than min_players tracked objects (halftime, warm-up)
    min_players=14,
    w_alive_ball_moving=-1.5,    # ball visible, inside pitch and moving -> alive
    ball_moving_speed=1.5,       # m/s
    w_alive_ball_visible=-0.6,   # ball visible inside pitch at all -> mildly alive
    bias_dead=-0.5,              # prior: most graded frames are alive
    # --- Smart Tagging priors (log-odds added to the relevant state) ---
    w_tag_phase=0.8,             # attacking / defending phase with a Team tag
    w_tag_setpiece_dead=2.0,     # first part of a set-piece window is dead ball
    w_tag_setpiece_owner=1.5,    # after the restart the tagged team has the ball
    setpiece_dead_fraction=0.5,  # fraction of the tagged window treated as pre-restart
    setpiece_owner_tail_frames=50,  # owner prior lasts this long after the window end
    w_tag_goal_dead=2.5,
    goal_dead_tail_frames=250,   # ~10 s of celebration + walk back after a GOALS tag
    # --- Viterbi transition penalties (log units) ---
    switch_penalty=32.0,         # H<->A: ~21 frames (~0.85 s) of clear control  [tuned, SkillCorner]
    dead_penalty=24.0,           # alive<->D  [tuned, SkillCorner]
    # --- post-processing ---
    min_possession_frames=0,     # 0 = trust the decoder; >0 merges micro-possessions
)

HOME, AWAY, DEAD = 0, 1, 2
LABEL = np.array(["H", "A", "D"])

ATTACK_CODES = {"build up", "progression", "creating chances", "ball in final third",
                "ball in the box", "players in the box", "attacking transition", "long ball"}
DEFEND_CODES = {"defending in defensive third", "defending in middle third",
                "defending in attacking third", "defensive transition"}
SETPIECE_CODE = "set pieces"
GOAL_CODE = "goals"


# ------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------
def load_meta(meta_path):
    """(home_name, away_name, team_of_slot[46]) from the FIFA/EPTS metadata XML.
    First team listed is home (EPTS convention, same as starter/baseline.py)."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("expected 2 teams, found {}".format(len(teams)))
    home_id = teams[0].get("id")
    names = tuple(t.findtext("Name") or "" for t in teams)
    team_of_slot = np.array([HOME if p.get("teamId") == home_id else AWAY
                             for p in root.iter("Player")], dtype=np.int8)
    return names[0], names[1], team_of_slot


def load_atd_raw(raw_path, n_frames, n_slots, pitch):
    """players (n, n_slots, 3) and ball (n, 3) in metres / m/s, NaN where missing."""
    length, width = pitch
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if i < 0 or i >= n_frames:
                continue
            vals = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            k = min(n_slots, len(vals) // 3)
            players[i, :k] = vals[:3 * k].reshape(k, 3)
            ball[i] = np.array(parts[2].split(","), dtype=np.float32)[:3]
    scale = np.array([length, width, 1.0], dtype=np.float32)
    return players * scale, ball * scale


def load_tags(xml_path, fps):
    out = []
    for inst in ET.parse(xml_path).iter("instance"):
        s, e = float(inst.findtext("start")), float(inst.findtext("end"))
        tags = {(l.findtext("group") or "").strip().lower(): (l.findtext("text") or "").strip()
                for l in inst.findall("label")}
        out.append(dict(code=(inst.findtext("code") or "").strip().lower(),
                        start=int(round(s * fps)), end=int(round(e * fps)), tags=tags))
    return out


def team_code_of_tag(value, home_name, away_name):
    """Map a Smart Tagging 'Team' value to HOME / AWAY / None by name overlap.
    The XML may spell clubs differently from the ATD metadata ('Atletic Club' vs 'Athletic')."""
    v = (value or "").lower()
    if not v or v == "n/a":
        return None

    def score(name):
        toks = [t for t in name.lower().replace("-", " ").split() if len(t) > 2]
        return max([len(os.path.commonprefix([t, w])) for t in toks for w in v.split()] or [0])
    sh, sa = score(home_name), score(away_name)
    if sh == sa:
        return None
    return HOME if sh > sa else AWAY


# ------------------------------------------------------------------------------------------
# features
# ------------------------------------------------------------------------------------------
def interpolate_ball(ball, max_gap):
    """Linear interpolation of ball x,y across gaps of at most max_gap frames."""
    out = ball.copy()
    vis = ~np.isnan(ball[:, 0])
    idx = np.flatnonzero(vis)
    if len(idx) < 2:
        return out, vis
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a
        if 1 < gap <= max_gap:
            t = np.arange(1, gap)[:, None] / gap
            out[a + 1:b, :2] = ball[a, :2] * (1 - t) + ball[b, :2] * t
    return out, vis


def rolling_median_speed(players, window=25):
    """Median speed of tracked objects per frame, then a running mean over +-window frames."""
    sp = players[:, :, 2]
    med = np.nanmedian(np.where(np.isnan(sp), np.nan, sp), axis=1)
    med = np.where(np.isnan(med), 0.0, med)
    kernel = np.ones(2 * window + 1) / (2 * window + 1)
    return np.convolve(med, kernel, mode="same")


def ball_speed(ballxy, fps):
    v = np.full(len(ballxy), np.nan, dtype=np.float32)
    d = np.linalg.norm(np.diff(ballxy, axis=0), axis=1) * fps
    v[1:] = d
    return v


def build_evidence(players, ball, team_of_slot, tags, home_name, away_name, pitch, fps, cfg):
    n = len(ball)
    length, width = pitch
    ballxy_i, vis = interpolate_ball(ball, cfg["max_interp_frames"])
    bxy = ballxy_i[:, :2]
    has_ball = ~np.isnan(bxy[:, 0])

    # --- team proximity -------------------------------------------------------------------
    d = np.linalg.norm(players[:, :, :2] - bxy[:, None, :], axis=2)   # (n, slots), NaN-safe
    dH = np.nanmin(np.where(team_of_slot[None, :] == HOME, d, np.nan), axis=1)
    dA = np.nanmin(np.where(team_of_slot[None, :] == AWAY, d, np.nan), axis=1)
    eH = np.exp(-np.nan_to_num(dH, nan=1e3) / cfg["prox_scale_m"]) + cfg["prox_floor"]
    eA = np.exp(-np.nan_to_num(dA, nan=1e3) / cfg["prox_scale_m"]) + cfg["prox_floor"]
    team_lo = np.clip(np.log(eH / eA), -cfg["max_team_logodds"], cfg["max_team_logodds"])
    team_lo[~has_ball] = 0.0        # no ball -> no team evidence -> carry forward

    # --- dead-ball evidence ---------------------------------------------------------------
    m = cfg["oob_margin_m"]
    raw_vis = ~np.isnan(ball[:, 0])
    oob = raw_vis & ((ball[:, 0] < -m) | (ball[:, 0] > length + m) |
                     (ball[:, 1] < -m) | (ball[:, 1] > width + m))
    n_tracked = np.sum(~np.isnan(players[:, :, 0]), axis=1)
    med_speed = rolling_median_speed(players)
    bspeed = ball_speed(bxy, fps)
    # length of the current ball gap, per frame
    gap_len = np.zeros(n, dtype=np.int32)
    run = 0
    for i in range(n):
        run = 0 if raw_vis[i] else run + 1
        gap_len[i] = run
    long_gap = gap_len > cfg["long_gap_frames"]
    slow = med_speed < cfg["slow_player_speed"]
    inside = has_ball & ~oob
    moving = inside & (np.nan_to_num(bspeed, nan=0.0) > cfg["ball_moving_speed"])

    dead_lo = (cfg["bias_dead"]
               + cfg["w_oob"] * oob
               + cfg["w_ball_missing_long"] * (long_gap & slow)
               + cfg["w_slow_players"] * slow
               + cfg["w_few_players"] * (n_tracked < cfg["min_players"])
               + cfg["w_alive_ball_visible"] * inside
               + cfg["w_alive_ball_moving"] * moving)

    # --- Smart Tagging priors -------------------------------------------------------------
    tagH = np.zeros(n); tagA = np.zeros(n); tagD = np.zeros(n)
    for ev in tags:
        s, e = max(0, ev["start"]), min(n, ev["end"])
        if e <= s:
            continue
        team = team_code_of_tag(ev["tags"].get("team"), home_name, away_name)
        code = ev["code"]
        if code == SETPIECE_CODE:
            mid = s + int((e - s) * cfg["setpiece_dead_fraction"])
            tagD[s:mid] += cfg["w_tag_setpiece_dead"]
            if team is not None:
                tail = min(n, e + cfg["setpiece_owner_tail_frames"])
                (tagH if team == HOME else tagA)[mid:tail] += cfg["w_tag_setpiece_owner"]
        elif code == GOAL_CODE:
            tagD[s:min(n, e + cfg["goal_dead_tail_frames"])] += cfg["w_tag_goal_dead"]
        elif team is not None and code in ATTACK_CODES:
            (tagH if team == HOME else tagA)[s:e] += cfg["w_tag_phase"]
        elif team is not None and code in DEFEND_CODES:
            (tagA if team == HOME else tagH)[s:e] += cfg["w_tag_phase"]

    # --- per-state log scores -------------------------------------------------------------
    # alive split: log P(H|alive) ~ team_lo/2, log P(A|alive) ~ -team_lo/2 ; dead: dead_lo
    scores = np.zeros((n, 3), dtype=np.float64)
    scores[:, HOME] = 0.5 * team_lo + tagH
    scores[:, AWAY] = -0.5 * team_lo + tagA
    scores[:, DEAD] = dead_lo + tagD
    diag = dict(ball_visible_raw=raw_vis, ball_after_interp=has_ball, oob=oob,
                n_tracked=n_tracked, dH=dH, dA=dA)
    return scores, diag


# ------------------------------------------------------------------------------------------
# decoding
# ------------------------------------------------------------------------------------------
def viterbi(scores, switch_penalty, dead_penalty):
    """Max-sum decoding over states {H, A, D} with constant transition penalties."""
    n = len(scores)
    pen = np.array([[0, switch_penalty, dead_penalty],
                    [switch_penalty, 0, dead_penalty],
                    [dead_penalty, dead_penalty, 0]], dtype=np.float64)
    back = np.zeros((n, 3), dtype=np.int8)
    cur = scores[0].copy()
    for t in range(1, n):
        cand = cur[:, None] - pen          # (from, to)
        best_from = np.argmax(cand, axis=0)
        cur = cand[best_from, np.arange(3)] + scores[t]
        back[t] = best_from
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(cur))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def merge_short_possessions(path, min_len):
    """Absorb H/A runs shorter than min_len into the preceding state (loose-ball rule)."""
    if min_len <= 0:
        return path
    out = path.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        if out[i] != DEAD and (j - i) < min_len and i > 0:
            out[i:j] = out[i - 1]
        i = j
    return out


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------
def run(repo, data_dir, out_path, cfg=CONFIG, game=None):
    sys.path.insert(0, repo)
    import games  # noqa: E402  (the organisers' per-game constants)

    game = game or games.GAME
    g = games.GAMES[game]
    n_frames, fps, pitch = g["n_video_frames"], g["fps"], g["pitch"]
    meta_path, raw_path, tag_path = games.require(["atd_meta", "atd_raw", "smart_tagging"],
                                                  game, data_dir)
    home_name, away_name, team_of_slot = load_meta(meta_path)
    print("H = {}   A = {}   slots = {}".format(home_name, away_name, len(team_of_slot)))

    print("reading ATD ...")
    players, ball = load_atd_raw(raw_path, n_frames, len(team_of_slot), pitch)
    tags = load_tags(tag_path, fps)
    print("  {} frames, ball visible on {:.1f}%, {} tags".format(
        n_frames, 100 * np.mean(~np.isnan(ball[:, 0])), len(tags)))

    scores, diag = build_evidence(players, ball, team_of_slot, tags, home_name, away_name,
                                  pitch, fps, cfg)

    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in games.half_video_blocks(game):
        path = viterbi(scores[first:last + 1], cfg["switch_penalty"], cfg["dead_penalty"])
        path = merge_short_possessions(path, cfg["min_possession_frames"])
        labels[first:last + 1] = LABEL[path]
        n_poss = int(np.sum((path[1:] != path[:-1]) & (path[1:] != DEAD))) + int(path[0] != DEAD)
        print("  half {}: {} possessions, H {:.1f}%  A {:.1f}%  D {:.1f}%".format(
            half, n_poss, *[100 * np.mean(path == k) for k in (HOME, AWAY, DEAD)]))

    with open(out_path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    print("wrote", out_path)
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)),
                    help="path to the more-than-a-hack-2027 checkout (for games.py)")
    ap.add_argument("--data-dir", default=None, help="data bundle root (default: MTAH_DATA_DIR)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--switch-penalty", type=float)
    ap.add_argument("--dead-penalty", type=float)
    args = ap.parse_args()
    cfg = dict(CONFIG)
    if args.switch_penalty is not None:
        cfg["switch_penalty"] = args.switch_penalty
    if args.dead_penalty is not None:
        cfg["dead_penalty"] = args.dead_penalty
    run(args.repo, args.data_dir, args.out, cfg)
