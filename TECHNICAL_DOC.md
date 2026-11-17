# Ball possession from automatic data — technical document

**Team:** KINESTATICS  
**Challenge:** More than a Hack 2027, Phase 1 (Barcelona – Athletic Club, tactical video)  
**Code:** https://github.com/Maruf97/mtah-possession.git

---

## 1. Approach

### The problem as segmentation

The ranking metric, spellF1, matches predicted possessions to true possessions by temporal overlap
(IoU 0.3–0.7). What is being predicted is therefore not a label per frame but a *partition of each
half into possessions and dead-ball periods*. A per-frame classifier, however accurate frame by
frame, scores poorly on that metric: each short burst of noise inside a 20-second possession creates
extra "possessions", and each is a false positive. The organisers' nearest-player baseline shows
this clearly (macroF1 0.65, spellF1 0.05).

We therefore model each half as a sequence over three hidden states — **H** (Barcelona in
possession), **A** (Athletic in possession), **D** (dead ball) — and decode it jointly with the
Viterbi algorithm. Each frame contributes a log-score for each state (the *evidence*, §2), and each
change of state costs a fixed *penalty*. Viterbi returns the path that maximises total evidence
minus total penalties. Equivalently, it is the maximum-a-posteriori path of a Markov chain in which
staying in a state is free and switching is expensive — a change-point model with a geometric prior
on segment length.

### The penalties are the definition of control

- **H ↔ A switch penalty = 32.** Per-frame team evidence is capped at ±1.5 log-units, so even in
  the clearest case the other team needs about 32 / 1.5 ≈ 21 frames (≈ 0.85 s) of being clearly
  closest to the ball before the decoder hands possession over. When the situation is contested and
  the evidence per frame is weaker, it needs longer. A touch, a deflection or a won header that is
  not followed by sustained proximity never pays the penalty, so possession stays with the previous
  team. This is the challenge's loose-ball rule ("the team that last had the ball keeps it"),
  obtained from the decoder rather than as a separate rule.
- **Alive ↔ D penalty = 24.** Dead-ball evidence per frame is weaker than team evidence but lasts
  several seconds, so the decoder enters D once that evidence is consistent over a stoppage.
- **No ball, no evidence.** On frames where the ball is not tracked (about 41 % of graded frames),
  team evidence is zero. Staying in the current state then costs nothing and switching costs the
  full penalty, so the current team is carried forward. We prefer holding to guessing: the scorer
  forward-fills abstentions anyway, so holding through an occlusion is the natural default.

## 2. Assets used, and how

| Asset | Used | How |
|---|---|---|
| Tactical ATD (`_atd.txt`, `_atd.xml`) | **Yes** | Player and ball positions in metres; team of each track slot from the metadata (first team listed = home = Barcelona). Ball gaps of up to 2 s bridged by linear interpolation. |
| Tactical Smart Tagging (`_smart_tagging.xml`) | **Yes**, as soft priors | Phase and set-piece windows add evidence (below). Team names matched to the ATD team names by prefix overlap. |
| Tactical homography | No | Not needed: the ATD is already in pitch coordinates. |
| Tactical video | Inspection only | Watched to understand what the ground truth counts as possession. Never used as model input and never used to check or correct output. |
| Broadcast video and ATD | No | On a separate timeline with no frame mapping to the tactical video. |
| Public 5-minute ground truth | Pipeline check only | Used to confirm frame alignment, file format and that H = Barcelona. Not used for training or tuning. |

### Per-frame evidence (log-odds)

**Team evidence.** For each frame, `d_H` and `d_A` are the distances from the ball to the nearest
Barcelona and nearest Athletic player. Each is turned into a soft "could be playing it" weight,
`e_T = exp(−d_T / 1.5 m) + 0.03` (1.5 m ≈ playing reach; the 0.03 floor keeps the ratio finite when
nobody is near the ball). The team log-odds `log(e_H / e_A)` is clipped to ±1.5 and split equally
between H and A. The clip acts as a tempered likelihood ratio: it stops any single very confident
frame from overriding the sequence prior.

**Dead-ball evidence** — a weighted sum of indicators:

| Cue | Weight |
|---|---|
| Ball more than 0.3 m beyond a touch line or goal line | +4.0 |
| Ball unseen for more than 3 s **and** players slow | +1.2 |
| Median tracked player speed below 1.2 m/s | +1.5 |
| Fewer than 14 tracked objects (halftime, warm-up, before kick-off) | +3.0 |
| Ball visible inside the pitch | −0.6 |
| Ball visible and moving faster than 1.5 m/s | −1.5 |
| Constant (most graded frames are in play) | −0.5 |

**Smart Tagging priors.** An attacking phase (build-up, progression, chance creation, final third,
box, attacking transition, long ball) tagged with team T adds +0.8 per frame to T. A defending phase
tagged with T adds +0.8 to the *other* team, since the tagged team is the one without the ball. A
set-piece window adds +2.0 to D over its first half (the stoppage) and +1.5 to the tagged team for
the rest of the window plus 2 s (the restart). A goal adds +2.5 to D over the tag window and the 10 s after it. All
of these are well below the switch penalty: tags bias the decoder but never dictate to it.

## 3. Training data

We trained no model; four parameters were **tuned** by grid search, entirely on external data.

| | |
|---|---|
| Dataset | SkillCorner open data — github.com/SkillCorner/opendata (MIT licence) |
| Matches | Three Australian A-League 2024/25 matches: IDs 1874553, 1886347, 1899585 |
| Size | About 5 hours of play at 10 fps (≈ 180,000 frames); 987 possessions in the derived labels |
| Labels | Per-frame `possession.group` field of SkillCorner's tracking (produced by SkillCorner, not by us), forward-filled across unlabelled in-play frames (last team keeps it). Ball-in-play periods from SkillCorner's `phases_of_play` file; all other in-period frames labelled D. |
| Inputs used | Player and ball positions only (no Smart Tagging equivalent exists in this data) |
| Objective | Mean spellF1 over the three matches, computed with the organisers' `possession_metrics.py` |

Frame-denominated constants (penalties, gap lengths) are rescaled from 25 fps to 10 fps during
tuning, so the tuned values carry over to the 25 fps tactical tracking.

**Two-stage search.**

1. `switch_penalty` ∈ {10, 14, 18, 24, 32}, `dead_penalty` ∈ {4, 8, 12},
   `prox_scale_m` ∈ {1.5, 2.0, 3.0}, `max_team_logodds` ∈ {2.0, 2.5, 3.5}.
   The best result was at the upper edge of both penalties, so the search was extended.
2. `switch_penalty` ∈ {32, 40, 50, 64}, `dead_penalty` ∈ {12, 16, 24},
   `prox_scale_m` ∈ {1.5, 2.0}, `max_team_logodds` ∈ {1.5, 2.0}.

| Configuration | spellF1 (3 SkillCorner matches) |
|---|---|
| Initial football priors (18 / 8 / 2.0 m / 2.5) | 0.330 (macroF1 0.746) |
| Best of first search (32 / 12 / 2.0 m / 2.0) | 0.351 |
| **Best of second search — frozen (32 / 24 / 1.5 m / 1.5)** | **0.360** |

We stopped at that point even though some values sat at the edge of the second grid: the gains were
shrinking (+0.021, then +0.009), and with three matches, differences of that size are close to the
noise. A finer search risked fitting those three games. The values were frozen before the model was
run on BAR-ATH, and nothing was changed afterwards. All other constants are football priors.

## 4. Possession rules implemented

- **Ball in play.** One team is always in possession; within a half the decoder never outputs
  "unknown". `X` is written only outside the two graded halves.
- **Loose balls, duels, deflections, aerial challenges.** Possession stays with the last team
  unless the other team shows sustained control (about 0.85 s or more of being clearly closest to
  the ball; §1).
- **Restarts.** A Smart Tagging set-piece window is decoded as D, then as the tagged team.
  Throw-ins have no team tag; after them, the team whose player is nearest the ball takes
  possession through the normal evidence.
- **Dead ball.** Ball out of bounds, the period after a goal, and stoppages with no play
  (players slow, few tracked objects). Grading of the second half starts before the kick-off;
  those frames come out as D from the inactivity cues, without any frame numbers hard-coded.

## 5. What we tried and dropped

- **Majority-vote smoothing of nearest-player labels** (the organisers' baseline approach). It
  still fragments long possessions and shifts their boundaries, which is costly under spellF1.
- **A minimum-possession-length post-filter.** Redundant once the switch penalty is in place, so
  it is disabled (`min_possession_frames = 0`).
- **Homography and the broadcast feed.** No added information for this task (§2).

## 6. Limitations, and what we would do with more time

- **Proximity, not control.** The model measures how close each team is to the ball, not whether
  a player is actually playing it. A ball-control test — the ball changes direction or speed while
  within a player's reach (as in databallpy's possession method) — would separate the two in duels.
- **Long ball gaps are held, not inferred.** Gaps longer than 2 s carry the previous team forward;
  a possession change inside a long occlusion is detected only when the ball reappears.
- **Late transitions after gaps.** Interpolation registers a change of team only once the ball is
  seen again, so some possession boundaries are placed a little late.
- **Smart Tagging weights are untuned.** The SkillCorner data has no Smart Tagging equivalent, so
  the tag weights are priors only. Team names are matched heuristically.
- **Domain gap in tuning.** The tuning data is broadcast tracking at 10 fps from another league;
  the target is tactical-camera tracking at 25 fps. Values are rescaled per frame, but the
  occlusion patterns differ.
- **Small tuning set.** Three matches; more external matches (for example databallpy's DFL open
  games) would make the tuned values more reliable.

## 7. Reproduction

See `README.md` in the repository. With the data bundle unpacked into `mtah\data`:

```powershell
python -m pip install -r requirements.txt
python possession_model.py --repo .\mtah --out <team_name>.csv
python .\mtah\evaluation\validate_submission.py <team_name>.csv
```

The pipeline is deterministic: the same data bundle and code produce the same file.
`tune_skillcorner.py` reproduces the parameter search from the SkillCorner open data.
