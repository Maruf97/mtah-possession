"""Synthetic end-to-end test: fake ATD + Smart Tagging in the official formats, scripted
possessions, run the model, score with the organisers' metric. NOT real data -- only checks the
pipeline runs and that the decoder recovers clean possessions from noisy proximity.

Usage:  python smoke_test.py .\\mtah"""
import os, sys, json, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "mtah")  # path to the hackathon repo
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import games, possession_model as pm, possession_metrics as metrics

rng = np.random.RandomState(0)
N, FPS, L, W = games.GAMES["BAR-ATH"]["n_video_frames"], 25, 105.3, 68.0
blocks = games.half_video_blocks("BAR-ATH")
NSLOT = 46

# scripted truth: inside each half, alternate possessions of 5-40 s with dead balls in between
truth = np.full(N, 3, dtype=np.int8)  # 3 = X
for half, a, b in blocks:
    t = a
    team = 0 if half == 1 else 1
    while t <= b:
        dead = rng.rand() < 0.3
        ln = int(rng.uniform(2, 8) * FPS) if dead else int(rng.uniform(5, 40) * FPS)
        truth[t:min(b + 1, t + ln)] = 2 if dead else team
        if not dead and rng.rand() < 0.8:
            team = 1 - team
        t += ln

# fake ATD: 22 "players" on slots, ball near a player of the possessing team, invisible 41%
tmp = os.path.join(HERE, "_synth"); shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(os.path.join(tmp, "tactical"))
meta = ['<?xml version="1.0"?><Metadata><Teams>',
        '<Team id="t1"><Name>Barcelona</Name></Team><Team id="t2"><Name>Athletic Club</Name></Team>',
        '</Teams><Players>']
for s in range(NSLOT):
    meta.append('<Player id="p{}" teamId="{}"><Name>Track {}</Name></Player>'.format(
        s + 1, "t1" if s % 2 == 0 else "t2", s + 1))
meta.append('</Players></Metadata>')
open(os.path.join(tmp, "tactical", "BAR-ATH_tactical_atd.xml"), "w").write("".join(meta))

pos = rng.uniform(0.1, 0.9, size=(NSLOT, 2))
with open(os.path.join(tmp, "tactical", "BAR-ATH_tactical_atd.txt"), "w") as fh:
    for i in range(N):
        pos += rng.normal(0, 0.002, size=pos.shape); pos = np.clip(pos, 0, 1)
        groups = []
        active = truth[i] != 3
        for s in range(NSLOT):
            if s < 22 and (active or rng.rand() < 0.2):
                spd = 2.0 + rng.rand() * 3 if truth[i] in (0, 1) else 0.3
                groups.append("{:.4f},{:.4f},{:.2f}".format(pos[s, 0], pos[s, 1], spd))
            else:
                groups.append("NaN,NaN,NaN")
        if truth[i] in (0, 1) and rng.rand() > 0.41:
            holder = rng.choice([s for s in range(22) if s % 2 == truth[i]]) if rng.rand() > 0.15 \
                else rng.randint(0, 22)  # 15% noise: nearest player is wrong team
            bx, by = pos[holder] + rng.normal(0, 0.004, 2)
            ballstr = "{:.4f},{:.4f},{:.2f}".format(bx, by, 5.0)
        elif truth[i] == 2 and rng.rand() > 0.5:
            ballstr = "{:.4f},{:.4f},0.00".format(-0.02, 0.5)   # out of play
        else:
            ballstr = "NaN,NaN,NaN"
        fh.write("{}:{}:{}\n".format(i + 1, ";".join(groups), ballstr))

# minimal smart tagging with the two kickoffs
xml = ['<file><ALL_INSTANCES>']
for k, (half, a, b) in enumerate(blocks):
    xml.append('<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>SET PIECES</code>'
               '<label><group>Team</group><text>Barcelona</text></label>'
               '<label><group>Type</group><text>Kick Off (Start)</text></label>'
               '<label><group>Half</group><text>{} Half</text></label></instance>'.format(
                   k, (a - 50) / FPS, (a + 50) / FPS, "1st" if half == 1 else "2nd"))
xml.append('</ALL_INSTANCES></file>')
open(os.path.join(tmp, "tactical", "BAR-ATH_tactical_smart_tagging.xml"), "w").write("".join(xml))

out = os.path.join(tmp, "synth_submission.csv")
labels = pm.run(REPO, tmp, out)

gt = np.array(["HADX"[c] for c in truth], dtype=object)
res = metrics.evaluate(metrics.label_codes(gt), metrics.label_codes(labels))
print("\nsynthetic score: spellF1 {:.3f}  macroF1 {:.3f}  gt spells {}  pred spells {}".format(
    res["spell_f1_mean"], res["macro_f1"], res["n_gt_spells"], res["n_pred_spells"]))
