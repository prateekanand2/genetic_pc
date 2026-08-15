#!/bin/bash
# Progress across all HMM training runs: per-chunk epoch counts and ETA.
cd "$(dirname "$0")"
python - <<'PY'
import os, re, time
from hmm_config import CONFIGS, model_paths

now = time.time()
rows, running = [], 0
for key in sorted(CONFIGS):
    c = CONFIGS[key]
    E = c["epochs"]
    for i, ckpt in enumerate(model_paths(c)):
        log = os.path.join(c["model_dir"],
                           f"{c['tag']}_{i}_{c['split']}_hmm_{c['n_train']}_128_{E}epochs_ps0.005.log")
        if not os.path.exists(log):
            rows.append((key, i, 0, E, "", "")); continue
        lines = [l for l in open(log) if l.startswith("[Epoch")]
        n = len(lines)
        done = n >= E
        # seconds/epoch from the last 20 logged epochs
        rate = ""
        eta = "done" if done else ""
        if len(lines) > 3:
            ts = [sum(map(float, re.findall(r"([\d.]+); val forward ([\d.]+)", l)[0]))
                  for l in lines[-20:]]
            s = sum(ts) / len(ts)
            rate = f"{s:.1f}s"
            if not done:
                fresh = (now - os.path.getmtime(log)) < 300
                running += fresh
                eta = f"{(E - n) * s / 3600:.1f}h" + ("" if fresh else "  (stalled)")
        rows.append((key, i, n, E, rate, eta))

print(f"{'config':28s} {'chunk':>5s} {'epochs':>12s} {'s/ep':>6s}  eta")
print("-" * 68)
last = None
for key, i, n, E, rate, eta in rows:
    label = key if key != last else ""
    last = key
    bar = f"{n}/{E}"
    print(f"{label:28s} {i:5d} {bar:>12s} {rate:>6s}  {eta}")

tot = sum(1 for r in rows if r[2] >= r[3])
print("-" * 68)
print(f"{tot}/{len(rows)} chunks complete; {running} actively training")
PY
