"""Merge bench.py logs: summ.py gluon_ref.log name1=a.log name2=b.log ...
Prints per (q, n, m): gluon us (from the ref log's gluon column) and each
variant's lamport us, plus the best variant and best/gluon."""

import re
import sys


def parse(path):
    rows = {}
    for line in open(path):
        m = re.match(r"\| (\d) \| (\d+) \| (\d+) \| \d+ \| ([\d.na]+) \| ([\d.na]+) \|", line)
        if m:
            q, n, mm, lam, glu = m.groups()
            rows[(int(q), int(n), int(mm))] = (float(lam), float(glu))
    return rows


ref = parse(sys.argv[1])
vars_ = [(a.split("=")[0], parse(a.split("=")[1])) for a in sys.argv[2:]]
keys = sorted(set().union(*[v.keys() for _, v in vars_]))
hdr = ["q", "n", "m", "gluon", "lamport_old"] + [n for n, _ in vars_] + ["best", "best/gluon"]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for k in keys:
    g = ref.get(k, (float("nan"), float("nan")))
    vals = [v.get(k, (float("nan"),))[0] for _, v in vars_]
    best_i = min(range(len(vals)), key=lambda i: vals[i] if vals[i] == vals[i] else 1e9)
    cells = [str(k[0]), str(k[1]), str(k[2]), f"{g[1]:.2f}", f"{g[0]:.2f}"]
    cells += [f"{v:.2f}" for v in vals]
    cells += [vars_[best_i][0], f"{vals[best_i] / g[1]:.3f}"]
    print("| " + " | ".join(cells) + " |")
