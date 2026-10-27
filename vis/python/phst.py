"""
Reader for the AthenaK dust particle history (file_type = phst, <basename>.phst).

One row per output time.  Every run writes 16
columns: time, dt, n_particles, mass, mom1..3, ekin, z_rms, rho_max and the centre
(rho_max_x1..x3) and level (rho_max_level) of that cell, chi_max, dt_stiff_min.  Runs
with G/R (refinable clouds with merging or resampling on a refined mesh) add g_groups
and r_removed (18).  The set is fixed for a run, restarts included; the header names
the columns.  The file is in the AthenaK history format (athena_read.hst reads it too);
a restarted run appends a new header block.  Only numpy is needed.

Example::

    import phst
    h = phst.read("st.phst")
    t, rho_max = h["time"], h["rho_max"]
    print(h.meta["velocity_frame"], len(t))

Rows that a restart has overwritten (time >= the first time after the restart) are dropped
unless keep_all=True; h.segments lists the header blocks (one per run or restart).
"""

import re

import numpy as np

COLUMNS = ("time", "dt", "n_particles", "mass", "mom1", "mom2", "mom3", "ekin",
           "z_rms", "rho_max", "rho_max_x1", "rho_max_x2", "rho_max_x3",
           "rho_max_level", "chi_max", "dt_stiff_min")
COLUMNS_GR = COLUMNS + ("g_groups", "r_removed")


class ParticleHistory(dict):
    """{column name: numpy array}; .meta: header facts of the last block; .comments: the
    comment lines of every block; .segments: (first row, row count) per header block."""


def read(path, keep_all=False):
    """Read a .phst file (see the module docstring)."""
    blocks = []          # (comments, names, rows)
    comments, names, rows = [], None, []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                if names is not None and rows:          # a new block after data
                    blocks.append((comments, names, rows))
                    comments, names, rows = [], None, []
                if line.startswith("# Athena++ history data"):
                    continue
                found = re.findall(r"\[(\d+)\]=(\S+)", line)
                if found:
                    names = [n for _, n in found]
                else:
                    comments.append(line.rstrip("\n"))
                continue
            if names is None or not line.strip():
                continue
            rows.append([float(v) for v in line.split()])
    if names is not None:
        blocks.append((comments, names, rows))
    if not blocks:
        raise ValueError(f"{path}: no phst header found")
    names = blocks[-1][1]
    for b in blocks:
        if b[1] != names:
            raise ValueError(f"{path}: header blocks with different columns")
    data, segments, all_comments = [], [], []
    for b in blocks:
        arr = np.array(b[2], dtype=float).reshape(-1, len(names))
        if not keep_all and data and len(arr) > 0:
            t0 = arr[0, 0]
            data = [d[d[:, 0] < t0] for d in data]
        data.append(arr)
        all_comments.extend(b[0])
    table = np.concatenate(data) if data else np.zeros((0, len(names)))
    start = 0
    for d in data:
        segments.append((start, len(d)))
        start += len(d)
    out = ParticleHistory()
    for k, n in enumerate(names):
        out[n] = table[:, k]
    out.comments = all_comments
    out.segments = segments
    meta = {}
    for c in blocks[-1][0]:
        for key in ("source_commit", "representation", "ndim", "nspecies",
                    "velocity_frame"):
            m = re.search(rf"{key} = ([^;:\s]+)", c)
            if m:
                meta[key] = m.group(1)
    out.meta = meta
    return out
