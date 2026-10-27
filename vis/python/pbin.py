"""
Reader for AthenaK particle science files (file_type = pbin, format version 1).

A pbin file holds the active dust particles of one output time as typed columns
(float64 positions, velocities, masses and stopping times, uint64 ids, uint32
species, and for refinable clouds a uint16 shape code), a small directory of the
MeshBlocks, and a text header with the run's complete input parameters, the
representation rule of the dust mass, the velocity frame and the column
directory.  The file can be read alone (without any code of the dust
project).  Only numpy is needed.

Example::

    import pbin
    snap = pbin.read("pbin/run.particles.00042.pbin")
    print(snap.time, snap.cycle, len(snap), snap.representation)
    x, v, m = snap.position, snap.velocity, snap.mass      # (N,3), (N,3), (N,)
    v_phys = snap.physical_velocity()     # adds the background shear if stored relative
    if snap.has_shape:                    # refinable clouds
        lev, lobes = snap.cloud_level, snap.lobes          # (N,), (N,3) in 0..3
        ell = snap.cloud_half_width()                      # (N,3) code length
        lo, hi = snap.cloud_support()                      # (N,3) each
    s1 = snap.select_species(1)           # species_id 1 (input index 2)
    print(pbin.check(snap))               # [] when every check passes
    gid = snap.owner_gid()                # owner MeshBlock of every particle

    # pairing with a mesh .bin file of the same output time
    import bin_convert
    mesh = bin_convert.read_binary("bin/run.dust_total.00042.bin")
    pbin.check_pair(snap, mesh)           # time, cycle and block topology must match

Only some columns can be read: ``pbin.read(path, columns=["x1", "x2", "mass"])``.
"""

import os

import numpy as np

MAGIC = b"AKPBIN01"
END = b"AKPBEND1"
PREAMBLE = 64
FOOTER = 16
SUPPORTED_VERSIONS = (1,)
SHAPE_VERSION = 1


class PbinError(Exception):
    """Raised for files that are not complete, consistent pbin files."""


def _parse_text(text):
    """Split the text header into {section: {key: value}}, the column directory and the
    raw input-parameter dump."""
    sections = {}
    columns = []
    current = None
    head, sep, dump = text.partition("\n[input]\n")
    if not sep:
        raise PbinError("text header has no [input] section")
    for line in head.split("\n")[1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("[") and line.rstrip().endswith("]"):
            current = line.strip()[1:-1]
            sections.setdefault(current, {})
            continue
        if current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if current == "columns" and key == "column":
            t = value.split()
            columns.append(dict(table=t[0], name=t[1], dtype=t[2], count=int(t[3]),
                                offset=int(t[4]), nbytes=int(t[5])))
            continue
        # drop trailing comments "  # ..." (values themselves contain no " # ")
        value = value.split("  #")[0].strip()
        sections[current][key] = value
    return sections, columns, dump


def parse_input(dump):
    """The parameter dump ("<block>" lines, "key = value # comment") as
    {block: {key: value-string}}."""
    params = {}
    block = None
    for line in dump.split("\n"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            block = line[1:-1]
            if block == "par_end":
                break
            params.setdefault(block, {})
            continue
        if block is not None and "=" in line:
            key, value = line.split("=", 1)
            params[block][key.strip()] = value.strip()
    return params


def read_header(path):
    """Header of a pbin file: (preamble dict, sections, column directory, input dump).
    Checks the magic, the byte-order mark, the format version, the footer and the
    file size."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        pre = f.read(PREAMBLE)
        if len(pre) < PREAMBLE or pre[:8] != MAGIC:
            raise PbinError(f"{path}: not a pbin file (magic)")
        mark = int.from_bytes(pre[8:12], "little")
        if mark == 0x01020304:
            bo = "<"
        elif mark == 0x04030201:
            bo = ">"
        else:
            raise PbinError(f"{path}: bad byte-order mark")
        version = int(np.frombuffer(pre[12:16], bo + "u4")[0])
        if version not in SUPPORTED_VERSIONS:
            raise PbinError(f"{path}: format version {version} not supported")
        vals = np.frombuffer(pre[16:64], bo + "u8")
        text_bytes, data_start, data_bytes, n_particles, n_blocks = (int(v) for v in vals[:5])
        f.seek(data_start + data_bytes)
        foot = f.read(FOOTER)
        if len(foot) < FOOTER or foot[:8] != END:
            raise PbinError(f"{path}: incomplete file (no footer)")
        total = int(np.frombuffer(foot[8:16], bo + "u8")[0])
        if total != size:
            raise PbinError(f"{path}: size {size} differs from the recorded {total}")
        f.seek(PREAMBLE)
        text = f.read(text_bytes).decode("utf-8", errors="replace")
    sections, columns, dump = _parse_text(text)
    pre = dict(version=version, byte_order=bo, text_bytes=text_bytes,
               data_start=data_start, data_bytes=data_bytes,
               n_particles=n_particles, n_blocks=n_blocks, file_size=size)
    for c in columns:
        if c["offset"] + c["nbytes"] > data_bytes:
            raise PbinError(f"{path}: column {c['name']} runs past the data section")
        if c["nbytes"] != c["count"] * np.dtype(c["dtype"]).itemsize:
            raise PbinError(f"{path}: column {c['name']} has an inconsistent size")
    return pre, sections, columns, dump


class PbinSnapshot:
    """The contents of one pbin file.  Columns are numpy arrays in ``columns`` (particle
    table) and ``blocks`` (MeshBlock table); header sections in ``header``; the input
    parameters in ``params`` ({block: {key: str}}) and raw in ``input_dump``."""

    def __init__(self, path, pre, header, coldir, dump, columns, blocks):
        self.path = path
        self.preamble = pre
        self.header = header
        self.column_directory = coldir
        self.input_dump = dump
        self.params = parse_input(dump)
        self.columns = columns
        self.blocks = blocks

    # ---- basic metadata
    def __len__(self):
        return self.preamble["n_particles"]

    @property
    def time(self):
        return float(self.header["snapshot"]["time"])

    @property
    def cycle(self):
        return int(self.header["snapshot"]["cycle"])

    @property
    def ndim(self):
        return int(self.header["mesh"]["ndim"])

    @property
    def representation(self):
        return self.header["representation"]["representation"]

    @property
    def velocity_frame(self):
        return self.header["frame"]["velocity_frame"]

    def _col(self, name):
        if name not in self.columns:
            raise KeyError(f"column {name!r} was not read (or is not in the file)")
        return self.columns[name]

    # ---- particle columns
    @property
    def id(self):
        return self._col("id")

    @property
    def species(self):
        return self._col("species")

    @property
    def mass(self):
        return self._col("mass")

    @property
    def stopping_time(self):
        return self._col("stopping_time")

    @property
    def position(self):
        return np.stack([self._col("x1"), self._col("x2"), self._col("x3")], axis=1)

    @property
    def velocity(self):
        """Velocities as stored (see ``velocity_frame``)."""
        return np.stack([self._col("v1"), self._col("v2"), self._col("v3")], axis=1)

    def physical_velocity(self):
        """Velocities in the lab frame: in a shearing box the stored azimuthal component
        is relative to the background shear -q Omega x1 (v2 in 3-D, v3 in the 2-D r-z
        box); this adds it back."""
        v = self.velocity.copy()
        fr = self.header["frame"]
        if fr["velocity_frame"] == "shear_relative":
            q = float(fr["shear_q"])
            om = float(fr["shear_omega"])
            comp = 1 if self.ndim == 3 else 2
            v[:, comp] -= q * om * self._col("x1")
        return v

    # ---- refinable clouds
    @property
    def has_shape(self):
        return "shape" in self.columns

    @property
    def cloud_level(self):
        """Cloud level (int): the hat half-width is root_dx / 2**level per direction."""
        return (self._col("shape") & 15).astype(np.int32)

    @property
    def lobes(self):
        """(N,3) lobe code along x1, x2, x3: 0 none, 1 lower, 2 upper, 3 both."""
        s = self._col("shape").astype(np.int32)
        return np.stack([(s >> 4) & 3, (s >> 6) & 3, (s >> 8) & 3], axis=1)

    def root_dx(self):
        r = self.header["representation"]
        return np.array([float(r["root_dx1"]), float(r["root_dx2"]), float(r["root_dx3"])])

    def cloud_half_width(self):
        """(N,3) half-width ell of the hat along each direction, root_dx * 2**-level (the
        rule in the header; a 2-D run has no cloud shape along x3: that column is nan)."""
        if self.representation != "refinable_hat":
            raise ValueError("not a refinable-cloud run (representation "
                             f"{self.representation})")
        ell = self.root_dx()[None, :] * np.ldexp(1.0, -self.cloud_level)[:, None]
        if self.ndim == 2:
            ell[:, 2] = np.nan
        return ell

    def cloud_width(self):
        """(N,3) full support width of every cloud: 2 ell without a lobe, 3 ell with a lower
        or upper lobe, 4 ell with both."""
        lo, hi = self.cloud_support()
        return hi - lo

    def cloud_support(self):
        """(lo, hi), each (N,3): the support of every cloud along each direction (lobe 0
        [x - ell, x + ell], 1 [x - 2 ell, x + ell], 2 [x - ell, x + 2 ell], 3 [x - 2 ell,
        x + 2 ell])."""
        ell = self.cloud_half_width()
        lob = self.lobes
        x = self.position
        lo = x - np.where((lob == 1) | (lob == 3), 2.0, 1.0) * ell
        hi = x + np.where((lob == 2) | (lob == 3), 2.0, 1.0) * ell
        return lo, hi

    # ---- selections
    def select(self, mask):
        """A new snapshot with the particle rows where mask is true (blocks unchanged)."""
        mask = np.asarray(mask)
        cols = {k: v[mask] for k, v in self.columns.items()}
        out = PbinSnapshot(self.path, dict(self.preamble), self.header,
                           self.column_directory, self.input_dump, cols, self.blocks)
        out.preamble["n_particles"] = int(np.count_nonzero(mask)) if mask.dtype == bool \
            else len(next(iter(cols.values())))
        return out

    def select_species(self, species_id):
        """Rows of one species (0-based species_id = input index - 1)."""
        return self.select(self.species == species_id)

    # ---- MeshBlocks
    def topology_hash(self):
        b = self.blocks
        return topology_hash(b["level"], b["lx1"], b["lx2"], b["lx3"])

    def owner_gid(self):
        """gid of the MeshBlock whose half-open box [min, max) holds each particle (-1 if
        none): searched level by level on the logical locations."""
        b = self.blocks
        m = self.header["mesh"]
        nmb_root = [int(m["nx1"]) // int(m["meshblock_nx1"]),
                    int(m["nx2"]) // int(m["meshblock_nx2"]),
                    int(m["nx3"]) // int(m["meshblock_nx3"])]
        xmin = [float(m["x1min"]), float(m["x2min"]), float(m["x3min"])]
        xmax = [float(m["x1max"]), float(m["x2max"]), float(m["x3max"])]
        x = self.position
        owner = np.full(len(self), -1, np.int64)
        for lev in np.unique(b["level"]):
            sel = np.nonzero(b["level"] == lev)[0]
            n = [nr << int(lev) for nr in nmb_root]
            key_b = ((b["lx3"][sel] * n[1] + b["lx2"][sel]) * n[0] + b["lx1"][sel])
            order = np.argsort(key_b)
            key_b, gid_b = key_b[order], b["gid"][sel][order]
            lx = []
            for d in range(3):
                if self.ndim == 2 and d == 2:
                    lx.append(np.zeros(len(self), np.int64))
                    continue
                w = (xmax[d] - xmin[d]) / n[d]
                lx.append(np.clip(np.floor((x[:, d] - xmin[d]) / w).astype(np.int64),
                                  0, n[d] - 1))
            key_p = (lx[2] * n[1] + lx[1]) * n[0] + lx[0]
            pos = np.searchsorted(key_b, key_p)
            pos = np.minimum(pos, len(key_b) - 1)
            hit = (key_b[pos] == key_p) & (owner < 0)
            owner[hit] = gid_b[pos[hit]]
        return owner


def topology_hash(level, lx1, lx2, lx3):
    """FNV-1a (64 bit) over (level, lx1, lx2, lx3) as little-endian int64 for every block
    in gid order: the block_topology_hash of the pbin header."""
    h = 1469598103934665603
    mask = (1 << 64) - 1
    arr = np.stack([np.asarray(level, np.int64), np.asarray(lx1, np.int64),
                    np.asarray(lx2, np.int64), np.asarray(lx3, np.int64)], axis=1)
    for byte in arr.astype("<i8").tobytes():
        h ^= byte
        h = (h * 1099511628211) & mask
    return h


def read(path, columns=None):
    """Read a pbin file.  columns: names of particle columns to read (default all); the
    block table is always read."""
    pre, sections, coldir, dump = read_header(path)
    cols, blocks = {}, {}
    with open(path, "rb") as f:
        for c in coldir:
            if c["table"] == "particles" and columns is not None and c["name"] not in columns:
                continue
            f.seek(pre["data_start"] + c["offset"])
            arr = np.fromfile(f, dtype=np.dtype(c["dtype"]), count=c["count"])
            if arr.size != c["count"]:
                raise PbinError(f"{path}: column {c['name']} is truncated")
            (cols if c["table"] == "particles" else blocks)[c["name"]] = arr
    if columns is not None:
        missing = set(columns) - set(cols)
        if missing:
            raise KeyError(f"columns not in the file: {sorted(missing)}")
    return PbinSnapshot(path, pre, sections, coldir, dump, cols, blocks)


def check(snap, raise_on_error=False):
    """Consistency checks of a snapshot; returns a list of problems (empty: all passed).
    Checks: counts, finite positive masses and stopping times, species in range, positions
    inside the domain and inside a block, shape bits (clouds), unique ids, the block
    directory (count, gids) and the header's anomaly counters."""
    problems = []
    n = len(snap)
    for k, v in snap.columns.items():
        if len(v) != n:
            problems.append(f"column {k}: {len(v)} rows, header says {n}")
    hdr = snap.header
    nb = int(hdr["file"]["n_blocks"])
    if len(snap.blocks.get("gid", [])) != nb:
        problems.append("block table size differs from n_blocks")
    elif not np.array_equal(snap.blocks["gid"], np.arange(nb)):
        problems.append("block gids are not 0..n_blocks-1 in order")
    elif int(hdr["snapshot"]["block_topology_hash"], 16) != snap.topology_hash():
        problems.append("block table does not match block_topology_hash")
    c = snap.columns
    if "mass" in c and not (np.all(np.isfinite(c["mass"])) and np.all(c["mass"] > 0)):
        problems.append("non-finite or non-positive masses")
    if "stopping_time" in c and not np.all(np.isfinite(c["stopping_time"]) &
                                           (c["stopping_time"] > 0)):
        problems.append("non-finite or non-positive stopping times")
    if "species" in c and n > 0:
        ns = int(hdr["species"]["nspecies"])
        if c["species"].max() >= ns:
            problems.append("species_id beyond nspecies")
    if all(k in c for k in ("x1", "x2", "x3")) and n > 0:
        m = hdr["mesh"]
        for d in range(snap.ndim):
            x = c[f"x{d+1}"]
            lo, hi = float(m[f"x{d+1}min"]), float(m[f"x{d+1}max"])
            # the code may store a position a round-off below a periodic face (e.g. -1e-28 after
            # a merge or resampling at x = 0): tolerate 1e-12 of the domain length
            tol = 1e-12 * (hi - lo)
            if not (np.all(np.isfinite(x)) and x.min() >= lo - tol and x.max() < hi + tol):
                problems.append(f"positions outside [x{d+1}min, x{d+1}max)")
        if np.any(snap.owner_gid() < 0):
            problems.append("particles in no MeshBlock")
    if "id" in c and n > 0 and len(np.unique(c["id"])) != n:
        problems.append(f"{n - len(np.unique(c['id']))} duplicate ids")
    if "shape" in c and n > 0:
        s = c["shape"].astype(np.int64)
        if np.any(s >> 10):
            problems.append("shape codes with bits above 9")
        if snap.ndim == 2 and np.any((s >> 8) & 3):
            problems.append("x3 lobes in a 2-D run")
    p = hdr["particles"]
    for key in ("invalid_mass_rows", "negative_tag_rows", "shape_unknown_bit_rows"):
        if key in p and int(p[key]) != 0:
            problems.append(f"header: {key} = {p[key]}")
    if raise_on_error and problems:
        raise PbinError("; ".join(problems))
    return problems


def check_pair(snap, mesh, raise_on_error=True):
    """Check that a pbin snapshot and a mesh .bin file (bin_convert.read_binary dict) are
    the same output time on the same block topology.  Returns the list of problems."""
    problems = []
    if mesh["cycle"] != snap.cycle:
        problems.append(f"cycle {mesh['cycle']} != {snap.cycle}")
    if not np.isclose(mesh["time"], snap.time, rtol=1e-14, atol=0.0):
        problems.append(f"time {mesh['time']} != {snap.time}")
    lg = np.asarray(mesh["mb_logical"])   # lx1, lx2, lx3, level per block, gid order
    if len(lg) != len(snap.blocks["gid"]):
        problems.append("different numbers of MeshBlocks (a sliced mesh file?)")
    else:
        h = topology_hash(lg[:, 3], lg[:, 0], lg[:, 1], lg[:, 2])
        if h != snap.topology_hash():
            problems.append("different block topology")
    if raise_on_error and problems:
        raise PbinError("; ".join(problems))
    return problems
