"""Oracle replays of FAST_SBM through the host build: KC-1 (warm), KC-2
(cold, hail and graupel builds) and the low-edge KC-2 tile.  Each replay
record holds the full pre-call state WRF's Fortran received and the
post-call state it returned; ``replay_call`` rebuilds the call in WRF's
memory layout, runs the host build on it and returns both."""

from __future__ import annotations

import numpy as np

from tools.fast_sbm_wrf480_oracle import fixtures as fx
from tools.fast_sbm_host import host as H

POST_FIELDS = ("th", "qv", "th_old", "qv_old", "chem", "qc", "qr", "qi", "qs", "qg",
               "qnc", "qnr", "qni", "qns", "qng", "qna", "rainnc", "rainncv", "snownc",
               "snowncv", "graupelnc", "graupelncv")


def build_tag(build: str) -> str:
    return "f64" if build == "o64" else "f32"


def host_for(build: str, flags: dict) -> H.Host:
    tag = build_tag(build)
    dtype = np.float64 if tag == "f64" else np.float32
    t = fx.load_fixture(f"tables_{build}")
    read = {k.split(".", 1)[1]: v[0] for k, v in t.items() if k.startswith("read.")}
    return H.Host(H.tables_from_read(read, dtype), flags, tag)


class Case:
    def __init__(self, family: str, build: str):
        self.family = family
        self.build = build
        self.dtype = np.float64 if build == "o64" else np.float32
        if family == "kc1":
            self.tile = H.KC1_TILE
            c = fx.load_fixture("kc1_case")
            self.c = {k.split(".", 1)[1]: v[0] for k, v in c.items() if k.startswith("case.")}
        else:
            self.tile = H.kc2_tile(edge=(family == "kc2edge"))
            if family == "kc2edge":
                e = fx.load_fixture(f"kc2edge_replay_{build}")
                self.c = {k.split(".", 1)[1]: v[0] for k, v in e.items() if k.startswith("initial.")}
            else:
                c = fx.load_fixture("kc2_case")
                self.c = {k.split(".", 1)[1]: v[0] for k, v in c.items() if k.startswith("case.")}

    def _r(self, x):
        return np.asarray(x, dtype=np.float64).astype(self.dtype)

    def state(self, pre) -> dict:
        t = self.tile
        s = t.blank(self.dtype)
        c = self.c
        kte = t.kte
        if self.family == "kc1":
            for col in range(4):
                i = t.i(col + 1)
                j = t.j(1)
                s["xland"][i, j] = self._r(c["xland"][col])
                for k in range(kte):
                    s["dz8w"][i, k, j] = self._r(c["dz"][k])
                    s["rho"][i, k, j] = self._r(c["rho"][col, k])
                    s["p"][i, k, j] = self._r(c["pressure"][col, k])
                    s["pii"][i, k, j] = self._r(c["exner"][col, k])
                s["dz8w"][i, t.k(t.kme), j] = s["dz8w"][i, t.k(kte), j]
            ii = slice(t.i(1), t.i(4) + 1)
            jj = t.j(1)
            s["w"][ii, :, jj] = pre["w"]
            for f in ("th", "qv", "th_old", "qv_old"):
                s[f][ii, :kte, jj] = pre[f]
            s["chem"][ii, :kte, jj, 1:] = pre["chem"]
            for f in ("rainnc", "snownc", "graupelnc"):
                s[f][ii, jj] = pre[f]
            return s
        edge = self.family == "kc2edge"
        for di in range(4):
            for dj in range(4):
                i, j = t.i(di + 1), t.j(dj + 1)
                s["xland"][i, j] = self._r(c["xland"][di, dj])
                for k in range(kte):
                    if edge:
                        s["dz8w"][i, k, j] = self._r(c["dz8w"][di, k, dj])
                        for f in ("rho", "p", "pii", "u", "v"):
                            s[f][i, k, j] = self._r(c[f][di, k, dj])
                    else:
                        s["dz8w"][i, k, j] = self._r(c["dz"][k])
                        s["rho"][i, k, j] = self._r(c["rho"][di, dj, k])
                        s["p"][i, k, j] = self._r(c["pressure"][di, dj, k])
                        s["pii"][i, k, j] = self._r(c["exner"][di, dj, k])
                        s["u"][i, k, j] = self._r(c["u"][di, dj, k])
                        s["v"][i, k, j] = self._r(c["v"][di, dj, k])
                kme = t.k(t.kme)
                s["dz8w"][i, kme, j] = s["dz8w"][i, t.k(kte), j]
                s["u"][i, kme, j] = s["u"][i, t.k(kte), j]
                s["v"][i, kme, j] = s["v"][i, t.k(kte), j]
        dom = (slice(t.i(1), t.i(4) + 1), slice(None, kte), slice(t.j(1), t.j(4) + 1))
        for f in ("th", "qv", "th_old", "qv_old"):
            s[f][dom] = pre[f]
        ti = slice(t.i(t.its), t.i(t.ite) + 1)
        tj = slice(t.j(t.jts), t.j(t.jte) + 1)
        s["w"][ti, :, tj] = pre["w"]
        s["chem"][ti, :kte, tj, 1:] = pre["chem"]
        for f in ("rainnc", "snownc", "graupelnc"):
            s[f][ti, tj] = pre[f]
        return s

    def tile_view(self, s: dict, field: str):
        t = self.tile
        kte = t.kte
        ti = slice(t.i(t.its), t.i(t.ite) + 1)
        tj = slice(t.j(t.jts), t.j(t.jte) + 1)
        a = s[field]
        if field in ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv"):
            return a[ti, tj] if self.family != "kc1" else a[ti, t.j(1)]
        if field == "chem":
            return a[ti, :kte, tj, 1:] if self.family != "kc1" else a[ti, :kte, t.j(1), 1:]
        return a[ti, :kte, tj] if self.family != "kc1" else a[ti, :kte, t.j(1)]


def replay_name(family: str, build: str) -> str:
    return f"{family}_replay_{build}"


def run_replays(family: str, build: str, flags: dict | None = None, host=None):
    """Yield (step, status, tally, got, want) per replay record."""
    if flags is None:
        flags = H.flags_wrf(hail=0 if family == "kc2g" else 1)
    if host is None:
        host = host_for(build, flags)
    case = Case(family, build)
    for step, pre, post in fx.iter_replay(replay_name(family, build)):
        s = case.state(pre.fields)
        status, tally = host.call(step, 12.0, 1000.0, 1000.0, case.tile.dims(), s)
        got = {f: np.array(case.tile_view(s, f)) for f in POST_FIELDS}
        want = {f: np.asarray(post.fields[f]) for f in POST_FIELDS}
        yield step, status, tally, got, want


def compare(got: dict, want: dict) -> dict:
    """Per field: (bitwise-identical count, total, max relative difference
    with a 1e-30 floor on the scale)."""
    out = {}
    for f in POST_FIELDS:
        g = np.asarray(got[f], dtype=np.float64)
        w = np.asarray(want[f], dtype=np.float64)
        same = int(np.sum(np.asarray(got[f]) == np.asarray(want[f])))
        scale = max(float(np.max(np.abs(w))), 1e-30)
        rel = float(np.max(np.abs(g - w)) / scale) if g.size else 0.0
        out[f] = (same, g.size, rel)
    return out
