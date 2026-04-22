from __future__ import annotations

from collections import defaultdict

import numpy as np

from .types import PairMatch, Track, TrackObservation


class UnionFind:
    def __init__(self):
        self.p = {}

    def add(self, x):
        if x not in self.p:
            self.p[x] = x

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _key(view: int, line: float, samp: float, q: float = 1.0):
    return (view, int(round(line / q)), int(round(samp / q)))


def build_tracks(pair_matches: list[PairMatch], min_track_length: int = 2, q: float = 1.0) -> list[Track]:
    uf = UnionFind()
    node_data = {}

    for pm in pair_matches:
        for k in range(pm.matches.shape[0]):
            li, si, lj, sj = pm.matches[k].tolist()
            ni = _key(pm.view_i, li, si, q=q)
            nj = _key(pm.view_j, lj, sj, q=q)
            uf.add(ni); uf.add(nj)
            uf.union(ni, nj)
            node_data[ni] = (pm.view_i, li, si, 1.0 if pm.scores is None else float(pm.scores[k]))
            node_data[nj] = (pm.view_j, lj, sj, 1.0 if pm.scores is None else float(pm.scores[k]))

    groups = defaultdict(list)
    for n in node_data.keys():
        groups[uf.find(n)].append(n)

    tracks = []
    tid = 0
    for _, nodes in groups.items():
        by_view = defaultdict(list)
        for n in nodes:
            v, l, s, sc = node_data[n]
            by_view[v].append((l, s, sc))
        if len(by_view) < min_track_length:
            continue
        obs = []
        for v, arr in by_view.items():
            arr_sorted = sorted(arr, key=lambda x: -x[2])
            l, s, sc = arr_sorted[0]
            obs.append(TrackObservation(view_idx=int(v), line=float(l), samp=float(s), score=float(sc)))
        if len(obs) >= min_track_length:
            tracks.append(Track(track_id=tid, observations=sorted(obs, key=lambda x: x.view_idx)))
            tid += 1
    return tracks
