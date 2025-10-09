from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os, json, re
from pyscipopt import Sepa, SCIP_RESULT

_PATTERNS = [
    (r'^(?P<kind>[uxsvwz])\[(?P<u>\d+)\s*,\s*(?P<t>\d+)\]$', "brackets"),
    (r'^(?P<kind>[uxsvwz])\((?P<u>\d+)\s*,\s*(?P<t>\d+)\)$', "parens"),
    (r'^(?P<kind>[uxsvwz])_u(?P<u>\d+)_t(?P<t>\d+)$',        "underscores"),
]
_KINDMAP = {'u':'x', 'x':'x', 'v':'s', 's':'s', 'w':'z', 'z':'z'}

class MinGenCoverSepa(Sepa):
    """
    Anti-overcommit: if sum_{i in S} Pmin_i > D_t, then sum_{i in S} y_{i,t} <= |S| - 1.
    """
    def __init__(self, json_path: Optional[str] = None, minvio: float = 1e-4, rootonly: bool=False, maxcuts_per_t: int = 1):
        self.json_path = json_path
        self.minvio = float(minvio)
        self.rootonly = bool(rootonly)
        self.maxcuts_per_t = int(maxcuts_per_t)
        #
        self._inited = False
        self.x: Dict[Tuple[int,int], object] = {}
        self.units: List[int] = []
        self.times: List[int] = []
        self.Pmin: Dict[int, float] = {}
        self.D: List[float] = []
        # stats
        self.ncuts_added = 0

    def _parse(self, name: str):
        for pat,_ in _PATTERNS:
            m = re.match(pat, name)
            if m:
                k = _KINDMAP.get(m.group('kind'), m.group('kind'))
                return k, int(m.group('u')), int(m.group('t'))
        return None

    def _lazy_init(self):
        if self._inited: return
        scip = self.model
        U, T = set(), set()
        for v in scip.getVars(transformed=True):
            p = self._parse(v.name)
            if not p: continue
            k,u,t = p
            if k == 'x':
                self.x[(u,t)] = v
                U.add(u); T.add(t)
        self.units = sorted(U)
        self.times = sorted(T)

        if self.json_path and os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                meta = json.load(f)
            Pmin = meta.get("Pmin", {})
            Demand = meta.get("Demand", [])
            if isinstance(Pmin, dict):
                for k,v in Pmin.items():
                    try: self.Pmin[int(k)] = float(v)
                    except: pass
            elif isinstance(Pmin, list):
                for idx,v in enumerate(Pmin):
                    if idx < len(self.units):
                        self.Pmin[self.units[idx]] = float(v)
            self.D = [float(x) for x in Demand] if Demand else []

        self._inited = True

    def _violating_S(self, t: int) -> Optional[List[int]]:
        if not self.D or t >= len(self.D): return None
        need = self.D[t]
        if len(self.Pmin) < max(1, int(0.8*len(self.units))):
            return None
        # sort by Pmin descending and keep adding until sum Pmin exceeds D_t
        u_sorted = sorted(self.units, key=lambda u: self.Pmin.get(u, 0.0), reverse=True)
        S, smin = [], 0.0
        for u in u_sorted:
            smin += self.Pmin.get(u, 0.0)
            S.append(u)
            if smin > need:
                return S
        return None

    def _try_add(self, t: int) -> Optional[int]:
        S = self._violating_S(t)
        if not S: return None
        scip = self.model
        rhs = float(len(S) - 1)
        row = scip.createEmptyRowSepa(self, f"mingen_t{t}", -scip.infinity(), rhs,
                                      local=False, modifiable=False, removable=True)
        s = 0.0
        for u in S:
            var = self.x.get((u,t))
            if var is None: continue
            scip.addVarToRow(row, var, 1.0)
            s += scip.getSolVal(None, var)
        # violation if s - rhs > minvio
        if s - rhs < self.minvio or not scip.isCutEfficacious(row):
            scip.releaseRow(row); return None
        scip.addCut(row)
        self.ncuts_added += 1
        scip.releaseRow(row)
        return SCIP_RESULT.SEPARATED

    def sepaexeclp(self):
        scip = self.model
        if self.rootonly and scip.getDepth() > 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        if scip.getLPSolstat() != "optimal":
            return {"result": SCIP_RESULT.DIDNOTRUN}

        self._lazy_init()
        if not self.units or not self.times or not self.D:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        result = SCIP_RESULT.DIDNOTFIND
        for t in self.times:
            r = self._try_add(t)
            if r == SCIP_RESULT.SEPARATED: result = r
        return {"result": result}
