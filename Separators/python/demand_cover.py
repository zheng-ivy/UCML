from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os, json, re
from pyscipopt import Sepa, SCIP_RESULT

# accept u/v/w and x/s/z; we only need commitment (u/x)
_PATTERNS = [
    (r'^(?P<kind>[uxsvwz])\[(?P<u>\d+)\s*,\s*(?P<t>\d+)\]$', "brackets"),
    (r'^(?P<kind>[uxsvwz])\((?P<u>\d+)\s*,\s*(?P<t>\d+)\)$', "parens"),
    (r'^(?P<kind>[uxsvwz])_u(?P<u>\d+)_t(?P<t>\d+)$',        "underscores"),
]
_KINDMAP = {'u':'x', 'x':'x', 'v':'s', 's':'s', 'w':'z', 'z':'z'}

class DemandReserveCoverSepa(Sepa):
    """
    Single-period cover: if remaining capacity without C can't meet D+R,
    then sum_{i in C} y_{i,t} >= 1.
    """
    def __init__(self, json_path: Optional[str] = None, minvio: float = 1e-4, rootonly: bool=False, maxcuts_per_t: int = 1):
        self.json_path = json_path
        self.minvio = float(minvio)
        self.rootonly = bool(rootonly)
        self.maxcuts_per_t = int(maxcuts_per_t)
        # discovered
        self._inited = False
        self.x: Dict[Tuple[int,int], object] = {}
        self.units: List[int] = []
        self.times: List[int] = []
        # data
        self.Pmax: Dict[int, float] = {}
        self.D: List[float] = []
        self.R: List[float] = []
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

        # sidecar
        if self.json_path and os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                meta = json.load(f)
            # accept dicts keyed by unit id (string or int)
            Pmax = meta.get("Pmax", {})
            Demand = meta.get("Demand", [])
            Reserve = meta.get("Reserve", None)
            if isinstance(Pmax, dict):
                for k,v in Pmax.items():
                    try: self.Pmax[int(k)] = float(v)
                    except: pass
            elif isinstance(Pmax, list):
                for idx,v in enumerate(Pmax):
                    if idx < len(self.units):
                        self.Pmax[self.units[idx]] = float(v)
            # time series
            self.D = [float(x) for x in Demand] if Demand else []
            if Reserve is None:
                self.R = [0.0]*(len(self.D) if self.D else (self.times[-1]+1 if self.times else 0))
            else:
                self.R = [float(x) for x in Reserve]

        self._inited = True

    def _cover_set(self, t: int) -> Optional[List[int]]:
        """Greedy: remove smallest Pmax until remaining cap < D+R; return removed set C."""
        if not self.D or t >= len(self.D): return None
        need = self.D[t] + (self.R[t] if t < len(self.R) else 0.0)
        # if we don't have Pmax for most units, skip
        if len(self.Pmax) < max(1, int(0.8*len(self.units))):
            return None
        # sort units by Pmax ascending
        u_sorted = sorted(self.units, key=lambda u: self.Pmax.get(u, 0.0))
        total = sum(self.Pmax.get(u, 0.0) for u in self.units)
        if total < need:  # globally infeasible; skip (let model prove it)
            return None
        C = []
        remaining = total
        for u in u_sorted:
            remaining -= self.Pmax.get(u, 0.0)
            C.append(u)
            if remaining < need:
                return C
        return None

    def _try_add_cover(self, t: int) -> Optional[int]:
        C = self._cover_set(t)
        if not C: return None
        scip = self.model
        row = scip.createEmptyRowSepa(self, f"demcov_t{t}", 1.0, scip.infinity(),
                                      local=False, modifiable=False, removable=True)
        lhs_value = 0.0
        for u in C:
            var = self.x.get((u,t))
            if var is None: continue
            scip.addVarToRow(row, var, 1.0)
            lhs_value += scip.getSolVal(None, var)  # LP val
        # violation: 1 - sum x < 0? add if violated
        if 1.0 - lhs_value < self.minvio or not scip.isCutEfficacious(row):
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
            cuts_here = 0
            r = self._try_add_cover(t)
            if r == SCIP_RESULT.SEPARATED:
                result = r; cuts_here += 1
            if cuts_here >= self.maxcuts_per_t:
                continue
        return {"result": result}
