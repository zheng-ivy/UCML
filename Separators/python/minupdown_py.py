# separators/python/minupdown_py.py
from __future__ import annotations
import re, json, os
from typing import Dict, Tuple, Optional, List
from pyscipopt import Sepa, SCIP_RESULT

# Accept both Pyomo-ish names and custom ones:
#   u[3,17], v[3,17], w[3,17]
#   u(3,17), v(3,17), w(3,17)
#   u_u3_t17, v_u3_t17, w_u3_t17
# Also allow x/s/z as aliases (commit/startup/shutdown).
_NAME_PATTERNS = [
    (r'^(?P<kind>[uxsvwz])\[(?P<u>\d+)\s*,\s*(?P<t>\d+)\]$', "brackets"),
    (r'^(?P<kind>[uxsvwz])\((?P<u>\d+)\s*,\s*(?P<t>\d+)\)$', "parens"),
    (r'^(?P<kind>[uxsvwz])_u(?P<u>\d+)_t(?P<t>\d+)$',        "underscores"),
]
_KIND_MAP = {'u':'x', 'v':'s', 'w':'z', 'x':'x', 's':'s', 'z':'z'}  # normalize

class MinUpDownSepa(Sepa):
    """
    Min up/down separator for UC:
      sum_{tau=t}^{t+L-1} x[u,tau] - L*s[u,t] >= 0
      sum_{tau=t}^{t+L-1} x[u,tau] + L*z[u,t] <= L

    If explicit s/z vars are missing, use delta-x fallback:
      s[u,t] = x[u,t] - x[u,t-1]; z[u,t] = x[u,t-1] - x[u,t]
    """

    def __init__(self, minviolation: float = 1e-3, rootonly: bool = False, json_path: Optional[str] = None):
        self.minviolation = float(minviolation)
        self.rootonly = bool(rootonly)
        self.json_path = json_path

        # discovered on first call
        self._inited = False
        self.x: Dict[Tuple[int,int], object] = {}
        self.s: Dict[Tuple[int,int], object] = {}
        self.z: Dict[Tuple[int,int], object] = {}
        self.units: List[int] = []
        self.periods: List[int] = []
        self.Lup: Dict[int,int] = {}
        self.Ldown: Dict[int,int] = {}
        self.maxcuts_per_round = int(maxcuts_per_round)
        self._seen_keys = set()   # across the whole solve (simple and effective)

        # stats
        self.ncuts_added = 0

    # ---------- helpers ----------
    def _parse_varname(self, name: str):
        for pat, _ in _NAME_PATTERNS:
            m = re.match(pat, name)
            if m:
                k = _KIND_MAP.get(m.group('kind'), m.group('kind'))
                return k, int(m.group('u')), int(m.group('t'))
        return None
    def _minup_key(self, u, t, L):   # include L because near the horizon L changes
        return ("up", u, t, L)
    def _mindown_key(self, u, t, L):
        return ("down", u, t, L)
    def _lazy_init(self):
        if self._inited:
            return
        scip = self.model

        # discover transformed variables
        uids, tids = set(), set()
        for var in scip.getVars(transformed=True):
            parsed = self._parse_varname(var.name)
            if not parsed:
                continue
            k, u, t = parsed
            if k == 'x':
                self.x[(u,t)] = var
            elif k == 's':
                self.s[(u,t)] = var
            elif k == 'z':
                self.z[(u,t)] = var
            uids.add(u); tids.add(t)

        self.units = sorted(uids)
        self.periods = sorted(tids)

        # load Lup/Ldown from sidecar if provided
        meta = None
        if self.json_path and os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r") as f: meta = json.load(f)
            except Exception:
                meta = None

        if meta:
            Lup = meta.get("Lup", {})
            Ldown = meta.get("Ldown", {})
            if isinstance(Lup, dict):
                for k,v in Lup.items(): self.Lup[int(k)] = int(v)
            else:
                for idx,v in enumerate(Lup): self.Lup[self.units[idx] if idx < len(self.units) else idx] = int(v)
            if isinstance(Ldown, dict):
                for k,v in Ldown.items(): self.Ldown[int(k)] = int(v)
            else:
                for idx,v in enumerate(Ldown): self.Ldown[self.units[idx] if idx < len(self.units) else idx] = int(v)

        # fallback if missing
        for u in self.units:
            self.Lup.setdefault(u,   2)
            self.Ldown.setdefault(u, 2)

        self._inited = True

    def _add_minup_cut(self, u: int, t: int) -> Optional[int]:
        scip = self.model
        # robust to sparse periods (works for Ldown too)
        rem = sum(1 for p in self.periods if p >= t)
        L = min(self.Lup.get(u, 1), rem)
        if L <= 1:
            return None

        key = self._minup_key(u, t, L)
        if key in self._seen_keys:
            return None

        row = scip.createEmptyRowSepa(self, f"minup_{u}_{t}", 0.0, scip.infinity(),
                                      local=False, modifiable=False, removable=True)

        for tau in range(t, t+L):
            v = self.x.get((u,tau))
            if v is not None:
                scip.addVarToRow(row, v, 1.0)

        if (u,t) in self.s:
            scip.addVarToRow(row, self.s[(u,t)], -float(L))
        else:
            xt = self.x.get((u,t))
            if xt is not None: scip.addVarToRow(row, xt, -float(L))
            if (u,t-1) in self.x and t-1 in self.periods:
                scip.addVarToRow(row, self.x[(u,t-1)], float(L))

        scip.flushRowExtensions(row)
        activity = scip.getRowSolActivity(row)
        viol = max(0.0, 0.0 - activity)
        if viol < self.minviolation or not scip.isCutEfficacious(row):
            scip.releaseRow(row)
            return None

        scip.addCut(row)   # add as global cut
        self.ncuts_added += 1
        self._seen_keys.add(key)

        scip.releaseRow(row)
        return SCIP_RESULT.SEPARATED

    def _add_mindown_cut(self, u: int, t: int) -> Optional[int]:
        scip = self.model
        # robust to sparse periods (works for Ldown too)
        rem = sum(1 for p in self.periods if p >= t)
        L = min(self.Ldown.get(u, 1), rem)
        if L <= 1:
            return None
        key = self._mindown_key(u, t, L)
        if key in self._seen_keys:
            return None

        row = scip.createEmptyRowSepa(self, f"mindown_{u}_{t}", -scip.infinity(), float(L),
                                      local=False, modifiable=False, removable=True)

        for tau in range(t, t+L):
            v = self.x.get((u,tau))
            if v is not None:
                scip.addVarToRow(row, v, 1.0)

        if (u,t) in self.z:
            scip.addVarToRow(row, self.z[(u,t)], float(L))
        elif (u,t) in self.x or (u,t-1) in self.x:
            if (u,t-1) in self.x:
                scip.addVarToRow(row, self.x[(u,t-1)], float(L))
            if (u,t) in self.x:
                scip.addVarToRow(row, self.x[(u,t)], -float(L))

        scip.flushRowExtensions(row)
        activity = scip.getRowSolActivity(row)
        viol = max(0.0, activity - float(L))
        if viol < self.minviolation or not scip.isCutEfficacious(row):
            scip.releaseRow(row)
            return None

        scip.addCut(row)
        self.ncuts_added += 1
        self._seen_keys.add(key)

        scip.releaseRow(row)
        return SCIP_RESULT.SEPARATED

    # ---------- SCIP callbacks ----------
    def sepaexeclp(self):
        scip = self.model
        if self.rootonly and scip.getDepth() > 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        if scip.getLPSolstat() != "optimal":
            return {"result": SCIP_RESULT.DIDNOTRUN}

        self._lazy_init()
        if not self.units or not self.periods:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        cuts_added_this_round = 0
        result = SCIP_RESULT.DIDNOTFIND

        # simple sweep; keep cheap for now
        for u in self.units:
            for t in self.periods:
                if cuts_added_this_round >= self.maxcuts_per_round:
                    break

                r = self._add_minup_cut(u, t)
                if r == SCIP_RESULT.SEPARATED:
                    result = r; cuts_added_this_round += 1
                if r == SCIP_RESULT.CUTOFF:
                    return {"result": r}

                if t != self.periods[0] and cuts_added_this_round < self.maxcuts_per_round:
                    r = self._add_mindown_cut(u, t)
                    if r == SCIP_RESULT.SEPARATED:
                        result = r; cuts_added_this_round += 1
                    if r == SCIP_RESULT.CUTOFF:
                        return {"result": r}
        return {"result": result}
