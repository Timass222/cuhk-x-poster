# -*- coding: utf-8 -*-
"""Порт прод-слоёв декодера (decode_final: пулинг повторов, i2b |dL|=1,
dl2 |dL|>=2 с гардом Жаккара, solo, fb2, репэйр) в честный харнесс —
на трейн-сегментах. Плюс гарды v2 (H5 агента): Жаккар на i2b,
согласованность лог-длительностей на dl2/solo.
make_stack(g, layers, jac_i2b, dur_gate, jac_thr) -> stack_pred(fused, segs, tau)
  layers: "none" (как vfull head: пулинг+fb2+репэйр) | "i2b" (прод без
  --align) | "align" (прод с --align = i2b + dl2 + solo)."""
import numpy as np
from scipy.optimize import linear_sum_assignment

EPS = 1e-12


def map_distinct(Cg, A_in, A1_in, top_m=8, w_tr=1.0):
    """H4: точный MAP по слотам чанка: unary = групповая сумма log post
    (Cg, L x K), переходы 2-го порядка log A_in (1-го — A1_in для второго
    слота), все метки различны; DFS с отсечением по верхней границе."""
    L, K = Cg.shape
    cands = [np.argsort(-Cg[i])[:top_m] for i in range(L)]
    lA1 = np.log(A1_in + 1e-12)
    lA2 = np.log(A_in + 1e-12)
    best_rem = np.array([Cg[i, cands[i]].max() for i in range(L)])
    suffix = np.concatenate([np.cumsum(best_rem[::-1])[::-1][1:], [0.0]])
    best = [-np.inf, None]

    def dfs(i, prev2, prev1, used, score, path):
        if i == L:
            if score > best[0]:
                best[0], best[1] = score, list(path)
            return
        if score + suffix[i] + (0 if i == 0 else w_tr * 0.0) <= best[0]:
            return
        for c in cands[i]:
            if c in used:
                continue
            s = score + Cg[i, c]
            if i == 1:
                s += w_tr * lA1[prev1, c]
            elif i >= 2:
                s += w_tr * lA2[prev2, prev1, c]
            if s + suffix[i] <= best[0]:
                continue
            used.add(c)
            path.append(c)
            dfs(i + 1, prev1, c, used, s, path)
            path.pop()
            used.discard(c)

    dfs(0, -1, -1, set(), 0.0, [])
    return best[1]


def make_stack(g, layers="align", jac_i2b=0.0, dur_gate=0.0, jac_thr=0.25,
               solo_gate=0.8, repair="hungarian", map_w=1.0, map_scope="dup",
               solo_rule="cos", gamma_pair=None, pos_beta=0.0):
    y, T_sec, K, THR, lf, fin = (g["y"], g["T_sec"], g["K"], g["THR"],
                                 g["lf"], g["fin"])
    BETA, GAMMA, CGATE = g["BETA"], g["GAMMA"], g["CGATE"]
    fb2, _nan_rows = g["fb2"], g["_nan_rows"]

    def cosm(E, a, b):
        num = (E[a] * E[b]).sum(1)
        den = np.linalg.norm(E[a], axis=1) * np.linalg.norm(E[b], axis=1) + EPS
        return num / den

    def dur_ok(seg, ia, ib):
        """mean |Δ log-длина| по спаренным позициям <= dur_gate (NaN -> ок)."""
        if not dur_gate:
            return True
        la, lb = lf[seg[ia]], lf[seg[ib]]
        m = np.isfinite(la) & np.isfinite(lb)
        if m.sum() == 0:
            return True
        return float(np.abs(la[m] - lb[m]).mean()) <= dur_gate

    def jaccard(logE, a, b):
        la = set(logE[a].argmax(1).tolist())
        lb = set(logE[b].argmax(1).tolist())
        return len(la & lb) / len(la | lb)

    def stack_pred(fused, segs, tau):
        lift, bins = g["lift"], g["bins"]
        pred = np.full(len(y), -1)
        okr = ~_nan_rows(fused)
        pred[okr] = fused[okr].argmax(1)
        for seg in segs:
            logE = np.where(np.isnan(fused[seg]), np.log(1.0 / K),
                            np.log(fused[seg] + EPS)) * tau
            logE = logE + BETA * np.where(fin[seg][:, None],
                                          lift[bins[seg]], 0.0)
            E = np.exp(logE - logE.max(1, keepdims=True))
            E /= E.sum(1, keepdims=True)
            gaps = np.diff(T_sec[seg])
            bnd = np.concatenate([[False], gaps > THR])
            cut = np.where(gaps > THR)[0]
            ch, s0 = [], 0
            for c in cut:
                ch.append(np.arange(s0, c + 1))
                s0 = c + 1
            ch.append(np.arange(s0, len(seg)))
            lift3 = g.get("lift3")
            if lift3 is not None:
                # ординал чанка в группе повторов: группы по базовому E
                ords = np.zeros(len(seg), int)
                gi0 = 0
                while gi0 < len(ch):
                    o = 0
                    while gi0 + 1 < len(ch) and len(ch[gi0 + 1]) == len(ch[gi0])                             and cosm(E, ch[gi0], ch[gi0 + 1]).mean() >= CGATE:
                        gi0 += 1
                        o += 1
                        ords[ch[gi0]] = min(o, 2)
                    gi0 += 1
                logE = np.where(np.isnan(fused[seg]), np.log(1.0 / K),
                                np.log(fused[seg] + EPS)) * tau
                logE = logE + BETA * np.where(
                    fin[seg][:, None], lift3[bins[seg], ords], 0.0)
                E = np.exp(logE - logE.max(1, keepdims=True))
                E /= E.sum(1, keepdims=True)
            if pos_beta and g.get("pos_lift") is not None:
                # позиционный приор: 0 = первый в чанке (len>=2), 1 = одиночка, 2 = прочие
                pos = np.full(len(seg), 2)
                for c in ch:
                    pos[c[0]] = 1 if len(c) == 1 else 0
                logE = logE + pos_beta * g["pos_lift"][pos]
                E = np.exp(logE - logE.max(1, keepdims=True))
                E /= E.sum(1, keepdims=True)
            pooled = logE.copy()
            groups, gi = [], 0
            while gi < len(ch):
                grp = [ch[gi]]
                while gi + 1 < len(ch) and len(ch[gi + 1]) == len(ch[gi]):
                    if cosm(E, ch[gi], ch[gi + 1]).mean() < CGATE:
                        break
                    grp.append(ch[gi + 1])
                    gi += 1
                groups.append(grp)
                if len(grp) > 1:
                    gam = (gamma_pair if (gamma_pair is not None and len(grp) == 2)
                           else GAMMA)
                    for i in range(len(grp[0])):
                        ri = [c[i] for c in grp]
                        s = sum(logE[r] for r in ri)
                        for r in ri:
                            pooled[r] = logE[r] + gam * (s - logE[r])
                gi += 1
            if layers in ("i2b", "align"):
                for a, b in zip(ch[:-1], ch[1:]):
                    if abs(len(a) - len(b)) != 1 or min(len(a), len(b)) < 2:
                        continue
                    if jac_i2b and jaccard(logE, a, b) < jac_i2b:
                        continue
                    sh, lg = (a, b) if len(a) < len(b) else (b, a)
                    best, bscore = None, -1.0
                    for kk in range(len(lg)):
                        idx_l = [j for j in range(len(lg)) if j != kk]
                        scv = float(cosm(E, sh, lg[idx_l]).mean())
                        if scv > bscore:
                            best, bscore = idx_l, scv
                    if bscore < CGATE:
                        continue
                    if not dur_ok(seg, sh, lg[best]):
                        continue
                    for i, j in zip(range(len(sh)), best):
                        pooled[sh[i]] = pooled[sh[i]] + GAMMA * logE[lg[j]]
                        pooled[lg[j]] = pooled[lg[j]] + GAMMA * logE[sh[i]]
            if layers == "align":
                for a, b in zip(ch[:-1], ch[1:]):
                    dL = abs(len(a) - len(b))
                    if dL < 2 or min(len(a), len(b)) < 2:
                        continue
                    if jaccard(logE, a, b) < jac_thr:
                        continue
                    sh, lg = (a, b) if len(a) < len(b) else (b, a)
                    best_o, bscore = None, -1.0
                    for o in range(dL + 1):
                        scv = float(cosm(E, sh, lg[o:o + len(sh)]).mean())
                        if scv > bscore:
                            best_o, bscore = o, scv
                    if bscore < CGATE:
                        continue
                    if not dur_ok(seg, sh, lg[best_o:best_o + len(sh)]):
                        continue
                    for i in range(len(sh)):
                        j = lg[best_o + i]
                        pooled[sh[i]] = pooled[sh[i]] + GAMMA * logE[j]
                        pooled[j] = pooled[j] + GAMMA * logE[sh[i]]
                for i2, c2 in enumerate(ch):
                    if len(c2) != 1:
                        continue
                    cands = []
                    if i2 > 0 and len(ch[i2 - 1]) >= 2:
                        cands.extend(ch[i2 - 1].tolist())
                    if i2 + 1 < len(ch) and len(ch[i2 + 1]) >= 2:
                        cands.extend(ch[i2 + 1].tolist())
                    if not cands:
                        continue
                    if solo_rule == "restart":
                        # одиночка = первый элемент соседнего листа (протокол:
                        # 97% повторов начинаются с 1-го элемента): кандидаты —
                        # первые элементы соседних чанков, cos-гейт как у solo
                        cands = []
                        if i2 > 0 and len(ch[i2 - 1]) >= 2:
                            cands.append(int(ch[i2 - 1][0]))
                        if i2 + 1 < len(ch) and len(ch[i2 + 1]) >= 2:
                            cands.append(int(ch[i2 + 1][0]))
                    ca = np.array(cands)
                    scv = (E[c2[0]][None, :] * E[ca]).sum(1) / (
                        np.linalg.norm(E[c2[0]]) * np.linalg.norm(E[ca], axis=1)
                        + EPS)
                    jb = int(np.argmax(scv))
                    if scv[jb] < solo_gate:
                        continue
                    j = cands[jb]
                    if not dur_ok(seg, np.array([c2[0]]), np.array([j])):
                        continue
                    pooled[c2[0]] = pooled[c2[0]] + GAMMA * logE[j]
                    pooled[j] = pooled[j] + GAMMA * logE[c2[0]]
            post = fb2(pooled, bnd)
            pd_seg = post.argmax(1)
            lp = np.log(post + EPS)
            for grp in groups:
                L = len(grp[0])
                if L < 2:
                    continue
                Cg = sum(lp[c] for c in grp)
                for c in grp:
                    lab = pd_seg[c]
                    if len(set(lab.tolist())) == L:
                        continue
                    if repair == "map":
                        lab_new = map_distinct(Cg, g["A_in"], g["A1_in"],
                                               w_tr=map_w)
                        if lab_new is not None:
                            pd_seg[c] = np.array(lab_new)
                        continue
                    vals, cnts_ = np.unique(lab, return_counts=True)
                    dup = set(vals[cnts_ > 1].tolist())
                    confl = [i for i in range(L) if lab[i] in dup]
                    keep = {lab[i] for i in range(L) if i not in confl}
                    cand = [k for k in range(K) if k not in keep]
                    sub = Cg[np.ix_(confl, cand)]
                    r_i, c_i = linear_sum_assignment(-sub)
                    for ri, ci in zip(r_i, c_i):
                        pd_seg[c[confl[ri]]] = cand[ci]
            pred[seg] = pd_seg
        return pred

    return stack_pred
