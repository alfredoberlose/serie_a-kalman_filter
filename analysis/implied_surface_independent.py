"""Il mercato e' sovra-identificato: 4 osservazioni indipendenti, 2 parametri.
Calibro lambda_h, lambda_a su Over 2.5 + (pi1 - pi2), poi uso i 2 gradi di liberta' residui:
  residuo 1 = pareggio   (gia' noto: e' il 'sorriso', rho di Dixon-Coles)
  residuo 2 = handicap asiatico   (MAI usato finora)
"""
import numpy as np, pandas as pd, glob, os
from scipy.stats import poisson, skellam
from scipy.optimize import brentq

import os as _os
D = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'data')
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, 'serie_a_*.csv')))
               if 'fixtures' not in f], ignore_index=True)

C = ['B365CH', 'B365CD', 'B365CA', 'B365C>2.5', 'B365C<2.5', 'AHCh', 'B365CAHH', 'B365CAHA', 'FTHG', 'FTAG']
d = d.dropna(subset=C).reset_index(drop=True)
print(f'partite con 1X2 + Over + handicap asiatico alla chiusura: {len(d)}')


def devig2(a, b):
    r = 1 / a + 1 / b
    return (1 / a) / r


def devig3(h, x, a):
    r = 1 / h + 1 / x + 1 / a
    return np.array([1 / h, 1 / x, 1 / a]) / r


def total_from_over(pU):
    """P(X+Y <= 2) = pU con X+Y ~ Poisson(T)."""
    f = lambda T: poisson.cdf(2, T) - pU
    return brentq(f, 0.05, 15.0)


def split_from_1x2(T, target):
    """trova d = lh - la tale che P(diff>0) - P(diff<0) = target, con lh+la = T."""
    def g(dd):
        lh, la = (T + dd) / 2, (T - dd) / 2
        return (1 - skellam.cdf(0, lh, la)) - skellam.cdf(-1, lh, la) - target
    lo, hi = -T + 1e-6, T - 1e-6
    if g(lo) * g(hi) > 0:
        return np.nan
    return brentq(g, lo, hi)


def ah_prob(lh, la, line):
    """P(vittoria), P(push) per la squadra di casa con handicap 'line' (anche quarti)."""
    if abs(line * 4 - round(line * 4)) > 1e-9:
        return np.nan, np.nan
    if abs(line * 2 - round(line * 2)) > 1e-9:                 # quarto di linea -> media di due mezze
        a = ah_prob(lh, la, line - 0.25)
        b = ah_prob(lh, la, line + 0.25)
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    # diff + line > 0 vince ; = 0 push
    k = -line
    if abs(k - round(k)) < 1e-9:                               # linea intera -> push possibile
        k = int(round(k))
        push = skellam.pmf(k, lh, la)
        win = 1 - skellam.cdf(k, lh, la)
    else:                                                      # mezza linea -> nessun push
        push = 0.0
        win = 1 - skellam.cdf(int(np.floor(k)), lh, la)
    return win, push


rows = []
for i, r in d.iterrows():
    p1, pX, p2 = devig3(r['B365CH'], r['B365CD'], r['B365CA'])
    pO = devig2(r['B365C>2.5'], r['B365C<2.5'])
    try:
        T = total_from_over(1 - pO)
        dd = split_from_1x2(T, p1 - p2)
        if not np.isfinite(dd):
            continue
    except Exception:
        continue
    lh, la = (T + dd) / 2, (T - dd) / 2

    # --- residuo 1: il pareggio (il 'sorriso' gia' noto)
    pX_mod = skellam.pmf(0, lh, la)

    # --- residuo 2: handicap asiatico (strumento mai usato)
    w, pu = ah_prob(lh, la, r['AHCh'])
    if not np.isfinite(w) or (w + (1 - w - pu)) <= 0:
        continue
    pAH_mod = w / (w + (1 - w - pu))
    pAH_mkt = devig2(r['B365CAHH'], r['B365CAHA'])

    rows.append(dict(T=T, lh=lh, la=la, line=r['AHCh'],
                     pX_mkt=pX, pX_mod=pX_mod, resX=pX - pX_mod,
                     pAH_mkt=pAH_mkt, pAH_mod=pAH_mod, resAH=pAH_mkt - pAH_mod,
                     home_cover=(r['FTHG'] - r['FTAG'] + r['AHCh']) > 0))

s = pd.DataFrame(rows)
print(f'partite calibrate: {len(s)}\n')
print('RESIDUO 1 - pareggio (il sorriso gia documentato nel paper)')
print(f"  mercato {s.pX_mkt.mean():.4f}  modello {s.pX_mod.mean():.4f}  "
      f"scarto medio {s.resX.mean():+.4f}  dev.std {s.resX.std():.4f}")
print('\nRESIDUO 2 - handicap asiatico (strumento nuovo)')
print(f"  mercato {s.pAH_mkt.mean():.4f}  modello {s.pAH_mod.mean():.4f}  "
      f"scarto medio {s.resAH.mean():+.4f}  dev.std {s.resAH.std():.4f}")

from scipy.stats import ttest_1samp, pearsonr
for nome, col in [('pareggio', 'resX'), ('handicap', 'resAH')]:
    t, p = ttest_1samp(s[col], 0)
    print(f"\n  {nome}: scarto medio {s[col].mean():+.5f}, t = {t:+.2f}, p = {p:.2e}")

# i due residui sono la stessa cosa o due cose diverse?
rr, pp = pearsonr(s.resX, s.resAH)
print(f"\ncorrelazione fra i due residui: r = {rr:+.3f} (p = {pp:.2e})")

# il residuo sull'handicap prevede chi copre lo spread?
import statsmodels.api as sm
X = sm.add_constant(s[['resAH']])
m = sm.Logit(s.home_cover.astype(int), X).fit(disp=0)
print('\nIL RESIDUO PREVEDE IL RISULTATO? (logit: copre la casa ~ residuo handicap)')
print(m.summary2().tables[1].round(4).to_string())
