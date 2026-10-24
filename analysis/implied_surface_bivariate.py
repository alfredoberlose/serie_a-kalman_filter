"""Poisson BIVARIATO calibrato esattamente su 1X2 + Over 2.5.
  l1, l2  <- dall'1X2 (la differenza reti e' Skellam(l1,l2): l3 si cancella)
  l3      <- dall'Over 2.5
  handicap asiatico -> MAI usato: e' il test alla cieca.
Domanda: il parametro di dipendenza l3 e' una SUPERFICIE liscia (struttura vera)
o rumore che assorbe lo scarto? E' la differenza fra Dupire e un parametro libero.
"""
import numpy as np, pandas as pd, glob, os
from scipy.stats import poisson, skellam
from scipy.optimize import brentq, fsolve

import os as _os
D = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'data')
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, 'serie_a_*.csv')))
               if 'fixtures' not in f], ignore_index=True)
C = ['B365CH', 'B365CD', 'B365CA', 'B365C>2.5', 'B365C<2.5', 'AHCh', 'B365CAHH', 'B365CAHA', 'FTHG', 'FTAG']
d = d.dropna(subset=C).reset_index(drop=True)


def devig2(a, b):
    return (1 / a) / (1 / a + 1 / b)


def devig3(h, x, a):
    v = np.array([1 / h, 1 / x, 1 / a])
    return v / v.sum()


def l1l2_from_1x2(p1, pX):
    """Skellam(l1,l2): P(diff>0)=p1 e P(diff=0)=pX. Due equazioni, due incognite -> esatto."""
    def eq(z):
        l1, l2 = np.exp(z)
        return [1 - skellam.cdf(0, l1, l2) - p1, skellam.pmf(0, l1, l2) - pX]
    z, info, ok, _ = fsolve(eq, np.log([1.4, 1.1]), full_output=True)
    if ok != 1:
        return np.nan, np.nan
    return tuple(np.exp(z))


def l3_from_over(l1, l2, pU):
    """P(tot<=2) = e^{-l3}[F_S(2) + l3 F_S(0)], S ~ Poisson(l1+l2)."""
    S = l1 + l2
    F2, F0 = poisson.cdf(2, S), poisson.pmf(0, S)
    f = lambda l3: np.exp(-l3) * (F2 + l3 * F0) - pU
    if f(0.0) * f(3.0) > 0:
        return np.nan
    return brentq(f, 0.0, 3.0)


def ah_prob(l1, l2, line):
    if abs(line * 4 - round(line * 4)) > 1e-9:
        return np.nan, np.nan
    if abs(line * 2 - round(line * 2)) > 1e-9:
        a, b = ah_prob(l1, l2, line - 0.25), ah_prob(l1, l2, line + 0.25)
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    k = -line
    if abs(k - round(k)) < 1e-9:
        k = int(round(k))
        return 1 - skellam.cdf(k, l1, l2), skellam.pmf(k, l1, l2)
    return 1 - skellam.cdf(int(np.floor(k)), l1, l2), 0.0


rows = []
for _, r in d.iterrows():
    p1, pX, p2 = devig3(r['B365CH'], r['B365CD'], r['B365CA'])
    l1, l2 = l1l2_from_1x2(p1, pX)
    if not np.isfinite(l1):
        continue
    l3 = l3_from_over(l1, l2, 1 - devig2(r['B365C>2.5'], r['B365C<2.5']))
    if not np.isfinite(l3):
        continue
    w, pu = ah_prob(l1, l2, r['AHCh'])
    if not np.isfinite(w):
        continue
    lose = 1 - w - pu
    if w + lose <= 0:
        continue
    rows.append(dict(l1=l1, l2=l2, l3=l3, T=l1 + l2 + 2 * l3, sq=abs(l1 - l2),
                     pAH_mod=w / (w + lose), pAH_mkt=devig2(r['B365CAHH'], r['B365CAHA']),
                     cover=(r['FTHG'] - r['FTAG'] + r['AHCh']) > 0))

s = pd.DataFrame(rows)
s['resAH'] = s.pAH_mkt - s.pAH_mod
print(f'partite calibrate esattamente: {len(s)} su {len(d)}')
print(f"l3 implicito: mediana {s.l3.median():.4f}, media {s.l3.mean():.4f}, "
      f"q10-q90 {s.l3.quantile(.1):.3f}-{s.l3.quantile(.9):.3f}, "
      f"frazione a zero {(s.l3 < 1e-6).mean():.1%}")

from scipy.stats import ttest_1samp
t, p = ttest_1samp(s.resAH, 0)
print(f"\nTEST ALLA CIECA sull'handicap asiatico (mai usato nella calibrazione)")
print(f"  scarto medio {s.resAH.mean():+.5f} ({s.resAH.mean()*100:+.3f} punti %), "
      f"dev.std {s.resAH.std():.5f}, t = {t:+.2f}, p = {p:.3f}")

print('\nE UNA SUPERFICIE LISCIA? l3 implicito per gruppi')
s['binT'] = pd.qcut(s['T'], 6)
print(s.groupby('binT', observed=True).agg(n=('l3', 'size'), T=('T', 'mean'),
      l3_medio=('l3', 'mean'), l3_sd=('l3', 'std'), resAH=('resAH', 'mean')).round(4).to_string())
s['binS'] = pd.qcut(s['sq'], 6)
print()
print(s.groupby('binS', observed=True).agg(n=('l3', 'size'), squilibrio=('sq', 'mean'),
      l3_medio=('l3', 'mean'), l3_sd=('l3', 'std'), resAH=('resAH', 'mean')).round(4).to_string())

# quanta parte della variazione di l3 e' spiegata dalle due coordinate?
import statsmodels.api as sm
X = sm.add_constant(pd.DataFrame({'T': s['T'], 'sq': s['sq'], 'T2': s['T'] ** 2, 'sq2': s['sq'] ** 2}))
m = sm.OLS(s.l3, X).fit()
print(f"\nR2 di l3 spiegato da (totale, squilibrio) e loro quadrati: {m.rsquared:.3f}")

# il residuo dell'handicap prevede chi copre?
mm = sm.Logit(s.cover.astype(int), sm.add_constant(s[['resAH']])).fit(disp=0)
print('\nIL RESIDUO PREVEDE CHI COPRE LO SPREAD?')
print(mm.summary2().tables[1].round(4).to_string())
