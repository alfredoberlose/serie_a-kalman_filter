"""Kalman + distribuzioni CONTRO il mercato, come previsori di 1X2.
Il filtro e' gia' walk-forward per costruzione: pred[t] usa solo le partite < t.
La mappa pred -> probabilita' 1X2 e' stimata SOLO sulle stagioni <= 2023 e applicata a 2024-2026.
"""
import numpy as np, pandas as pd, glob, os, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize, brentq
import statsmodels.api as sm

import os as _os
D = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'data')
fs = [f for f in sorted(glob.glob(os.path.join(D, 'serie_a_*.csv'))) if 'fixtures' not in f]
parts = []
for f in fs:
    x = pd.read_csv(f)
    x['Date'] = pd.to_datetime(x['Date'], dayfirst=True, errors='coerce')
    d0 = x['Date'].dropna().iloc[0]
    x['season'] = d0.year if d0.month >= 7 else d0.year - 1
    parts.append(x)
d = pd.concat(parts, ignore_index=True).dropna(subset=['HST', 'AST', 'Date', 'FTR']).copy()
d = d.sort_values('Date').reset_index(drop=True)

TEAMS = sorted(set(d.HomeTeam) | set(d.AwayTeam)); IDX = {t: i for i, t in enumerate(TEAMS)}
IN_SEASON = {s: {IDX[t] for t in set(g.HomeTeam) | set(g.AwayTeam)} for s, g in d.groupby('season')}


def kalman(par):
    n, k = len(d), len(TEAMS)
    home, away = d.HomeTeam.map(IDX).to_numpy(), d.AwayTeam.map(IDX).to_numpy()
    y = (d.HST - d.AST).to_numpy(float); season = d.season.to_numpy()
    b, P = np.zeros(k), np.eye(k) * par['P0']
    pred, F, v = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        if t > 0 and season[t] != season[t - 1]:
            bef, now = IN_SEASON[season[t - 1]], IN_SEASON[season[t]]
            for i in bef & now: P[i, i] += par['Q_summer']
            rel = [b[i] for i in bef - now]; mr = np.mean(rel) if rel else 0.0
            for i in now - bef:
                b[i], P[i, :], P[:, i] = mr, 0.0, 0.0; P[i, i] = par['P0']
        i, j = home[t], away[t]
        P[i, i] += par['Q']; P[j, j] += par['Q']
        pred[t] = b[i] - b[j] + par['h']
        F[t] = P[i, i] + P[j, j] - 2 * P[i, j] + par['R']
        v[t] = y[t] - pred[t]
        g = (P[:, i] - P[:, j]) / F[t]
        b = b + g * v[t]; P = P - np.outer(g, g) * F[t]
    return pred, F, v


up = lambda x: dict(Q=np.exp(x[0]), R=np.exp(x[1]), h=x[2], Q_summer=np.exp(x[3]), P0=np.exp(x[4]))
nll = lambda x: 0.5 * np.sum(np.log(2 * np.pi * kalman(up(x))[1]) + kalman(up(x))[2] ** 2 / kalman(up(x))[1])
# stima dei parametri sulle sole stagioni <= 2023 (niente look-ahead)
mask_tr = (d.season <= 2023).to_numpy()


def nll_tr(x):
    p, F, v = kalman(up(x))
    return 0.5 * np.sum(np.log(2 * np.pi * F[mask_tr]) + v[mask_tr] ** 2 / F[mask_tr])


r = minimize(nll_tr, [np.log(.05), np.log(8.), .5, np.log(.5), np.log(2.)],
             method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000))
par = up(r.x)
print('parametri stimati solo su stagioni <= 2023:', {k: round(v, 4) for k, v in par.items()})
pred, F, v = kalman(par)
d['pred'] = pred

y = d.FTR.map({'H': 0, 'D': 1, 'A': 2}).to_numpy()
d = d.dropna(subset=['B365CH', 'B365CD', 'B365CA'])
keep = d.index.to_numpy()
y = pd.Series(y, index=range(len(y))).loc[keep].to_numpy()
d = d.reset_index(drop=True)


def dvpow(o):
    rr = 1 / o
    a = brentq(lambda a: (rr ** a).sum() - 1, 0.3, 3.0)
    return rr ** a


Pm = np.array([dvpow(d.loc[i, ['B365CH', 'B365CD', 'B365CA']].to_numpy(float)) for i in range(len(d))])

tr = (d.season <= 2023).to_numpy(); te = ~tr
print(f'\naddestramento {tr.sum()} partite (<=2023), test {te.sum()} partite (2024-2026)')

# mappa pred -> 1X2: logit multinomiale stimato SOLO sul training
X = sm.add_constant(d[['pred']])
mk = sm.MNLogit(y[tr], X[tr]).fit(disp=0)
Pk = mk.predict(X[te])
Pk = np.asarray(Pk)

ll = lambda P, yy: -np.log(np.clip(P[np.arange(len(yy)), yy], 1e-12, 1)).mean()
br = lambda P, yy: ((P - np.eye(3)[yy]) ** 2).sum(1).mean()
yt = y[te]
print('\nFUORI CAMPIONE (2024-2026), come previsori dell 1X2')
print(f"  Kalman + distribuzioni : log-loss {ll(Pk, yt):.5f}   Brier {br(Pk, yt):.5f}")
print(f"  mercato (quote chiusura): log-loss {ll(Pm[te], yt):.5f}   Brier {br(Pm[te], yt):.5f}")
print(f"  costante (frequenze storiche): log-loss {ll(np.tile(np.eye(3)[y[tr]].mean(0),(te.sum(),1)), yt):.5f}")

# quanto costa usare il Kalman al posto del mercato?
diff = (-np.log(Pk[np.arange(len(yt)), yt])) - (-np.log(Pm[te][np.arange(len(yt)), yt]))
from scipy.stats import ttest_1samp
t_, p_ = ttest_1samp(diff, 0)
print(f"\n  differenza log-loss (Kalman meno mercato): {diff.mean():+.5f}, t = {t_:+.2f}, p = {p_:.2e}")

# e se li metto insieme? (regressione di encompassing in forma logit)
Z = sm.add_constant(pd.DataFrame({'mkt_h': np.log(Pm[:, 0] / Pm[:, 1]), 'mkt_a': np.log(Pm[:, 2] / Pm[:, 1]),
                                  'pred': d['pred']}))
me = sm.MNLogit(y[tr], Z[tr]).fit(disp=0)
print('\nENCOMPASSING: logit con mercato (offset logit) + Kalman, stimato sul training')
print(me.params.round(4).to_string())
print('\np-value dei coefficienti:')
print(me.pvalues.round(4).to_string())
