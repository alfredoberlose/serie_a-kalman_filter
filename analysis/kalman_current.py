"""Filtro di Kalman di lega stimato per massima verosimiglianza su tutto lo storico,
aggiornato fino all'ultima partita disponibile. Nessun parametro scelto a mano."""
import numpy as np, pandas as pd, glob, os
from scipy.optimize import minimize

import os as _os
D = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'data')
fs = [f for f in sorted(glob.glob(os.path.join(D, 'serie_a_*.csv'))) if 'fixtures' not in f]
parts = []
for f in fs:
    x = pd.read_csv(f)
    x['Date'] = pd.to_datetime(x['Date'], dayfirst=True, errors='coerce')
    x['season'] = x['Date'].dropna().iloc[0].year if x['Date'].dropna().iloc[0].month >= 7 \
        else x['Date'].dropna().iloc[0].year - 1
    parts.append(x)
d = pd.concat(parts, ignore_index=True).dropna(subset=['HST', 'AST', 'Date'])
d = d.sort_values('Date').reset_index(drop=True)

TEAMS = sorted(set(d.HomeTeam) | set(d.AwayTeam))
IDX = {t: i for i, t in enumerate(TEAMS)}
IN_SEASON = {s: {IDX[t] for t in set(g.HomeTeam) | set(g.AwayTeam)} for s, g in d.groupby('season')}
print(f'{len(d)} partite, {len(TEAMS)} squadre, stagioni {sorted(IN_SEASON)}')


def kalman_league(d, par):
    n, k = len(d), len(TEAMS)
    home, away = d.HomeTeam.map(IDX).to_numpy(), d.AwayTeam.map(IDX).to_numpy()
    y = (d.HST - d.AST).to_numpy(float)
    season = d.season.to_numpy()
    b, P = np.zeros(k), np.eye(k) * par['P0']
    pred, F, v = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        if t > 0 and season[t] != season[t - 1]:
            before, now = IN_SEASON[season[t - 1]], IN_SEASON[season[t]]
            for i in before & now:
                P[i, i] += par['Q_summer']
            rel = [b[i] for i in before - now]
            mrel = np.mean(rel) if rel else 0.0
            for i in now - before:
                b[i], P[i, :], P[:, i] = mrel, 0.0, 0.0
                P[i, i] = par['P0']
        i, j = home[t], away[t]
        P[i, i] += par['Q']; P[j, j] += par['Q']
        pred[t] = b[i] - b[j] + par['h']
        F[t] = P[i, i] + P[j, j] - 2 * P[i, j] + par['R']
        v[t] = y[t] - pred[t]
        g = (P[:, i] - P[:, j]) / F[t]
        b = b + g * v[t]
        P = P - np.outer(g, g) * F[t]
    return dict(pred=pred, F=F, v=v, b=b, P=P)


def unpack(x):
    return dict(Q=np.exp(x[0]), R=np.exp(x[1]), h=x[2], Q_summer=np.exp(x[3]), P0=np.exp(x[4]))


def nll(x):
    r = kalman_league(d, unpack(x))
    return 0.5 * np.sum(np.log(2 * np.pi * r['F']) + r['v'] ** 2 / r['F'])


res = minimize(nll, [np.log(.05), np.log(8.), .5, np.log(.5), np.log(2.)],
               method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000))
par = unpack(res.x)
print('\nPARAMETRI STIMATI PER MASSIMA VEROSIMIGLIANZA (nessuno scelto a mano)')
for k_, v_ in par.items():
    print(f'  {k_:10s} = {v_:.4f}')
print(f'  log-verosimiglianza = {-res.fun:.1f}   AIC = {2*5+2*res.fun:.1f}')

r = kalman_league(d, par)
b, P = r['b'], r['P']
cur = sorted(IN_SEASON[max(IN_SEASON)])
f = pd.DataFrame({'squadra': [TEAMS[i] for i in cur], 'forza': b[cur],
                  'incertezza': np.sqrt(np.diag(P)[cur])}).sort_values('forza', ascending=False)
print(f"\nFORZA LATENTE AL {d.Date.max():%d/%m/%Y} (tiri in porta sopra la media, per partita)")
print(f.round(3).to_string(index=False))

PART = [('Como', 'Sassuolo', '1'), ('Lecce', 'Juventus', '2'), ('Inter', 'Fiorentina', '1'),
        ('Lazio', 'Parma', '1'), ('Torino', 'Monza', '1'), ('Cagliari', 'Bologna', '2'),
        ('Napoli', 'Roma', '2'), ('Atalanta', 'Frosinone', '1'), ('Genoa', 'Venezia', '1'),
        ('Udinese', 'Milan', '2')]
print('\nPREVISIONE DEL FILTRO PER IL TURNO (scarto tiri in porta atteso, casa meno trasferta)')
out = []
for h, a, _ in PART:
    if h not in IDX or a not in IDX:
        out.append(dict(partita=f'{h}-{a}', nota='squadra non nello storico')); continue
    i, j = IDX[h], IDX[a]
    m = b[i] - b[j] + par['h']
    sd = np.sqrt(P[i, i] + P[j, j] - 2 * P[i, j] + par['R'])
    out.append(dict(partita=f'{h}-{a}', scarto_atteso=m, incertezza=sd, rapporto=m / sd))
print(pd.DataFrame(out).round(3).to_string(index=False))
