import numpy as np, pandas as pd

BK = ['Netwin','Betsson','Bet365','GoldBet','Marathonbet','Starcasino','Snai','Vincitu','Eurobet',
      'Betflag','WilliamHill','Lottomatica','NetBet','LeoVegas','Planetwin365','AdmiralBet','Sisal',
      'bwin','Stake']

M = {
'Torino-Monza': [
 [2.15,2.15,2.05,2.15,2.13,2.11,2.10,2.12,2.10,2.15,2.11,2.15,2.12,1.98,2.15,2.10,2.10,2.00,2.10],
 [3.40,3.35,3.40,3.35,3.35,3.30,3.40,3.35,3.35,3.35,3.35,3.35,3.40,3.25,3.35,3.40,3.40,3.30,3.51],
 [3.60,3.60,3.70,3.60,3.54,3.55,3.60,3.55,3.50,3.60,3.60,3.60,3.55,3.35,3.60,3.55,3.60,3.20,3.46]],
'Cagliari-Bologna': [
 [3.10,3.10,3.20,3.10,3.09,3.05,3.10,3.05,3.15,3.10,3.10,3.10,3.05,2.90,3.10,3.05,3.10,2.90,3.18],
 [3.15,3.05,3.00,3.00,3.05,3.00,3.00,3.05,3.00,3.00,3.05,3.00,3.10,2.95,3.00,3.10,3.00,3.10,2.98],
 [2.55,2.55,2.45,2.50,2.50,2.50,2.50,2.51,2.45,2.50,2.50,2.50,2.50,2.32,2.50,2.50,2.50,2.25,2.44]],
'Como-Sassuolo': [
 [1.49,1.48,1.45,1.48,1.47,1.47,1.48,1.47,1.45,1.48,1.48,1.48,1.48,1.41,1.48,1.47,1.48,1.43,1.45],
 [4.90,4.85,5.00,4.75,5.00,4.80,4.75,4.90,4.90,4.75,4.90,4.75,4.85,4.30,4.75,5.00,4.75,4.60,4.95],
 [6.10,6.00,6.00,6.00,5.90,5.90,6.00,6.00,6.00,6.00,6.00,6.00,6.00,5.75,6.00,5.90,6.00,5.25,5.95]],
'Napoli-Roma': [
 [2.80,2.85,2.88,2.80,2.76,2.80,2.80,2.80,2.85,2.80,2.85,2.80,2.80,2.65,2.80,2.75,2.80,2.75,2.71],
 [3.35,3.25,3.10,3.20,3.27,3.20,3.25,3.30,3.15,3.20,3.15,3.20,3.30,3.05,3.20,3.30,3.25,3.40,3.36],
 [2.60,2.60,2.60,2.55,2.62,2.55,2.50,2.57,2.55,2.55,2.60,2.55,2.57,2.40,2.55,2.55,2.50,2.40,2.48]],
'Lazio-Parma': [
 [1.41,1.41,1.36,1.38,1.40,1.40,1.40,1.41,1.37,1.38,1.41,1.38,1.40,1.35,1.38,1.40,1.40,1.39,1.35],
 [4.80,4.60,4.75,4.75,4.75,4.55,4.75,4.70,4.60,4.75,4.60,4.75,4.75,4.30,4.75,4.65,4.75,4.50,4.65],
 [8.00,8.00,9.00,8.00,7.90,7.90,9.00,8.00,9.00,8.00,8.00,8.00,8.00,7.00,8.00,7.75,9.00,7.00,8.80]],
'Inter-Fiorentina': [
 [1.37,1.37,1.36,1.33,1.36,1.34,1.33,1.36,1.35,1.33,1.35,1.33,1.36,1.32,1.33,1.35,1.33,1.33,1.34],
 [5.60,5.40,5.25,5.50,5.60,5.30,5.50,5.50,5.40,5.50,5.40,5.50,5.60,4.90,5.50,5.50,5.50,5.25,5.10],
 [7.60,7.40,8.00,8.00,7.50,7.25,8.00,7.50,7.80,8.00,7.40,8.00,7.50,6.75,8.00,7.30,8.00,6.50,7.80]],
'Atalanta-Frosinone': [
 [1.85,1.85,1.80,1.82,1.83,1.81,1.80,1.84,1.82,1.82,1.83,1.82,1.83,1.72,1.82,1.82,1.80,1.71,1.79],
 [4.20,4.10,4.20,4.00,4.18,4.00,4.00,4.10,4.20,4.00,4.00,4.00,4.15,3.85,4.00,4.10,4.00,4.10,4.10],
 [3.85,3.80,3.80,3.85,3.79,3.70,3.60,3.80,3.75,3.85,3.75,3.85,3.80,3.65,3.85,3.75,3.60,3.40,3.74]],
'Genoa-Venezia': [
 [2.15,2.15,2.10,2.10,2.14,2.12,2.10,2.15,2.10,2.10,2.13,2.10,2.14,2.07,2.10,2.10,2.10,2.10,2.08],
 [3.40,3.30,3.40,3.20,3.38,3.25,3.10,3.35,3.25,3.20,3.30,3.20,3.40,3.20,3.20,3.35,3.10,3.30,3.34],
 [3.50,3.50,3.50,3.65,3.48,3.45,3.60,3.50,3.60,3.65,3.45,3.65,3.50,3.10,3.65,3.45,3.60,3.20,3.45]],
'Lecce-Juventus': [
 [7.10,7.00,8.00,7.50,7.30,6.85,7.50,7.10,7.80,7.50,7.00,7.50,7.00,7.00,7.50,6.90,7.50,6.25,7.60],
 [4.55,4.40,4.20,4.20,4.25,4.30,4.00,4.45,4.10,4.20,4.40,4.20,4.50,4.10,4.20,4.45,4.00,4.20,4.10],
 [1.47,1.47,1.44,1.45,1.48,1.44,1.47,1.46,1.45,1.45,1.45,1.45,1.46,1.37,1.45,1.45,1.47,1.43,1.42]],
'Udinese-Milan': [
 [5.20,5.10,5.50,5.15,5.05,4.95,5.25,5.10,5.20,5.15,5.00,5.15,5.10,4.70,5.15,5.00,5.25,4.60,5.00],
 [3.90,3.75,3.70,3.65,3.85,3.70,3.75,3.80,3.70,3.65,3.75,3.65,3.85,3.55,3.65,3.80,3.75,3.75,3.56],
 [1.70,1.70,1.67,1.68,1.69,1.67,1.65,1.69,1.67,1.68,1.68,1.68,1.68,1.61,1.68,1.68,1.65,1.61,1.67]],
}

matches = list(M)
O = np.array([M[m] for m in matches])          # (10 partite, 3 esiti, 19 book)

# ---------- 1) margine (overround) di ogni bookmaker, partita per partita
marg = (1.0 / O).sum(axis=1) - 1.0             # (10, 19)
tab = pd.DataFrame({'margine medio %': marg.mean(0) * 100,
                    'margine mediano %': np.median(marg, 0) * 100,
                    'peggior partita %': marg.max(0) * 100}, index=BK).sort_values('margine medio %')
print('MARGINE PER BOOKMAKER (10 partite Serie A, 1X2)')
print(tab.round(2).to_string())

# ---------- 2) quanto costa stare su UN solo book invece che prendere sempre la quota migliore
best = O.max(axis=2)                            # (10,3) quota migliore disponibile
marg_best = (1.0 / best).sum(axis=1) - 1.0
print(f"\nmargine del 'miglior prezzo su 19 book': {marg_best.mean()*100:.2f}% "
      f"(il migliore singolo book sta a {tab['margine medio %'].min():.2f}%)")

# perdita media per euro di quota rispetto al best price, e quante volte e' il migliore
loss = ((best[:, :, None] - O) / best[:, :, None]).mean(axis=(0, 1)) * 100
isbest = (O >= best[:, :, None] - 1e-9).mean(axis=(0, 1)) * 100
t2 = pd.DataFrame({'quota sotto il best price %': loss, 'volte che e il best price %': isbest},
                  index=BK).sort_values('quota sotto il best price %')
print('\nCONFRONTO COL MIGLIOR PREZZO DISPONIBILE (30 esiti)')
print(t2.round(2).to_string())

# ---------- 3) probabilita' eque di consenso e quota equa, partita per partita
#  consenso = media delle probabilita' implicite de-viggate di TUTTI i book (proporzionale)
p_impl = 1.0 / O
p_fair_bk = p_impl / p_impl.sum(axis=1, keepdims=True)     # de-vig proporzionale per book
p_cons = p_fair_bk.mean(axis=2)                            # (10,3)

rows = []
for i, m in enumerate(matches):
    for j, lab in enumerate(['1', 'X', '2']):
        k = int(np.argmax(O[i, j]))
        rows.append({'partita': m, 'esito': lab, 'p equa consenso': p_cons[i, j],
                     'quota equa': 1 / p_cons[i, j], 'miglior quota': O[i, j].max(),
                     'book': BK[k], 'EV % al best price': (O[i, j].max() * p_cons[i, j] - 1) * 100,
                     'EV % su Bet365': (O[i, j, BK.index('Bet365')] * p_cons[i, j] - 1) * 100})
ev = pd.DataFrame(rows)
print('\nEV AL MIGLIOR PREZZO (consenso de-viggato come probabilita vera)')
print(ev.sort_values('EV % al best price', ascending=False).head(12).round(3).to_string(index=False))
print(f"\nEV medio al best price: {ev['EV % al best price'].mean():+.2f}%   "
      f"| EV medio su Bet365: {ev['EV % su Bet365'].mean():+.2f}%")
print(f"esiti con EV > 0 al best price: {(ev['EV % al best price'] > 0).sum()} su 30")
