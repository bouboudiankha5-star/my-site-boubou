#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:03:44 2026

@author: abdoulayedianka
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation uniforme des croyances sur le simplexe et ajustement de lois Beta.
Les limites graphiques ont été élargies pour éviter que les annotations ne débordent.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from scipy.optimize import fsolve

# ----------------------- Paramètres -----------------------
D1 = 175 / 600       # seuil E1 / E2
D2 = 325 / 600       # seuil E2 / E3
N_TARGET = 2500      # nombre de riverains
SEED = 12345

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "panel_croyances.csv")
FIG_PATH = os.path.join(HERE, "simplex_rectangle.png")


# ----------------------- Ajustement Beta -----------------------
def fit_beta(P1, P3):
    """Résout Phi(d1)=P1 et 1-Phi(d2)=P3. Renvoie (a,b) ou None."""
    def eq(theta):
        a, b = np.exp(theta)
        return [beta_dist.cdf(D1, a, b) - P1,
                beta_dist.cdf(D2, a, b) - (1 - P3)]
                
    for a0, b0 in [(1, 1), (2, 2), (0.5, 0.5), (5, 1), (1, 5)]:
        try:
            sol, _, ier, _ = fsolve(eq, [np.log(a0), np.log(b0)], full_output=True, xtol=1e-10)
            if ier == 1:
                a, b = np.exp(sol)
                if a > 0 and b > 0 and np.isfinite(a) and np.isfinite(b):
                    if abs(beta_dist.cdf(D1, a, b) - P1) < 1e-4 and \
                       abs(beta_dist.cdf(D2, a, b) - (1 - P3)) < 1e-4:
                        return a, b
        except Exception:
            continue
    return None


def w_prelec(p, eta, gamma):
    """Poids de Prelec : w(p) = exp(-eta * (-ln p)^gamma)."""
    return np.exp(-eta * (-np.log(p)) ** gamma)


def w_prelec_inv(s, eta, gamma):
    """Inverse de Prelec : P = exp(-((-ln s)/eta)^(1/gamma))."""
    return np.exp(-((-np.log(s)) / eta) ** (1 / gamma))



def ce_lottery(x, y, wP, alpha):
    """Équivalent certain : ce = [(x^a - y^a)*w(P(E)) + y^a]^(1/a)."""
    return ((x**alpha - y**alpha) * wP + y**alpha) ** (1/alpha)

# Les 3 paires de gains (x, y) des loteries du protocole
LOTTERIES = [(500, 0), (250, 0), (500, 250)]   # L1, L2, L3

#### ----------------------- Simulation du Panel -----------------------####
def simuler_panel(n_target=N_TARGET, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    # --- ÉTAPE 1
    while len(rows) < n_target:
        P1 = rng.uniform(0, 1)
        P3 = rng.uniform(0, 1)
        P2 = 1 - P1 - P3
        
        if P2 < 0:  # Rejet hors du simplexe
            continue
            
        res = fit_beta(P1, P3)
        if res is None:
            continue
            
        a, b = res
        

        # --- ÉTAPE 2 : fonction de déformation de Prelec (1998) ---
        # w(p) = exp(-eta * (-ln p)^gamma)
        eta   = rng.uniform(0.1, 1.5)   # (a) pessimisme
        gamma = rng.uniform(0.1, 1.5)   # (b) sensibilité à la vraisemblance

        

        # --- ÉTAPE 3 : pondérations de croyances ---
        wP1    = w_prelec(P1, eta, gamma)        # (a)
        wP2    = w_prelec(P2, eta, gamma)        # (b)
        wP3    = w_prelec(P3, eta, gamma)        # (c)
        wP1P2  = w_prelec(P1 + P2, eta, gamma)   # (d)


        # --- ÉTAPE 4 : paramètre de la fonction d'utilité (puissance) ---
        alpha = rng.uniform(0.1, 2.1)   # (a) courbure : concave<1, linéaire=1, convexe>1


# --- ÉTAPE 6 : fonction de préférences sociales (ratio comportemental) ---
        s = rng.uniform(0, 1)   # (a) s(s1,s2,r) ~ U[0,1]
        
        
        
        # --- ÉTAPE 7 : contribution c = F^{-1}(1 - w^{-1}(s)) ---
        c = beta_dist.ppf(1 - w_prelec_inv(s, eta, gamma), a, b)
        
        
        # coefficient de bruit, tiré une fois par individu
        sigma = rng.uniform(0, 0.025)
        


        row={"P1": P1, "P2": P2, "P3": P3, "a": a, "b": b,
                     "eta": eta, "gamma": gamma,
                     "wP1": wP1, "wP2": wP2, "wP3": wP3, "wP1P2": wP1P2,
                     "alpha": alpha, "s": s, "c": c, "sigma": sigma}
        
    
       # --- ÉTAPE 5 : 12 équivalents certains AVEC terme d'erreur ---
        wEvents = {"E1": wP1, "E2": wP2, "E3": wP3, "E1E2": wP1P2}
        for li, (x, y) in enumerate(LOTTERIES, start=1):
            for ev, wP in wEvents.items():
                ce_sans_bruit = ce_lottery(x, y, wP, alpha)
                eps = rng.normal(0, sigma * abs(x - y))   # bruit ~ N(0, sigma*|x-y|)
                row[f"ce_L{li}_{ev}"] = ce_sans_bruit + eps
 
        rows.append(row)
    return pd.DataFrame(rows)




##### --------------LE TRIANGLE RECTANGLE-------------------------

def tracer_triangle(df):
    fig, ax = plt.subplots(figsize=(9, 9)) 
    
    # Affichage des points (河川 / riverains)
    sc = ax.scatter(df["P1"], df["P3"], c=df["a"] / (df["a"] + df["b"]),
                    cmap="viridis", s=8, alpha=0.85)
    
    # Dessin des frontières du triangle
    ax.plot([0, 1], [0, 0], "k-", lw=1.2)   # Bas (P3 = 0)
    ax.plot([0, 0], [0, 1], "k-", lw=1.2)   # Gauche (P1 = 0)
    ax.plot([0, 1], [1, 0], "k-", lw=1.2)   # Hypoténuse (P2 = 0)
    
    # Marquer les 3 sommets (Coins)
    ax.scatter([1, 0, 0], [0, 0, 1], color="red", s=55, zorder=5)
    
    # --- ANNOTATIONS DES COINS ---
    # E1 pur (Coin bas-droite)
    ax.annotate("$E_1$ pur\n($P_1=1$)", (1, 0), xytext=(10, -20),
                textcoords="offset points", fontsize=10, color="red",
                ha="left", va="top")
    
    # E2 pur (Coin bas-gauche / Origine)
    ax.annotate("$E_2$ pur\n($P_2=1$)", (0, 0), xytext=(-15, -20),
                textcoords="offset points", fontsize=10, color="red",
                ha="right", va="top")
    
    # E3 pur (Coin haut-gauche)
    ax.annotate("$E_3$ pur\n($P_3=1$)", (0, 1), xytext=(-15, 15),
                textcoords="offset points", fontsize=10, color="red",
                ha="right", va="bottom")
    
    # Centre (Croyance uniforme)
    ax.scatter([1/3], [1/3], marker="s", color="blue", s=80, zorder=6)
    ax.annotate("Croyance uniforme\n$(1/3,1/3,1/3)$\nBeta(1,1)", (1/3, 1/3),
                xytext=(30, 30), textcoords="offset points", fontsize=9,
                color="blue", arrowprops=dict(arrowstyle="->", color="blue"))
    
    # Texte sur l'hypoténuse
    ax.text(0.60, 0.51, "$P_2 = 0$ (hypoténuse)", rotation=-45,
            fontsize=9, color="dimgray", ha="center")
    
    # Labellisation et titres
    ax.set_xlabel("$P_1$  —  croyance « contribution faible » ($E_1$)", fontsize=11, labelpad=10)
    ax.set_ylabel("$P_3$  —  croyance « contribution forte » ($E_3$)", fontsize=11, labelpad=10)
    ax.set_title("Cartographie du 2-simplexe des croyances\n"
                 f"{len(df)} riverains virtuels  ($P_2 = 1 - P_1 - P_3$)",
                 fontsize=12, pad=15)
    
    # --- LIMITES DES AXES : j'essaye d'Élargir pour donner de l'espace aux textes) ---
    ax.set_xlim(-0.15, 1.20)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal")
    
    # Barre de couleur
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label(r"Croyance moyenne  $\mathbb{E}[X] = a/(a+b)$", fontsize=10)
    
    # Sauvegarde et affichage
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.show()


# =========================================================================
# EXÉCUTION
# =========================================================================
if __name__ == "__main__":
    df = simuler_panel()
    df.to_csv(CSV_PATH, index=False)
    
    print(f"{len(df)} riverains simulés.")
    print(f"CSV enregistré sous : {CSV_PATH}")
    print(df.head())
    
    tracer_triangle(df)
    
    
    # ============================================================
# Mettre les 12 colonnes "ce" EN LIGNES.
# Chaque riverain occupe 12 lignes ; ses paramètres sont répétés.
# ============================================================

CSV_CE_LIGNE = os.path.join(HERE, "panel_ce_en_ligne.csv")

# Identifiant du riverain (si absent)
if "riverain" not in df.columns:
    df.insert(0, "riverain", range(1, len(df) + 1))

# Les 12 colonnes ce, et tout le reste répéter
ce_cols = [c for c in df.columns if c.startswith("ce_")]
autres  = [c for c in df.columns if c not in ce_cols]


panel_long = df.melt(id_vars=autres, value_vars=ce_cols,
                     var_name="loterie_evenement", value_name="ce")

# Tri pour regrouper les 12 lignes de chaque riverain
panel_long = (panel_long
              .sort_values(["riverain", "loterie_evenement"])
              .reset_index(drop=True))

panel_long.to_csv(CSV_CE_LIGNE, index=False)
print(f"Panel avec les 12 ce en ligne -> {CSV_CE_LIGNE}")
print(f"{len(panel_long)} lignes ({df['riverain'].nunique()} riverains x 12 ce)")
    
    
    
    
    
# ============================================================
# EXPORT LATEX — les 12 premiers riverains, TOUS les paramètres,
# ce EN LIGNE (12 lignes par riverain = 144 lignes), via longtable
# Préambule requis : \usepackage{longtable}  (+ \usepackage{pdflscape} si paysage)
# ============================================================

TEX_12RIV = os.path.join(HERE, "tableau_12_riverains_long.tex")

if "riverain" not in df.columns:
    df.insert(0, "riverain", range(1, len(df) + 1))

ce_cols = [c for c in df.columns if c.startswith("ce_")]
autres  = [c for c in df.columns if c not in ce_cols and c != "sigma"]  # sigma exclu

sub = df.head(12)
long = sub.melt(id_vars=autres, value_vars=ce_cols,
                var_name="loterie_evenement", value_name="ce")
long = long.sort_values(["riverain", "loterie_evenement"]).reset_index(drop=True)

# En-têtes LaTeX
entetes = {
    "riverain": "$i$", "P1": "$P_1$", "P2": "$P_2$", "P3": "$P_3$",
    "a": "$a$", "b": "$b$", "eta": r"$\eta$", "gamma": r"$\gamma$",
    "alpha": r"$\alpha$", "s": "$s$",
    "wP1": "$w(P_1)$", "wP2": "$w(P_2)$", "wP3": "$w(P_3)$", "wP1P2": "$w(P_1{+}P_2)$",
    "c": "$c$", "sigma": r"$\sigma$",
    "loterie_evenement": "lot./évén.", "ce": "$ce$",
}
cols = list(long.columns)
head = " & ".join(entetes.get(c, c) for c in cols)

def fmt(col, val):
    if col == "riverain":
        return str(int(val))
    if col == "loterie_evenement":
        return val.replace("_", r"\_")
    if col == "ce":
        return f"{val:.1f}"
    return f"{val:.3f}"

# Construire les lignes ; ajouter un filet entre riverains pour la lisibilité
lignes = []
prev = None
for _, r in long.iterrows():
    if prev is not None and r["riverain"] != prev:
        lignes.append("\\hline")
    lignes.append(" & ".join(fmt(c, r[c]) for c in cols) + r" \\")
    prev = r["riverain"]
corps = "\n".join(lignes)

ncol = len(cols)
tableau = (
    "\\begin{longtable}{" + "c" * ncol + "}\n"
    "\\caption{Panel simulé~: les douze premiers riverains virtuels "
    "(les douze équivalents certains en ligne).}\\\\\n"
    "\\label{tab:douze_riverains_long}\\\\\n"
    "\\hline\n"
    + head + " \\\\\n\\hline\n\\endfirsthead\n"
    # en-tête répété sur les pages suivantes
    "\\multicolumn{" + str(ncol) + "}{l}{\\small\\itshape suite du tableau \\ref{tab:douze_riverains_long}}\\\\\n"
    "\\hline\n"
    + head + " \\\\\n\\hline\n\\endhead\n"
    "\\hline\n\\multicolumn{" + str(ncol) + "}{r}{\\small\\itshape suite à la page suivante}\\\\\n\\endfoot\n"
    "\\hline\n\\endlastfoot\n"
    + corps + "\n"
    "\\end{longtable}\n"
)

with open(TEX_12RIV, "w", encoding="utf-8") as f:
    f.write(tableau)
print(f"Tableau longtable des 12 riverains -> {TEX_12RIV}")
print(f"{len(long)} lignes (12 riverains x 12 ce)")







# ============================================================
# SIMULATION INVERSE (contre-analyse)
# Réutilise : np, pd, os, beta_dist, fsolve, D1, D2, HERE, df.
# N'ajoute que les imports et fonctions nouveaux.
# ============================================================
from scipy.optimize import minimize, least_squares

if "riverain" not in df.columns:
    df.insert(0, "riverain", range(1, len(df) + 1))

LOT = [(500, 0), (250, 0), (500, 250)]      # paires de gains L1, L2, L3
EVENTS = ["E1", "E2", "E3", "E1E2"]          # 4 événements


# ------------------------------------------------------------
# fit_beta ROBUSTE : renvoie TOUJOURS un (a, b), jamais None.
# fsolve d'abord (exact, comme la directe) ; sinon repli moindres carrés.
# ------------------------------------------------------------

def fit_beta_robuste(P1, P3):
    c1 = min(max(P1,     1e-6), 1 - 1e-6)
    c2 = min(max(1 - P3, 1e-6), 1 - 1e-6)
    def resid(theta):
        a, b = np.exp(theta)
        return [beta_dist.cdf(D1, a, b) - c1, beta_dist.cdf(D2, a, b) - c2]
    for a0, b0 in [(1,1),(2,2),(0.5,0.5),(5,1),(1,5)]:
        try:
            sol, _, ier, _ = fsolve(resid, [np.log(a0), np.log(b0)],
                                    full_output=True, xtol=1e-10)
            if ier == 1:
                a, b = np.exp(sol)
                if a>0 and b>0 and np.isfinite(a) and np.isfinite(b):
                    if abs(beta_dist.cdf(D1,a,b)-c1)<1e-4 and abs(beta_dist.cdf(D2,a,b)-c2)<1e-4:
                        return a, b
        except Exception:
            continue
    best = None
    for a0, b0 in [(1,1),(2,2),(0.5,0.5),(5,1),(1,5)]:
        try:
            r = least_squares(resid, [np.log(a0), np.log(b0)], xtol=1e-12, ftol=1e-12)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            continue
    a, b = np.exp(best.x)
    return a, b


# ------------------------------------------------------------
# ÉTAPE 2 — All-at-once : alpha + pondérations w(P(E))
# ------------------------------------------------------------
def all_at_once(ce_obs):
    def somme_carres(theta):
        alpha = theta[0]
        w = {"E1": theta[1], "E2": theta[2], "E3": theta[3], "E1E2": theta[4]}
        tot = 0.0
        for li, (x, y) in enumerate(LOT, start=1):
            for ev in EVENTS:
                pred = ((x**alpha - y**alpha) * w[ev] + y**alpha) ** (1/alpha)
                tot += (ce_obs[(li, ev)] - pred) ** 2
        return tot
    best = None
    for alpha0 in [0.5, 1.0, 1.5]:
        res = minimize(somme_carres, [alpha0, .5, .5, .5, .5],
                       bounds=[(0.05, 3)] + [(1e-4, 1)] * 4, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    return best.x


# ------------------------------------------------------------
# ÉTAPE 3 — estimation de (eta, gamma) et recouvrement de P(E)
# Fidèle à la méthodologie : on estime (eta, gamma) en exploitant
# les paris sur les événements simples ET leur union (Erazo-Kpegli),
# c.-à-d. en cherchant (eta, gamma, P1, P3) tels que la fonction de
# source W(P) reproduise les pondérations estimées :
#     W(P1)=w1, W(P2)=w2, W(P3)=w3, W(P1+P2)=w12,  avec P2 = 1-P1-P3.
# P2 se déduit automatiquement ; aucune normalisation arbitraire.
# ------------------------------------------------------------
def W_source(P, eta, gamma):
    """Fonction de source de Prelec : W(P) = exp(-eta*(-ln P)^gamma)."""
    P = np.clip(P, 1e-12, 1 - 1e-12)
    return np.exp(-eta * (-np.log(P)) ** gamma)

def estimer_source(w1, w2, w3, w12):
    """Estime conjointement (eta, gamma, P1, P3) ; P2 = 1 - P1 - P3."""
    def err(theta):
        eta, gamma, P1, P3 = theta
        P2 = 1 - P1 - P3
        if P2 <= 0 or P1 <= 0 or P3 <= 0:
            return 1e6
        return ((W_source(P1, eta, gamma) - w1) ** 2
                + (W_source(P2, eta, gamma) - w2) ** 2
                + (W_source(P3, eta, gamma) - w3) ** 2
                + (W_source(P1 + P2, eta, gamma) - w12) ** 2)
    best = None
    for e0 in [0.3, 0.7, 1.1]:
        for g0 in [0.4, 0.8, 1.2]:
            for P10, P30 in [(0.3, 0.3), (0.2, 0.1), (0.1, 0.6)]:
                res = minimize(err, [e0, g0, P10, P30],
                               bounds=[(0.1, 1.5), (0.1, 1.5),
                                       (1e-3, 0.999), (1e-3, 0.999)],
                               method="L-BFGS-B")
                if best is None or res.fun < best.fun:
                    best = res
    return best.x   # [eta, gamma, P1, P3]


# ------------------------------------------------------------
# BOUCLE : reconstruction de TOUS les paramètres
# ------------------------------------------------------------
lignes = []
for _, r in df.iterrows():
    ce_obs = {(li, ev): r[f"ce_L{li}_{ev}"]
              for li in range(1, 4) for ev in EVENTS}

    # étape 2 : alpha + pondérations
    alpha_e, w1, w2, w3, w12 = all_at_once(ce_obs)

    # étape 3 : (eta, gamma, P1, P3) estimés conjointement ; P2 déduit
    eta_e, gamma_e, P1, P3 = estimer_source(w1, w2, w3, w12)
    P2 = 1 - P1 - P3

    # étape 4 : a, b , puis E[X]
    a_e, b_e = fit_beta_robuste(P1, P3)
    EX_e = a_e / (a_e + b_e)

    # étape 5 : preferences sociales  s = W(1 - F(c))
    # W_source est définie localement (indépendante du code direct).
    Fc = beta_dist.cdf(r["c"], a_e, b_e)
    s_e = W_source(1 - Fc, eta_e, gamma_e)

    lignes.append({
        "riverain": r["riverain"],
        "P1_est": P1, "P2_est": P2, "P3_est": P3,
        "a_est": a_e, "b_est": b_e, "EX_est": EX_e,
        "eta_est": eta_e, "gamma_est": gamma_e, "alpha_est": alpha_e,
        "wP1_est": w1, "wP2_est": w2, "wP3_est": w3, "wP1P2_est": w12,
        "s_est": s_e,
    })

panel_inverse = pd.DataFrame(lignes)
panel_inverse.to_csv(os.path.join(HERE, "panel_inverse_large.csv"), index=False)
print(f"Simulation inverse : {len(panel_inverse)} riverains, "
      f"NaN total = {int(panel_inverse.isna().sum().sum())}.")
print(panel_inverse.head().round(3).to_string(index=False))


# ============================================================
# FIGURE CDF : distributions cumulées vraies vs estimées
# (style Abdellaoui et al. 2020). Trait plein = vrai, tireté = estimé.
# L'APPEL est inclus en bas : la figure s'affiche directement.
# ============================================================

def _cdf(ax, vrai, est, titre, clip=None):
    vrai = np.asarray(vrai, float); est = np.asarray(est, float)
    vrai = vrai[np.isfinite(vrai)]; est = est[np.isfinite(est)]
    if clip is not None:
        vrai = vrai[(vrai >= clip[0]) & (vrai <= clip[1])]
        est  = est[(est  >= clip[0]) & (est  <= clip[1])]
    xv = np.sort(vrai); yv = np.arange(1, len(xv)+1) / len(xv)
    xe = np.sort(est);  ye = np.arange(1, len(xe)+1) / len(xe)
    ax.plot(xv, yv, "-",  color="#C0392B", lw=1.5, label="Vraie")
    ax.plot(xe, ye, "--", color="#2C3E80", lw=1.5, label="Estimée")
    ax.set_title(titre, fontsize=9)
    ax.set_xlabel("valeur", fontsize=7)
    ax.set_ylabel("prop. cumulée", fontsize=7)
    ax.set_ylim(0, 1); ax.legend(fontsize=6, loc="lower right")


def figure_cdf(df, panel_inverse, path="figure_cdf.png"):
    vrais = df[["riverain","P1","P2","P3","a","b","eta","gamma","alpha","s",
                "wP1","wP2","wP3","wP1P2"]]
    m = vrais.merge(panel_inverse, on="riverain")
    EXv = m["a"]/(m["a"]+m["b"])

    fig, ax = plt.subplots(4, 4, figsize=(15, 14))

    # Ligne 1 : structure
    _cdf(ax[0,0], m["alpha"], m["alpha_est"], r"Utilité $\alpha$")
    _cdf(ax[0,1], m["eta"],   m["eta_est"],   r"Pessimisme $\eta$")
    _cdf(ax[0,2], m["gamma"], m["gamma_est"], r"Insensibilité $\gamma$")
    _cdf(ax[0,3], m["s"],     m["s_est"],     r"Ratio social $s$")

    # Ligne 2 : pondérations
    _cdf(ax[1,0], m["wP1"],   m["wP1_est"],   r"$w(P_1)$")
    _cdf(ax[1,1], m["wP2"],   m["wP2_est"],   r"$w(P_2)$")
    _cdf(ax[1,2], m["wP3"],   m["wP3_est"],   r"$w(P_3)$")
    _cdf(ax[1,3], m["wP1P2"], m["wP1P2_est"], r"$w(P_1{+}P_2)$")

    # Ligne 3 : croyances
    _cdf(ax[2,0], m["P1"], m["P1_est"], r"Croyance $P_1$")
    _cdf(ax[2,1], m["P2"], m["P2_est"], r"Croyance $P_2$")
    _cdf(ax[2,2], m["P3"], m["P3_est"], r"Croyance $P_3$")
    _cdf(ax[2,3], EXv,     m["EX_est"], r"Croyance moyenne $\mathbb{E}[X]$")

    # Ligne 4 : paramètres Beta (limités à [0,10] pour la lisibilité)
    _cdf(ax[3,0], m["a"], m["a_est"], r"Paramètre Beta $a$", clip=(0,10))
    _cdf(ax[3,1], m["b"], m["b_est"], r"Paramètre Beta $b$", clip=(0,10))
    ax[3,2].axis("off"); ax[3,3].axis("off")

    fig.suptitle(f"Distributions cumulées : vraies (trait plein) vs estimées "
                 f"(tireté), $N = {len(m)}$", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(path, dpi=140)
    print(f"Figure CDF enregistrée -> {path}")
    return fig


figure_cdf(df, panel_inverse, "figure_cdf.png")
plt.show()

# ============================================================
# TABLEAU DE COMPARAISON (style Kpegli et al. 2023, Table 2)
# TOUS les paramètres : EAM, REQM, r entre vrai et estimé.
# ============================================================

_vrais = df[["riverain","P1","P2","P3","a","b","eta","gamma","alpha","s",
             "wP1","wP2","wP3","wP1P2"]]
_m = _vrais.merge(panel_inverse, on="riverain")
_m["EX_vrai"] = _m["a"]/(_m["a"]+_m["b"])

# (colonne vraie, colonne estimée, libellé)
_paires = [
    ("alpha","alpha_est",  r"Utilité $\alpha$"),
    ("eta","eta_est",      r"Pessimisme $\eta$"),
    ("gamma","gamma_est",  r"Insensibilité $\gamma$"),
    ("s","s_est",          r"Ratio social $s$"),
    ("wP1","wP1_est",      r"Pondération $w(P_1)$"),
    ("wP2","wP2_est",      r"Pondération $w(P_2)$"),
    ("wP3","wP3_est",      r"Pondération $w(P_3)$"),
    ("wP1P2","wP1P2_est",  r"Pondération $w(P_1{+}P_2)$"),
    ("P1","P1_est",        r"Croyance $P_1$"),
    ("P2","P2_est",        r"Croyance $P_2$"),
    ("P3","P3_est",        r"Croyance $P_3$"),
    ("EX_vrai","EX_est",   r"Croyance moyenne $\mathbb{E}[X]$"),
    ("a","a_est",          r"Paramètre Beta $a$"),
    ("b","b_est",          r"Paramètre Beta $b$"),
]

_lignes = []
for v, e, lab in _paires:
    d = _m[[v, e]].replace([np.inf,-np.inf], np.nan).dropna()
    diff = d[v] - d[e]
    eam  = diff.abs().mean()
    reqm = np.sqrt((diff**2).mean())
    r    = np.corrcoef(d[v], d[e])[0, 1]
    _lignes.append(f"{lab} & {eam:.4f} & {reqm:.4f} & {r:.3f} \\\\")

_corps = "\n".join(_lignes)
tableau_comparaison = (
    "\\begin{table}[htbp]\n\\centering\n"
    "\\caption{Qualité de la récupération des paramètres~: erreur absolue "
    "moyenne (EAM), racine de l'erreur quadratique moyenne (REQM) et "
    "corrélation $r$ entre valeurs vraies et estimées.}\n"
    "\\label{tab:recuperation}\n"
    "\\begin{tabular}{lccc}\n\\hline\n"
    "Paramètre & EAM & REQM & $r$ \\\\\n\\hline\n"
    + _corps + "\n\\hline\n"
    "\\end{tabular}\n\\end{table}\n"
)

with open(os.path.join(HERE, "tableau_recuperation.tex"), "w", encoding="utf-8") as f:
    f.write(tableau_comparaison)
print("Tableau (TOUS paramètres) -> tableau_recuperation.tex")
print(tableau_comparaison)




# ============================================================
# EXPORT LATEX — les 12 premiers riverains de la simulation INVERSE.
# Paramètres ESTIMÉS, 1 ligne par riverain (sans la colonne EX).
# ============================================================

TEX_INV12 = os.path.join(HERE, "tableau_12_riverains_inverse.tex")

# Colonnes à afficher (estimées)
cols = ["riverain", 
    "alpha_est", 
    "wP1_est", "wP2_est", "wP3_est", "wP1P2_est", 
    "eta_est", "gamma_est", 
    "P1_est", "P2_est", "P3_est", 
    "a_est", "b_est", 
    "s_est"]

sub = panel_inverse[cols].head(12)

# En-têtes LaTeX (avec chapeau pour signaler les valeurs estimées)
entetes = {"riverain": "$i$",
    "P1_est": r"$\widehat{P_1}$", "P2_est": r"$\widehat{P_2}$", "P3_est": r"$\widehat{P_3}$",
    "a_est": r"$\widehat{a}$", "b_est": r"$\widehat{b}$",
    "eta_est": r"$\widehat{\eta}$", "gamma_est": r"$\widehat{\gamma}$",
    "alpha_est": r"$\widehat{\alpha}$",
    "wP1_est": r"$\widehat{w(P_1)}$", "wP2_est": r"$\widehat{w(P_2)}$",
    "wP3_est": r"$\widehat{w(P_3)}$", "wP1P2_est": r"$\widehat{w(P_1{+}P_2)}$",
    "s_est": r"$\widehat{s}$",
}
head = " & ".join(entetes[c] for c in cols)

def fmt(col, val):
    if col == "riverain":
        return str(int(val))
    return f"{val:.3f}"

lignes = []
for _, r in sub.iterrows():
    lignes.append(" & ".join(fmt(c, r[c]) for c in cols) + r" \\")
corps = "\n".join(lignes)

ncol = len(cols)
tableau = (
    "\\begin{longtable}{" + "c" * ncol + "}\n"
    "\\caption{Contre-analyse~: paramètres estimés pour les douze premiers "
    "riverains virtuels.}\\\\\n"
    "\\label{tab:douze_riverains_inverse}\\\\\n"
    "\\hline\n"
    + head + " \\\\\n\\hline\n\\endfirsthead\n"
    "\\multicolumn{" + str(ncol) + "}{l}{\\small\\itshape suite du tableau \\ref{tab:douze_riverains_inverse}}\\\\\n"
    "\\hline\n"
    + head + " \\\\\n\\hline\n\\endhead\n"
    "\\hline\n\\endfoot\n"
    "\\hline\n\\endlastfoot\n"
    + corps + "\n"
    "\\end{longtable}\n"
)

with open(TEX_INV12, "w", encoding="utf-8") as f:
    f.write(tableau)
print(f"Tableau LaTeX des 12 riverains (inverse) -> {TEX_INV12}")





# ============================================================
# NUAGES DE POINTS : vrai vs estimé (style Kpegli, parameter recovery)
# Axe X = valeur vraie (simulation directe)
# Axe Y = valeur estimée (contre-analyse)
# Ligne 45° : récupération parfaite si les points tombent dessus.
# ============================================================
import numpy as np
import matplotlib.pyplot as plt


def _nuage(ax, vrai, est, titre, clip=None):
    vrai = np.asarray(vrai, float); est = np.asarray(est, float)
    msk = np.isfinite(vrai) & np.isfinite(est)
    vrai, est = vrai[msk], est[msk]
    if clip is not None:
        m = (vrai >= clip[0]) & (vrai <= clip[1]) & (est >= clip[0]) & (est <= clip[1])
        vrai, est = vrai[m], est[m]
    ax.scatter(vrai, est, s=10, color="#2C3E80", alpha=0.5, edgecolors="none")
    lo = min(vrai.min(), est.min()); hi = max(vrai.max(), est.max())
    ax.plot([lo, hi], [lo, hi], "-", color="black", lw=1.2)   # ligne 45°
    r = np.corrcoef(vrai, est)[0, 1]
    ax.set_title(f"{titre}  ($r = {r:.3f}$)", fontsize=9)
    ax.set_xlabel("valeur vraie", fontsize=7)
    ax.set_ylabel("valeur estimée", fontsize=7)
    ax.set_aspect("equal", adjustable="box")


def figure_nuages(df, panel_inverse, path="figure_nuages.png"):
    v = df[["riverain","P1","P2","P3","a","b","eta","gamma","alpha","s",
            "wP1","wP2","wP3","wP1P2"]]
    m = v.merge(panel_inverse, on="riverain")
    EXv = m["a"]/(m["a"]+m["b"])

    fig, ax = plt.subplots(4, 4, figsize=(15, 14))

    _nuage(ax[0,0], m["alpha"], m["alpha_est"], r"Utilité $\alpha$")
    _nuage(ax[0,1], m["eta"],   m["eta_est"],   r"Pessimisme $\eta$")
    _nuage(ax[0,2], m["gamma"], m["gamma_est"], r"Insensibilité $\gamma$")
    _nuage(ax[0,3], m["s"],     m["s_est"],     r"Ratio social $s$")

    _nuage(ax[1,0], m["wP1"],   m["wP1_est"],   r"$w(P_1)$")
    _nuage(ax[1,1], m["wP2"],   m["wP2_est"],   r"$w(P_2)$")
    _nuage(ax[1,2], m["wP3"],   m["wP3_est"],   r"$w(P_3)$")
    _nuage(ax[1,3], m["wP1P2"], m["wP1P2_est"], r"$w(P_1{+}P_2)$")

    _nuage(ax[2,0], m["P1"], m["P1_est"], r"Croyance $P_1$")
    _nuage(ax[2,1], m["P2"], m["P2_est"], r"Croyance $P_2$")
    _nuage(ax[2,2], m["P3"], m["P3_est"], r"Croyance $P_3$")
    _nuage(ax[2,3], EXv,     m["EX_est"], r"Croyance moyenne $\mathbb{E}[X]$")

    _nuage(ax[3,0], m["a"], m["a_est"], r"Paramètre Beta $a$", clip=(0,10))
    _nuage(ax[3,1], m["b"], m["b_est"], r"Paramètre Beta $b$", clip=(0,10))
    ax[3,2].axis("off"); ax[3,3].axis("off")

    fig.suptitle(f"Récupération des paramètres : valeur estimée contre valeur "
                 f"vraie (la droite est la diagonale à 45°), $N = {len(m)}$",
                 fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(path, dpi=140)
    print(f"Figure nuages enregistrée -> {path}")
    return fig


figure_nuages(df, panel_inverse, "figure_nuages.png")
plt.show()


# ============================================================
# FIGURE: nuage unique vrai vs estimé,
# tous paramètres confondus, deux couleurs + légende + ligne 45°.
#
# Deux familles distinguées selon leur statut dans la chaîne :
#   - "Quantités décisionnelles" (bleu) : pondérations w(P(E)),
#     quasi directement révélées par les équivalents certains.
#   - "Paramètres reconstruits" (rouge) : croyances, loi Beta,
#     préférences sociales, en bout de chaîne d'inversion.
# Les valeurs sont normalisées par paramètre (min-max) pour être
# comparables sur un même graphe, comme chez Kpegli.
# ============================================================

def figure_recovery_kpegli(df, panel_inverse, path="figure_recovery_kpegli.png"):
    v = df[["riverain","P1","P2","P3","a","b","eta","gamma","alpha","s",
            "wP1","wP2","wP3","wP1P2"]]
    m = v.merge(panel_inverse, on="riverain")
    m["EX_vrai"] = m["a"]/(m["a"]+m["b"])

    # (vrai, estimé, famille) ; famille 0 = décisionnel (bleu), 1 = reconstruit (rouge)
    paires = [
        ("wP1","wP1_est",0), ("wP2","wP2_est",0),
        ("wP3","wP3_est",0), ("wP1P2","wP1P2_est",0),
        ("alpha","alpha_est",0), ("eta","eta_est",0),
        ("P1","P1_est",1), ("P2","P2_est",1), ("P3","P3_est",1),
        ("EX_vrai","EX_est",1), ("gamma","gamma_est",1),
        ("s","s_est",1), ("a","a_est",1), ("b","b_est",1),
    ]

    def norm(x):
        x = np.asarray(x, float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo) if hi > lo else x*0

    xb, yb, xr, yr = [], [], [], []
    for vrai, est, fam in paires:
        a = norm(m[vrai]); b = norm(m[est])
        msk = np.isfinite(a) & np.isfinite(b)
        if fam == 0:
            xb += list(a[msk]); yb += list(b[msk])
        else:
            xr += list(a[msk]); yr += list(b[msk])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(xr, yr, s=7, color="#8B2E3C", alpha=0.35, edgecolors="none",
               label="Paramètres reconstruits")
    ax.scatter(xb, yb, s=7, color="#2C3E80", alpha=0.40, edgecolors="none",
               label="Quantités décisionnelles")
    ax.plot([0,1],[0,1], "-", color="black", lw=1.3)        # ligne 45°
    ax.set_xlabel("valeur vraie (normalisée)", fontsize=10)
    ax.set_ylabel("valeur estimée (normalisée)", fontsize=10)
    ax.set_title("Récupération des paramètres : estimé contre vrai",
                 fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9, framealpha=1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"-> {path}")
    return fig


figure_recovery_kpegli(df, panel_inverse, "figure_recovery_kpegli.png")
plt.show()
