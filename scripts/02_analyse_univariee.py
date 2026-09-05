#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02 — Association univariée de chaque prédicteur candidat au devenir rénal à six mois.

CE QUE FAIT CE SCRIPT. Pour chaque candidat : un odds ratio par unité cliniquement lisible
(la créatininémie par 100 µmol/L, l'âge par dix ans…), son intervalle de confiance, son p, et
l'aire sous la courbe ROC de la variable prise seule. Puis le graphique en forêt.

DEUX PARTIS PRIS, QUI CHANGENT LES CHIFFRES
  · aucune imputation : chaque prédicteur est estimé sur ses seuls cas complets, et son
    effectif est affiché dès qu'il diffère de celui de la cohorte. Imputer par la médiane
    gonfle les effectifs et déplace les odds ratios sans que rien ne le signale ;
  · l'unité est choisie pour que l'odds ratio se lise. Un odds ratio « par µmol/L » vaut
    1,002 et ne dit rien à personne ; le même, par 100 µmol/L, se lit d'un coup d'œil.

Un odds ratio univarié ne mesure pas un effet propre : il ignore tout le reste. Il sert à
choisir des candidats, jamais à conclure. Le modèle multivarié est en 03.

Lancer :  python scripts/02_analyse_univariee.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htam import config, donnees, figures
from htam.metriques import fr


def candidats(df):
    """(libellé, colonne, transformation vers l'unité d'interprétation)."""
    proteinurie = df.get("PU_admission_g_par_g")
    if proteinurie is not None and proteinurie.notna().sum() > 3:
        logs = np.log(proteinurie.dropna().to_numpy(dtype=float) + 0.01)
        ecart_type_log = float(np.std(logs, ddof=1))
    else:
        ecart_type_log = 1.0
    return [
        ("Créatininémie /100 µmol/L", "creat_admi", lambda x: x / 100),
        ("Volume rénal total /10 cm³", config.VOLUME, lambda x: x / 10),
        ("Protéinurie /écart-type de log", "PU_admission_g_par_g",
         lambda x: np.log(x + 0.01) / ecart_type_log),
        ("Hémoglobine d'admission /g/dL", "Hb_adm", lambda x: x),
        ("Indice de masse corporelle /5 kg/m²", "IMC", lambda x: x / 5),
        ("Âge /10 ans", "age", lambda x: x / 10),
        ("Hémolyse biologique", "MAT", lambda x: x),
        ("Sexe féminin", "sexe", lambda x: x),
        ("Pression systolique /10 mmHg", "PAS_admission", lambda x: x / 10),
        ("Pression diastolique /10 mmHg", "PAD_admission", lambda x: x / 10),
    ]


def main():
    config.preparer_dossiers()
    df, synthetique = donnees.charger_cohorte()
    n_total = len(df)
    print(f"\nAnalyse univariée — cohorte de {n_total} patients, "
          f"{int(df[config.CRITERE].sum())} non-récupérations.\n")

    resultats = []
    for libelle, colonne, transformation in candidats(df):
        if colonne not in df.columns:
            print(f"  (variable absente du fichier, ligne sautée : {libelle})")
            continue
        complet = df[[colonne, config.CRITERE]].dropna()
        if len(complet) < 20 or complet[colonne].nunique() < 2:
            print(f"  (effectif ou variabilité insuffisants, ligne sautée : {libelle})")
            continue
        x = complet[colonne].to_numpy(dtype=float)
        x = np.array([transformation(v) for v in x])
        y = complet[config.CRITERE].to_numpy(dtype=int)
        modele = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
        bas, haut = np.exp(modele.conf_int()[1])
        aire = roc_auc_score(y, x)
        resultats.append(dict(
            libelle=libelle if len(complet) == n_total else f"{libelle}  (n = {len(complet)})",
            n=len(complet), odds_ratio=float(np.exp(modele.params[1])),
            bas=float(bas), haut=float(haut), p=float(modele.pvalues[1]),
            # L'aire est rendue au-dessus de 0,5 : une variable protectrice discrimine autant
            # qu'une variable délétère, seul son sens change.
            auc=float(max(aire, 1 - aire))))

    print(f"{'prédicteur':46s}{'n':>4s}  {'OR [IC 95 %]':>22s}  {'p':>9s}  {'AUC':>6s}")
    print("-" * 92)
    for r in resultats:
        intervalle = f"{fr(r['odds_ratio'], 2)} [{fr(r['bas'], 2)} – {fr(r['haut'], 2)}]"
        p = "< 0,001" if r["p"] < 0.001 else fr(r["p"], 3)
        print(f"{r['libelle']:46s}{r['n']:>4d}  {intervalle:>22s}  {p:>9s}  {fr(r['auc'], 3):>6s}")

    sortie = config.DOSSIER_RESULTATS / "02_analyse_univariee.csv"
    with open(sortie, "w", newline="", encoding="utf-8") as fichier:
        plume = csv.writer(fichier, delimiter=";")
        plume.writerow(["prédicteur", "n", "OR", "IC bas", "IC haut", "p", "AUC"])
        for r in resultats:
            plume.writerow([r["libelle"], r["n"], round(r["odds_ratio"], 3), round(r["bas"], 3),
                            round(r["haut"], 3), round(r["p"], 4), round(r["auc"], 3)])
    print(f"\nÉcrit : {sortie}")

    # ── graphique en forêt ────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    figures.appliquer()
    fig, ax = plt.subplots(figsize=(figures.DOUBLE, 0.42 * len(resultats) + 1.4))
    positions = np.arange(len(resultats))[::-1]
    for position, r in zip(positions, resultats):
        # La significativité est aussi portée par l'épaisseur du trait : la couleur ne code
        # jamais seule une information.
        significatif = r["p"] < 0.05
        couleur = figures.NON_RECUPERATION if significatif else figures.NEUTRE
        ax.plot([r["bas"], r["haut"]], [position, position], color=couleur,
                lw=2.4 if significatif else 1.4)
        ax.plot(r["odds_ratio"], position, "o", color=couleur, ms=7 if significatif else 5.5)
    ax.axvline(1, color=figures.ENCRE, ls="--", alpha=0.6, lw=1)
    ax.set_yticks(positions)
    ax.set_yticklabels([r["libelle"] for r in resultats], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio univarié [IC 95 %] — échelle logarithmique")
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    figures.enregistrer(fig, "02_forest_univarie")

    if synthetique:
        print("Rappel : données synthétiques — aucune de ces valeurs n'est clinique.")


if __name__ == "__main__":
    main()
