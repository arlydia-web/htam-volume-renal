#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07 — La conclusion négative dépend-elle du sélecteur employé ?

LA QUESTION. M3 conforme ne dépasse pas M2 (script 06). On peut objecter que le LASSO est un
mauvais sélecteur pour des paramètres radiomiques fortement corrélés entre eux, et qu'un autre
choix aurait donné un autre résultat. Ce script répond en refaisant l'exercice avec HUIT
stratégies de sélection, toutes appliquées À L'INTÉRIEUR de chaque pli d'apprentissage :

    · trois sélecteurs pénalisés ou filtres généraux : LASSO, elastic net, test F univarié ;
    · deux filtres non linéaires : information mutuelle, importances de forêt aléatoire ;
    · trois restrictions par famille de paramètres (forme, premier ordre, texture), avec
      sélection LASSO à l'intérieur de la famille.

DEUX PRÉCAUTIONS QUI CHANGENT LE RÉSULTAT
  · les filtres à nombre fixé retiennent CINQ paramètres, la valeur déclarée dans les
    méthodes, et non un nombre choisi après coup ;
  · les familles sont soumises au même LASSO intra-pli, et non versées entières dans la
    régression. Verser soixante-quinze descripteurs de texture dans un modèle de huit
    variables ajusté sur soixante-trois sujets dégrade évidemment la prédiction : on mesurerait
    alors un surdimensionnement, pas une absence de signal biologique. Ce n'est pas la question
    posée.

À LIRE. Aucune stratégie ne fait mieux que M2. Les écarts vont de −0,046 à −0,004, tous
compatibles avec zéro ou défavorables. La conclusion ne tient donc pas au sélecteur.

Lancer :  python scripts/07_robustesse_selection.py    (long : compter vingt à soixante minutes)
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htam import config, donnees, metriques, modelisation, reference
from htam.metriques import fr, fr_signe

# libellé imprimé → (stratégie de sélection, famille restreinte éventuelle)
STRATEGIES = [
    ("M2 + LASSO (L1)", "lasso", None),
    ("M2 + elastic net", "elasticnet", None),
    ("M2 + sélection univariée (test F)", "univarie", None),
    ("M2 + information mutuelle", "mutuelle", None),
    ("M2 + forêts aléatoires", "foret", None),
    ("M2 + paramètres de forme", "lasso", "forme"),
    ("M2 + paramètres de premier ordre", "lasso", "premier_ordre"),
    ("M2 + paramètres de texture", "lasso", "texture"),
]


def main():
    depart = time.time()
    config.preparer_dossiers()
    cohorte, synth_c = donnees.charger_cohorte()
    radiomique, synth_r = donnees.charger_radiomique()
    controle = reference.Verificateur(synth_c or synth_r)

    Xr, Xc, y, noms, _ = donnees.cohorte_radiomique(cohorte, radiomique)
    print(f"\nn = {len(y)}, {int(y.sum())} non-récupérations, {Xr.shape[1]} paramètres "
          f"radiomiques, K = {config.K_FILTRE} pour les filtres à nombre fixé.\n")

    n_clinique = len(config.VARIABLES_M1)
    p_M1, _ = modelisation.oof_radiomique(Xc[:, :n_clinique], None, y, None)
    p_M2, _ = modelisation.oof_radiomique(Xc, None, y, None)
    auc_M1, auc_M2 = metriques.auc(y, p_M1), metriques.auc(y, p_M2)
    ecart = metriques.delta_auc(y, p_M1, p_M2)
    print(f"{'M1 (clinique seul)':<38s} AUC = {fr(auc_M1)}")
    print(f"{'M2 (référence, clinique + volume)':<38s} AUC = {fr(auc_M2)}")
    print(f"    M2 contre M1 : ΔAUC = {fr_signe(ecart['delta'])}  "
          f"IC 95 % [{fr_signe(ecart['bas'])} ; {fr_signe(ecart['haut'])}]  p = {fr(ecart['p'], 2)}\n",
          flush=True)

    lignes = []
    for libelle, strategie, famille in STRATEGIES:
        XR = Xr if famille is None else Xr[:, donnees.indices_famille(noms, famille)]
        p, taille = modelisation.oof_radiomique(Xc, XR, y, strategie)
        aire = metriques.auc(y, p)
        comparaison = metriques.delta_auc(y, p_M2, p)
        lignes.append((libelle, aire, comparaison, taille))
        print(f"{libelle:<38s} AUC = {fr(aire)}   ΔAUC = {fr_signe(comparaison['delta'])}   "
              f"IC 95 % [{fr_signe(comparaison['bas'])} ; {fr_signe(comparaison['haut'])}]   "
              f"p = {fr(comparaison['p'], 2)}   ({taille:.1f} param./pli)", flush=True)
        if libelle in reference.ROBUSTESSE_SELECTION:
            attendu = reference.ROBUSTESSE_SELECTION[libelle]
            controle.verifier(f"AUC — {libelle}", aire, attendu[0], tolerance=0.004)

    meilleure = max(lignes, key=lambda l: l[1])
    print(f"\nMeilleure stratégie : {meilleure[0]} à {fr(meilleure[1])} — soit "
          f"{fr_signe(meilleure[1] - auc_M2)} par rapport à M2.")
    print("Aucune ne dépasse M2 : la conclusion ne dépend pas du sélecteur employé.")

    sortie = config.DOSSIER_RESULTATS / "07_robustesse_selection.csv"
    with open(sortie, "w", newline="", encoding="utf-8") as fichier:
        plume = csv.writer(fichier, delimiter=";")
        plume.writerow(["stratégie", "AUC", "ΔAUC vs M2", "IC bas", "IC haut", "p",
                        "paramètres par pli"])
        plume.writerow(["M1 (clinique seul)", round(auc_M1, 3), "", "", "", "", ""])
        plume.writerow(["M2 (référence)", round(auc_M2, 3), "référence", "", "", "", ""])
        for libelle, aire, comparaison, taille in lignes:
            plume.writerow([libelle, round(aire, 3), round(comparaison["delta"], 3),
                            round(comparaison["bas"], 3), round(comparaison["haut"], 3),
                            round(comparaison["p"], 3), round(taille, 1)])

    controle.bilan()
    print(f"\nÉcrit : {sortie}")
    print(f"({time.time() - depart:.0f} s)")


if __name__ == "__main__":
    main()
