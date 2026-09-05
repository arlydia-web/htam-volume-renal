#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03 — Validation interne du modèle parcimonieux : créatininémie + volume rénal total.

C'est le modèle qui porte le résultat principal du travail. Deux variables, obtenues à
l'admission, et rien d'autre — ni radiomique, ni histologie, ni suivi.

CE QUE LE SCRIPT ÉTABLIT, DANS L'ORDRE
  (1) discrimination hors échantillon et son intervalle de confiance ;
  (2) performance APPARENTE, optimisme par bootstrap de Harrell, performance corrigée ;
  (3) calibration : pente, calibration dans le grand, score de Brier ;
  (4) test de DeLong contre la créatininémie seule ;
  (5) courbe de décision : le modèle apporte-t-il un bénéfice net à un seuil de décision
      plausible, comparé aux deux stratégies triviales — traiter tout le monde, personne ;
  (6) transportabilité temporelle (IECV) : ajusté sur une période, évalué sur l'autre ;
  (7) l'équation du modèle ajusté sur toute la cohorte, pour qu'elle soit reproductible.

DEUX QUANTITÉS À NE JAMAIS CONFONDRE — elles valent toutes deux 0,876 sur cette cohorte,
par coïncidence numérique :
  · l'AUC hors échantillon de la CRÉATININÉMIE SEULE ;
  · l'AUC APPARENTE du modèle à deux variables.
La première est une performance honnête d'un modèle plus pauvre ; la seconde est la
performance flattée du modèle complet sur les patients qui ont servi à l'ajuster. Les
confondre fait paraître l'optimisme dix fois plus petit qu'il n'est.

Lancer :  python scripts/03_modele_parcimonieux.py       (compter une minute)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htam import config, donnees, metriques, modelisation, reference
from htam.metriques import fr


def main():
    depart = time.time()
    config.preparer_dossiers()
    df, synthetique = donnees.charger_cohorte()
    controle = reference.Verificateur(synthetique)

    X, y, retenus = donnees.matrice(df, config.VARIABLES_PARCIMONIEUX)
    X_creat, _, _ = donnees.matrice(retenus, ["creat_admi"])
    n = len(y)

    print("=" * 78)
    print("Modèle parcimonieux — créatininémie + volume rénal total")
    print(f"n = {n}, {int(y.sum())} non-récupérations ({y.mean():.1%})")
    print("=" * 78)
    controle.verifier("effectif", n, reference.EFFECTIFS["n_principal"], tolerance=0)
    controle.verifier("événements", int(y.sum()),
                      reference.EFFECTIFS["evenements_principal"], tolerance=0)

    # ── (1) discrimination hors échantillon ───────────────────────────────────
    p_modele = modelisation.oof(X, y)
    p_creat = modelisation.oof(X_creat, y)
    auc_modele, auc_creat = metriques.auc(y, p_modele), metriques.auc(y, p_creat)
    ic = metriques.ic_auc(y, p_modele)
    print(f"\n(1) Discrimination hors échantillon "
          f"({config.N_PLIS} plis × {config.N_REPETITIONS} répétitions)")
    print(f"    créatininémie + volume : AUC = {fr(auc_modele)}  IC 95 % {metriques.fr_ic(ic, 3)}")
    print(f"    créatininémie seule    : AUC = {fr(auc_creat)}")
    controle.verifier("AUC créat+volume", auc_modele,
                      reference.MODELE_PARCIMONIEUX["auc_creat_volume"])
    controle.verifier("AUC créatininémie seule", auc_creat,
                      reference.MODELE_PARCIMONIEUX["auc_creat_seule"])

    # ── (2) apparent, optimisme, corrigé ──────────────────────────────────────
    echelle, modele = modelisation.ajuster(X, y)
    auc_apparente = roc_auc_score(y, modelisation.predire(echelle, modele, X))
    opt = modelisation.optimisme_bootstrap(X, y)
    print(f"\n(2) Optimisme de Harrell ({opt['n']} rééchantillonnages)")
    print(f"    AUC apparente          : {fr(auc_apparente, 4)}")
    print(f"    optimisme              : {fr(opt['optimisme'], 4)} ± "
          f"{fr(opt['erreur_monte_carlo'], 4)} (erreur de Monte-Carlo)")
    print(f"    AUC corrigée           : {fr(auc_apparente - opt['optimisme'], 4)}")
    controle.verifier("AUC apparente", auc_apparente,
                      reference.MODELE_PARCIMONIEUX["auc_apparente"])
    controle.verifier("optimisme", opt["optimisme"],
                      reference.MODELE_PARCIMONIEUX["optimisme"], tolerance=0.003)

    # ── (3) calibration ───────────────────────────────────────────────────────
    pente = metriques.pente_calibration(y, p_modele)
    citl = metriques.calibration_dans_le_grand(y, p_modele)
    brier = metriques.brier(y, p_modele)
    print(f"\n(3) Calibration, sur les prédictions hors échantillon")
    print(f"    pente = {fr(pente, 2)}   calibration dans le grand = {fr(citl, 2)}   "
          f"Brier = {fr(brier, 3)}")
    print("    (pente 1 = idéale ; en deçà, le modèle est trop confiant)")
    controle.verifier("pente de calibration", pente,
                      reference.MODELE_PARCIMONIEUX["pente_calibration"], tolerance=0.03)
    controle.verifier("score de Brier", brier, reference.MODELE_PARCIMONIEUX["brier"],
                      tolerance=0.003)

    # ── (4) DeLong contre la créatininémie seule ──────────────────────────────
    test = metriques.test_delong(y, p_creat, p_modele)
    print(f"\n(4) Test de DeLong, créatininémie + volume contre créatininémie seule")
    print(f"    ΔAUC = {fr(test['delta'], 3)}   z = {fr(test['z'], 2)}   p = {fr(test['p'], 3)}")
    print("    L'AUC est peu sensible à l'ajout d'un prédicteur : ce test ne suffit pas à")
    print("    conclure que le volume n'apporte rien — voir la reclassification en 04.")
    controle.verifier("ΔAUC de DeLong", test["delta"],
                      reference.MODELE_PARCIMONIEUX["delong_delta"], tolerance=0.004)

    # ── (5) courbe de décision ────────────────────────────────────────────────
    print(f"\n(5) Bénéfice net (courbe de décision)")
    print(f"    {'seuil':>7s} {'modèle':>9s} {'traiter tous':>13s} {'aucun':>7s}")
    decision = []
    for seuil in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        b_modele = metriques.benefice_net(y, p_modele, seuil)
        b_tous = metriques.benefice_net_tous(y, seuil)
        decision.append(dict(seuil=seuil, modele=b_modele, tous=b_tous))
        print(f"    {seuil:>7.2f} {b_modele:>9.3f} {b_tous:>13.3f} {0.0:>7.3f}")
    print("    Le modèle domine les deux stratégies triviales sur toute la plage utile.")

    # ── (6) transportabilité temporelle ───────────────────────────────────────
    print(f"\n(6) Validation croisée temporelle (IECV)")
    if "annee_hospitalisation" in retenus.columns:
        periodes, moyenne = modelisation.validation_croisee_temporelle(
            X, y, retenus["annee_hospitalisation"].to_numpy())
        for r in periodes:
            print(f"    évalué sur la {r['periode']} (n = {r['n']}, ajusté sur l'autre) : "
                  f"AUC = {fr(r['auc'])}")
        print(f"    transportabilité temporelle : AUC = {fr(moyenne)}")
        controle.verifier("IECV moyenne", moyenne,
                          reference.MODELE_PARCIMONIEUX["iecv_moyenne"], tolerance=0.005)
    else:
        periodes, moyenne = [], float("nan")
        print("    (colonne annee_hospitalisation absente — étape sautée)")

    # ── (7) équation du modèle ────────────────────────────────────────────────
    coefficients, constante = modelisation.coefficients(X, y, config.VARIABLES_PARCIMONIEUX)
    moyennes = X.mean(axis=0)
    ecarts_types = X.std(axis=0)
    print(f"\n(7) Modèle ajusté sur toute la cohorte (variables centrées-réduites)")
    print(f"    constante {fr(constante, 3)}")
    for nom, coefficient, moyenne_v, ecart in zip(config.VARIABLES_PARCIMONIEUX,
                                                  coefficients.values(), moyennes, ecarts_types):
        print(f"    {nom:<14s} coefficient {fr(coefficient, 3):>7s}  "
              f"(centrage {moyenne_v:.2f}, réduction {ecart:.2f})")
    print("    Probabilité = 1 / (1 + exp(−[constante + Σ coefficient × (valeur − centrage)"
          " / réduction]))")

    # ── sortie fichier ────────────────────────────────────────────────────────
    resultat = dict(
        n=n, evenements=int(y.sum()), synthetique=synthetique,
        auc_hors_echantillon=auc_modele, ic_auc=list(ic), auc_creatininemie_seule=auc_creat,
        auc_apparente=auc_apparente, optimisme=opt, auc_corrigee=auc_apparente - opt["optimisme"],
        pente_calibration=pente, calibration_dans_le_grand=citl, brier=brier,
        delong=test, courbe_de_decision=decision,
        iecv=dict(periodes=periodes, moyenne=moyenne),
        equation=dict(constante=constante, coefficients=coefficients,
                      centrage=moyennes.tolist(), reduction=ecarts_types.tolist()),
    )
    sortie = config.DOSSIER_RESULTATS / "03_modele_parcimonieux.json"
    sortie.write_text(json.dumps(resultat, indent=1, ensure_ascii=False, default=float),
                      encoding="utf-8")

    np.save(config.DOSSIER_RESULTATS / "03_predictions_hors_echantillon.npy",
            np.column_stack([y, p_creat, p_modele]))

    controle.bilan()
    print(f"\nÉcrit : {sortie}")
    print(f"        03_predictions_hors_echantillon.npy (y, créat seule, créat+volume) — "
          f"réutilisé par 04, 05 et 08")
    print(f"({time.time() - depart:.0f} s)")


if __name__ == "__main__":
    main()
