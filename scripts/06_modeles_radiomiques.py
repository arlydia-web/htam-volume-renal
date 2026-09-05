#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06 — Les trois modèles emboîtés M1, M2, M3, et ce que coûte une fuite de sélection.

LES TROIS MODÈLES
    M1  clinico-biologique : âge, sexe, indice de masse corporelle, pressions systolique et
        diastolique, créatininémie, hémolyse biologique ;
    M2  M1 + volume rénal total ;
    M3  M2 + paramètres radiomiques, sélectionnés par LASSO REFAIT DANS CHAQUE PLI
        d'apprentissage, sans fixer d'avance leur nombre.

CE QUE CE SCRIPT DÉMONTRE, ET QUI EST LE POINT MÉTHODOLOGIQUE DU TRAVAIL

Une quatrième ligne est calculée : « M3 avec fuite de sélection ». Elle reproduit une erreur
courante — choisir les cinq paramètres radiomiques les mieux associés au critère de jugement
sur TOUTE la cohorte, puis valider proprement par validation croisée. La validation qui suit
est irréprochable ; le mal est déjà fait, puisque le jeu de paramètres a vu le devenir des
patients qui servent à l'évaluer.

L'écart entre les deux versions de M3 n'est pas une subtilité de statisticien : c'est la
différence entre un modèle qui semble égaler M2 et un modèle qui lui est inférieur. Il se
mesure ici, sur les mêmes données et le même plan de validation.

CE QUE M3 CONFORME DIT DU FOND. Ajouter une douzaine de paramètres radiomiques à un modèle de
huit variables ajusté sur environ soixante-trois sujets par pli dégrade la prédiction. L'écart
mesure un COÛT EN DEGRÉS DE LIBERTÉ, pas une information qui irait à rebours de la texture :
il ne faut pas l'invoquer comme s'il renforçait la conclusion.

SUR LES SEUILS. Les sensibilités, spécificités et exactitudes de la colonne principale sont
prises au seuil FIXE de 0,50, ce qui rend les trois lignes comparables entre elles. Les
valeurs au seuil de Youden — qui maximise sensibilité + spécificité — sont données à part :
ce seuil est choisi SUR les données évaluées, il flatte donc légèrement les modèles.

Lancer :  python scripts/06_modeles_radiomiques.py       (compter une vingtaine de minutes)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htam import config, donnees, metriques, modelisation, reference
from htam.metriques import fr, fr_signe


def mesures(y, p):
    """Les sept colonnes du tableau des modèles."""
    bas, haut = metriques.ic_auc(y, p)
    sensibilite, specificite, exactitude = metriques.au_seuil(y, p, 0.50)
    s_y, sp_y, ex_y = metriques.au_seuil(y, p, metriques.seuil_youden(y, p))
    return dict(auc=metriques.auc(y, p), ic=(bas, haut),
                sensibilite=sensibilite, specificite=specificite, exactitude=exactitude,
                sensibilite_youden=s_y, specificite_youden=sp_y, exactitude_youden=ex_y,
                pente=metriques.pente_calibration(y, p), brier=metriques.brier(y, p))


def main():
    depart = time.time()
    config.preparer_dossiers()
    cohorte, synth_c = donnees.charger_cohorte()
    radiomique, synth_r = donnees.charger_radiomique()
    controle = reference.Verificateur(synth_c or synth_r)

    Xr, Xc, y, noms, _ = donnees.cohorte_radiomique(cohorte, radiomique)
    print(f"\nSous-cohorte à radiomique exploitable : n = {len(y)}, "
          f"{int(y.sum())} non-récupérations, {Xr.shape[1]} paramètres radiomiques.")
    for famille in config.FAMILLES_RADIOMIQUES:
        print(f"    {famille:<16s} {len(donnees.indices_famille(noms, famille)):>4d} paramètres")
    controle.verifier("effectif radiomique", len(y),
                      reference.EFFECTIFS["n_radiomique"], tolerance=0)

    print("\nCalcul des prédictions hors échantillon "
          f"({config.N_PLIS} × {config.N_REPETITIONS} ajustements par modèle)…", flush=True)
    n_clinique = len(config.VARIABLES_M1)
    p1, _ = modelisation.oof_radiomique(Xc[:, :n_clinique], None, y, None)
    print("    M1 calculé", flush=True)
    p2, _ = modelisation.oof_radiomique(Xc, None, y, None)
    print("    M2 calculé", flush=True)
    p3, taille_moyenne = modelisation.oof_radiomique(Xc, Xr, y, "lasso")
    print(f"    M3 conforme calculé ({taille_moyenne:.1f} paramètres retenus par pli)",
          flush=True)
    p3_fuite, choisis = modelisation.oof_radiomique_avec_fuite(Xc, Xr, y, k=5)
    print("    M3 avec fuite calculé", flush=True)

    resultats = {"M1": mesures(y, p1), "M2": mesures(y, p2),
                 "M3": mesures(y, p3), "M3_avec_fuite": mesures(y, p3_fuite)}
    resultats["M3"]["parametres_par_pli"] = taille_moyenne
    resultats["M3_avec_fuite"]["parametres_figes"] = [noms[i] for i in choisis]

    libelles = (("M1", "M1 clinico-biologique"), ("M2", "M2 = M1 + volume"),
                ("M3", "M3 conforme (LASSO intra-pli)"),
                ("M3_avec_fuite", "M3 avec fuite de sélection ⚠"))
    print("\n" + "=" * 100)
    print(f"n = {len(y)}, {int(y.sum())} non-récupérations")
    print(f"{'modèle':<32}{'AUC':>7}{'IC 95 %':>20}{'Se':>7}{'Sp':>7}{'Exact.':>8}"
          f"{'pente':>8}{'Brier':>8}")
    print("-" * 100)
    for cle, libelle in libelles:
        r = resultats[cle]
        print(f"{libelle:<32}{fr(r['auc']):>7}{metriques.fr_ic(r['ic'], 2):>20}"
              f"{fr(r['sensibilite'], 2):>7}{fr(r['specificite'], 2):>7}"
              f"{fr(r['exactitude'], 2):>8}{fr(r['pente'], 2):>8}{fr(r['brier']):>8}")

    controle.verifier("AUC M1", resultats["M1"]["auc"], reference.MODELES_RADIOMIQUES["M1"])
    controle.verifier("AUC M2", resultats["M2"]["auc"], reference.MODELES_RADIOMIQUES["M2"])
    controle.verifier("AUC M3 conforme", resultats["M3"]["auc"],
                      reference.MODELES_RADIOMIQUES["M3_conforme"])
    controle.verifier("AUC M3 avec fuite", resultats["M3_avec_fuite"]["auc"],
                      reference.MODELES_RADIOMIQUES["M3_avec_fuite"])
    controle.verifier("paramètres retenus par pli", taille_moyenne,
                      reference.MODELES_RADIOMIQUES["M3_parametres_par_pli"], tolerance=0.5)

    print(f"\nM3 conforme : {taille_moyenne:.1f} paramètres radiomiques retenus en moyenne par "
          f"pli, nombre non fixé d'avance.")
    print("M3 avec fuite : cinq paramètres figés, choisis sur toute la cohorte —")
    for nom in resultats["M3_avec_fuite"]["parametres_figes"]:
        print(f"    {nom}")

    print("\nAu seuil de Youden (donné à titre indicatif : ce seuil est choisi sur les données)")
    for cle, libelle in libelles:
        r = resultats[cle]
        print(f"    {libelle:<32s} Se {fr(r['sensibilite_youden'], 2)}  "
              f"Sp {fr(r['specificite_youden'], 2)}  Exact. {fr(r['exactitude_youden'], 2)}")

    print("\nComparaisons deux à deux (bootstrap apparié, "
          f"{config.N_BOOTSTRAP} tirages)")
    comparaisons = {}
    for libelle, reference_p, alternatif_p in (("M2 contre M1", p1, p2),
                                               ("M3 contre M2", p2, p3),
                                               ("M3 contre M1", p1, p3),
                                               ("M3 avec fuite contre M2", p2, p3_fuite)):
        c = metriques.delta_auc(y, reference_p, alternatif_p)
        comparaisons[libelle] = c
        print(f"    {libelle:<26s} ΔAUC = {fr_signe(c['delta']):>7s}   "
              f"IC 95 % [{fr_signe(c['bas'])} ; {fr_signe(c['haut'])}]   p = {fr(c['p'], 2)}")
    controle.verifier("ΔAUC M3 − M2", comparaisons["M3 contre M2"]["delta"],
                      reference.MODELES_RADIOMIQUES["delta_M3_M2"], tolerance=0.005)

    ecart_fuite = resultats["M3_avec_fuite"]["auc"] - resultats["M3"]["auc"]
    print(f"\nCE QUE LA FUITE FAIT GAGNER : {fr_signe(ecart_fuite)} d'AUC "
          f"({fr(resultats['M3']['auc'])} → {fr(resultats['M3_avec_fuite']['auc'])}).")
    print("C'est un gain apparent, sans contrepartie chez un nouveau patient : il vient")
    print("uniquement du fait que les paramètres ont été choisis en connaissant le devenir.")

    sortie = config.DOSSIER_RESULTATS / "06_modeles_radiomiques.json"
    sortie.write_text(json.dumps(dict(n=len(y), evenements=int(y.sum()),
                                      modeles=resultats, comparaisons=comparaisons),
                                 indent=1, ensure_ascii=False, default=float),
                      encoding="utf-8")
    controle.bilan()
    print(f"\nÉcrit : {sortie}")
    print(f"({time.time() - depart:.0f} s)")


if __name__ == "__main__":
    main()
