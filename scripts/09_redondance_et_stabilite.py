#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09 — Deux contre-épreuves du résultat négatif de la radiomique.

LA QUESTION. M3 n'ajoute rien à M2 (scripts 06 et 07). Deux objections restent possibles.
La première : les 107 paramètres comportent des doublons (volume de voxels et volume du
maillage, variantes d'un même indice de longueur de plage), et une sélection pénalisée
répartit arbitrairement le signal entre paramètres jumeaux, ce qui peut diluer une information
réelle. La seconde : beaucoup de paramètres de texture reflètent le tracé du contour plus que
le tissu, et le signal utile aurait été noyé dans ce bruit de segmentation.

CE QUE FAIT CE SCRIPT
  (1) Redondance. Dans chaque pli d'apprentissage, avant la sélection LASSO, les paramètres
      dont la corrélation de rang dépasse 0,85 en valeur absolue sont dédoublonnés (règle
      déterministe : on écarte celui des deux qui est le plus corrélé au reste du jeu).
  (2) Stabilité. Ne sont proposés à la sélection que les paramètres dont le coefficient de
      corrélation intraclasse sous perturbation du masque (dilatation de 2 et 4 mm, érosion
      de 2 mm, 80 patients) est au moins de 0,75. Ces ICC ont été calculés une fois pour toutes
      à partir des images et sont livrés dans `donnees/icc_perturbation_masque.csv` ; les
      images elles-mêmes ne sont pas distribuées.

Dans les deux cas, M1 et M2 sont recalculés comme contrôle : ils doivent retomber sur les
valeurs des scripts 06 et 07, puisque seul le bloc radiomique change.

À LIRE. Après dédoublonnage, M3 vaut 0,813 ; restreint aux 23 paramètres stables, 0,824. Ni
l'un ni l'autre ne dépasse M2 (0,847). Le résultat négatif ne tient donc ni aux doublons ni au
bruit de segmentation : ce sont les conditions les plus favorables à la radiomique, et elle
n'y ajoute toujours rien au volume.

Lancer :  python scripts/09_redondance_et_stabilite.py    (compter dix à vingt minutes)
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htam import config, donnees, metriques, modelisation, reference
from htam.metriques import fr, fr_signe

FICHIER_ICC = "icc_perturbation_masque.csv"


def parametres_stables(seuil):
    chemin = config.DOSSIER_DONNEES / FICHIER_ICC
    if not chemin.exists():
        chemin = config.RACINE / "donnees" / FICHIER_ICC
    with open(chemin, newline="", encoding="utf-8") as fichier:
        lignes = list(csv.DictReader(fichier))
    return {l["parametre"] for l in lignes if float(l["ICC"]) >= seuil}, len(lignes)


def main():
    depart = time.time()
    config.preparer_dossiers()
    cohorte, synth_c = donnees.charger_cohorte()
    radiomique, synth_r = donnees.charger_radiomique()
    controle = reference.Verificateur(synth_c or synth_r)
    attendu = reference.REDONDANCE_ET_STABILITE

    Xr, Xc, y, noms, _ = donnees.cohorte_radiomique(cohorte, radiomique)
    n_clinique = len(config.VARIABLES_M1)
    print(f"\nn = {len(y)}, {int(y.sum())} non-récupérations, {Xr.shape[1]} paramètres radiomiques.\n")

    p_M1, _ = modelisation.oof_radiomique(Xc[:, :n_clinique], None, y, None)
    p_M2, _ = modelisation.oof_radiomique(Xc, None, y, None)
    p_M3, k_M3 = modelisation.oof_radiomique(Xc, Xr, y, "lasso")
    auc_M1, auc_M2, auc_M3 = (metriques.auc(y, p) for p in (p_M1, p_M2, p_M3))
    print(f"{'M1 (clinique seul)':<44s} AUC = {fr(auc_M1)}")
    print(f"{'M2 (clinique + volume)':<44s} AUC = {fr(auc_M2)}")
    print(f"{'M3 (M2 + LASSO sur les 107 paramètres)':<44s} AUC = {fr(auc_M3)}   "
          f"({k_M3:.1f} param./pli)", flush=True)
    controle.verifier("M1", auc_M1, reference.MODELES_RADIOMIQUES["M1"])
    controle.verifier("M2", auc_M2, reference.MODELES_RADIOMIQUES["M2"])
    controle.verifier("M3 conforme", auc_M3, reference.MODELES_RADIOMIQUES["M3_conforme"])

    # ── (1) dédoublonnage intra-pli ───────────────────────────────────────────
    print(f"\n(1) Retrait des paramètres redondants (|rho| > {attendu['seuil_rho']}) dans chaque pli",
          flush=True)
    p_red, k_red, survivants = modelisation.oof_radiomique(Xc, Xr, y, "lasso",
                                                            seuil_redondance=attendu["seuil_rho"])
    auc_red = metriques.auc(y, p_red)
    c_red = metriques.delta_auc(y, p_M2, p_red)
    print(f"    {survivants:.1f} paramètres survivent au filtre par pli, {k_red:.1f} sont ensuite "
          f"retenus par le LASSO")
    print(f"    AUC = {fr(auc_red)}   ΔAUC vs M2 = {fr_signe(c_red['delta'])}   "
          f"IC 95 % [{fr_signe(c_red['bas'])} ; {fr_signe(c_red['haut'])}]   p = {fr(c_red['p'], 2)}",
          flush=True)
    controle.verifier("M3 après dédoublonnage", auc_red, attendu["M3_apres_filtrage_redondance"],
                      tolerance=0.004)

    # ── (2) paramètres stables sous perturbation du masque ────────────────────
    stables, n_total = parametres_stables(attendu["seuil_icc"])
    idx = [j for j, nom in enumerate(noms) if nom in stables]
    print(f"\n(2) Paramètres stables sous perturbation du masque (ICC ≥ {attendu['seuil_icc']}) : "
          f"{len(idx)} sur {n_total}", flush=True)
    if not idx:
        print("    Aucun nom de paramètre ne correspond au fichier d'ICC : les colonnes du fichier "
              "radiomique doivent suivre la nomenclature PyRadiomics.")
    else:
        p_sta, k_sta = modelisation.oof_radiomique(Xc, Xr[:, idx], y, "lasso")
        auc_sta = metriques.auc(y, p_sta)
        c_sta = metriques.delta_auc(y, p_M2, p_sta)
        c_sta_M3 = metriques.delta_auc(y, p_M3, p_sta)
        print(f"    AUC = {fr(auc_sta)}   ({k_sta:.1f} param./pli)")
        print(f"    ΔAUC vs M2 = {fr_signe(c_sta['delta'])}   IC 95 % [{fr_signe(c_sta['bas'])} ; "
              f"{fr_signe(c_sta['haut'])}]   p = {fr(c_sta['p'], 2)}")
        print(f"    ΔAUC vs M3 (107 paramètres) = {fr_signe(c_sta_M3['delta'])}   "
              f"IC 95 % [{fr_signe(c_sta_M3['bas'])} ; {fr_signe(c_sta_M3['haut'])}]   "
              f"p = {fr(c_sta_M3['p'], 2)}", flush=True)
        controle.verifier("M3 restreint aux paramètres stables", auc_sta,
                          attendu["M3_parametres_stables"], tolerance=0.004)

    print("\nNi le dédoublonnage ni la restriction aux paramètres stables ne font dépasser M2 : "
          "le résultat négatif ne tient ni aux doublons ni au bruit de segmentation.")

    sortie = config.DOSSIER_RESULTATS / "09_redondance_et_stabilite.json"
    resultat = dict(
        n=len(y), evenements=int(y.sum()), synthetique=(synth_c or synth_r),
        M1=auc_M1, M2=auc_M2, M3=auc_M3, M3_parametres_par_pli=k_M3,
        redondance=dict(seuil=attendu["seuil_rho"], auc=auc_red, survivants_par_pli=survivants,
                        retenus_par_pli=k_red, comparaison_M2=c_red),
        stabilite=(dict(seuil_icc=attendu["seuil_icc"], n_stables=len(idx), auc=auc_sta,
                        retenus_par_pli=k_sta, comparaison_M2=c_sta, comparaison_M3=c_sta_M3)
                   if idx else None),
    )
    with open(sortie, "w", encoding="utf-8") as fichier:
        json.dump(resultat, fichier, ensure_ascii=False, indent=2)

    controle.bilan()
    print(f"\nÉcrit : {sortie}")
    print(f"({time.time() - depart:.0f} s)")


if __name__ == "__main__":
    main()
