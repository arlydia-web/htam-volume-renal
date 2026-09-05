#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 — Ce que le volume rénal change au classement des patients : NRI et IDI.

POURQUOI CES INDICES. L'aire sous la courbe ROC est une statistique de rang : elle bouge très
peu quand on ajoute un prédicteur à un modèle déjà bon, même quand ce prédicteur déplace
réellement des patients d'une catégorie de risque à l'autre. Sur cette cohorte, l'AUC ne
distingue pas la créatininémie seule du modèle à deux variables ; les indices de
reclassification, eux, montrent que dix-sept patients changent de catégorie dans le bon sens.

CE QUI EST CALCULÉ
  · NRI continu : proportion nette de patients dont le risque prédit bouge dans le bon sens ;
  · NRI catégoriel aux seuils 0,20 / 0,50 / 0,80 : le même bilan, compté entre catégories de
    risque — c'est celui qui a un sens clinique, car il compte des changements de décision ;
  · IDI : de combien l'écart moyen de risque prédit entre malades et non-malades s'élargit ;
  · la table de reclassification complète, à lire ligne par ligne.

LA RÉSERVE À NE PAS OMETTRE. Ces indices ont été critiqués comme tests de significativité :
leur distribution sous l'hypothèse nulle est mal calibrée, et un NRI positif s'obtient trop
facilement. Ils sont rapportés ici pour DÉCRIRE un déplacement, avec leur intervalle de
confiance, jamais comme preuve à eux seuls.

Lancer :  python scripts/04_reclassification.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htam import config, donnees, metriques, modelisation, reference
from htam.metriques import fr


def main():
    config.preparer_dossiers()
    df, synthetique = donnees.charger_cohorte()
    controle = reference.Verificateur(synthetique)

    y, p_creat, p_modele, _ = modelisation.predictions_parcimonieux(df)

    print("=" * 78)
    print("Reclassification — créatininémie seule  →  créatininémie + volume")
    print(f"n = {len(y)}, {int(y.sum())} non-récupérations. "
          f"Prédictions hors échantillon ({config.N_PLIS} × {config.N_REPETITIONS}).")
    print("=" * 78)
    print(f"\n  AUC créatininémie seule    : {fr(metriques.auc(y, p_creat))}")
    print(f"  AUC créatininémie + volume : {fr(metriques.auc(y, p_modele))}")

    resultats = {}

    continu, part_evenements, part_non_evenements = metriques.nri_continu(y, p_creat, p_modele)
    ic, p = metriques.ic_bootstrap(y, p_creat, p_modele, metriques.nri_continu)
    print(f"\n  NRI continu    : {fr(continu)}  {metriques.fr_ic(ic, 3)}  p = {fr(p, 3)}")
    print(f"     dont non-récupérations {fr(part_evenements)}")
    print(f"     dont récupérations     {fr(part_non_evenements)}")
    resultats["nri_continu"] = dict(valeur=continu, ic=list(ic), p=p,
                                    evenements=part_evenements,
                                    non_evenements=part_non_evenements)
    controle.verifier("NRI continu", continu, reference.RECLASSIFICATION["nri_continu"],
                      tolerance=0.01)

    categoriel, _, _, net_ev, net_ne = metriques.nri_categoriel(y, p_creat, p_modele)
    ic_cat, p_cat = metriques.ic_bootstrap(y, p_creat, p_modele, metriques.nri_categoriel)
    seuils = " / ".join(fr(s, 2) for s in config.SEUILS_NRI)
    print(f"\n  NRI catégoriel (seuils {seuils}) : {fr(categoriel)}  "
          f"{metriques.fr_ic(ic_cat, 3)}  p = {fr(p_cat, 3)}")
    print(f"     non-récupérations mieux classées (net) : {net_ev:+d} patients")
    print(f"     récupérations mieux classées (net)     : {net_ne:+d} patients")
    print(f"     bilan net                              : {net_ev + net_ne:+d} patients")
    resultats["nri_categoriel"] = dict(valeur=categoriel, ic=list(ic_cat), p=p_cat,
                                       net_evenements=net_ev, net_non_evenements=net_ne)
    controle.verifier("NRI catégoriel", categoriel,
                      reference.RECLASSIFICATION["nri_categoriel"], tolerance=0.01)
    controle.verifier("bilan net (patients)", net_ev + net_ne,
                      reference.RECLASSIFICATION["nri_categoriel_net"], tolerance=0)

    valeur_idi = metriques.idi(y, p_creat, p_modele)
    ic_idi, p_idi = metriques.ic_bootstrap(y, p_creat, p_modele, metriques.idi)
    print(f"\n  IDI            : {fr(valeur_idi)}  {metriques.fr_ic(ic_idi, 3)}  "
          f"p = {fr(p_idi, 3)}")
    resultats["idi"] = dict(valeur=valeur_idi, ic=list(ic_idi), p=p_idi)
    controle.verifier("IDI", valeur_idi, reference.RECLASSIFICATION["idi"], tolerance=0.005)

    # ── table de reclassification ─────────────────────────────────────────────
    noms = ["0–20 %", "20–50 %", "50–80 %", "80–100 %"]
    c1 = np.digitize(p_creat, config.SEUILS_NRI)
    c2 = np.digitize(p_modele, config.SEUILS_NRI)
    print(f"\n  Table de reclassification — lignes : créatininémie seule, "
          f"colonnes : créatininémie + volume")
    tables = {}
    for libelle, masque in (("non-récupérations", y == 1), ("récupérations", y == 0)):
        print(f"\n     {libelle} (n = {int(masque.sum())})")
        print("        " + " " * 11 + "".join(f"{n:>10s}" for n in noms))
        table = []
        for a in range(4):
            ligne = [int(np.sum((c1 == a) & (c2 == b) & masque)) for b in range(4)]
            table.append(ligne)
            print(f"        {noms[a]:>10s} " + "".join(f"{v:>10d}" for v in ligne))
        tables[libelle] = table
    print("\n     Au-dessus de la diagonale : patients dont le risque prédit MONTE quand on")
    print("     ajoute le volume ; en dessous : patients dont il descend.")
    resultats["table"] = tables

    sortie = config.DOSSIER_RESULTATS / "04_reclassification.json"
    sortie.write_text(json.dumps(resultats, indent=1, ensure_ascii=False, default=float),
                      encoding="utf-8")
    controle.bilan()
    print(f"\nÉcrit : {sortie}")


if __name__ == "__main__":
    main()
