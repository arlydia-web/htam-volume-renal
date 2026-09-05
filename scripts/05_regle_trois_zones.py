#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05 — La règle de décision à trois zones, et les tables de contingence dont elle sort.

L'IDÉE. Un modèle de risque ne devient utile au lit du malade que si l'on dit à partir de
quelle probabilité on agit. Plutôt qu'un seuil unique — qui force une décision là où le
modèle est le moins sûr —, la règle en retient deux et assume une zone d'incertitude :

    · en deçà de 0,31, la non-récupération est écartée (sensibilité élevée, on ne rate presque
      personne) ;
    · au-delà de 0,66, elle est retenue (spécificité élevée, on se trompe rarement en
      l'annonçant) ;
    · entre les deux, le modèle ne tranche pas, et c'est la seule réponse honnête.

CE QUE PRODUIT LE SCRIPT. Les tables 2 × 2 aux deux seuils, avec sensibilité, spécificité,
valeurs prédictives, exactitude et intervalles exacts de Clopper-Pearson, puis la table 3 × 2
de la règle complète avec le risque observé de chaque zone. Les effectifs bruts sont donnés :
sans eux, un lecteur ne peut rien recalculer ni rien contrôler.

⚠️ LA RÉSERVE À NE JAMAIS OMETTRE. Sensibilité et spécificité ne dépendent pas de la
prévalence ; les valeurs prédictives, si. Celles calculées ici valent pour une prévalence de
51 %, celle d'une unité de soins intensifs néphrologiques. Dans une population moins sévère,
la valeur prédictive positive chuterait. Les rapports de vraisemblance, eux, se transportent :
c'est pourquoi ils sont donnés.

Lancer :  python scripts/05_regle_trois_zones.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htam import config, donnees, metriques, modelisation, reference
from htam.metriques import fr


def afficher_table(table, titre):
    print(f"\n{titre}  —  seuil de probabilité {fr(table['seuil'], 2)}")
    print("  " + "-" * 74)
    print(f"  {'':<26s} {'non-récupération':>18s} {'récupération':>16s} {'total':>8s}")
    print(f"  {'prédiction ≥ seuil':<26s} {table['vp']:>18d} {table['fp']:>16d} "
          f"{table['vp'] + table['fp']:>8d}")
    print(f"  {'prédiction < seuil':<26s} {table['fn']:>18d} {table['vn']:>16d} "
          f"{table['fn'] + table['vn']:>8d}")
    print(f"  {'total':<26s} {table['vp'] + table['fn']:>18d} "
          f"{table['fp'] + table['vn']:>16d} {table['n']:>8d}")
    for libelle, cle, cle_ic in (("sensibilité", "sensibilite", "ic_sensibilite"),
                                 ("spécificité", "specificite", "ic_specificite"),
                                 ("valeur prédictive +", "vpp", "ic_vpp"),
                                 ("valeur prédictive −", "vpn", "ic_vpn"),
                                 ("exactitude", "exactitude", "ic_exactitude")):
        print(f"  {libelle:<22s} {fr(table[cle], 2)} {metriques.fr_ic(table[cle_ic], 2)}")
    print(f"  rapport de vraisemblance positif {fr(table['rv_positif'], 2)}   "
          f"négatif {fr(table['rv_negatif'], 2)}")


def main():
    config.preparer_dossiers()
    df, synthetique = donnees.charger_cohorte()
    controle = reference.Verificateur(synthetique)

    y, _, p, _ = modelisation.predictions_parcimonieux(df)
    n = len(y)

    print("=" * 78)
    print("Règle de décision à trois zones — modèle créatininémie + volume")
    print(f"n = {n} · {int(y.sum())} non-récupérations ({y.mean():.1%}) · "
          f"prédictions hors échantillon")
    print("=" * 78)

    basse = metriques.contingence(y, p, config.SEUIL_BAS)
    haute = metriques.contingence(y, p, config.SEUIL_HAUT)
    afficher_table(basse, "SEUIL BAS — écarter la non-récupération")
    afficher_table(haute, "SEUIL HAUT — retenir la non-récupération")

    controle.verifier("sensibilité au seuil bas", basse["sensibilite"],
                      reference.REGLE_TROIS_ZONES["bas_sensibilite"], tolerance=0.02)
    controle.verifier("spécificité au seuil bas", basse["specificite"],
                      reference.REGLE_TROIS_ZONES["bas_specificite"], tolerance=0.02)
    controle.verifier("sensibilité au seuil haut", haute["sensibilite"],
                      reference.REGLE_TROIS_ZONES["haut_sensibilite"], tolerance=0.02)
    controle.verifier("spécificité au seuil haut", haute["specificite"],
                      reference.REGLE_TROIS_ZONES["haut_specificite"], tolerance=0.02)

    print("\n" + "=" * 78)
    print("RÈGLE À TROIS ZONES — table 3 × 2")
    print("=" * 78)
    zones = [(f"faible risque   (p < {fr(config.SEUIL_BAS, 2)})", p < config.SEUIL_BAS),
             (f"zone grise      ({fr(config.SEUIL_BAS, 2)} – {fr(config.SEUIL_HAUT, 2)})",
              (p >= config.SEUIL_BAS) & (p < config.SEUIL_HAUT)),
             (f"haut risque     (p ≥ {fr(config.SEUIL_HAUT, 2)})", p >= config.SEUIL_HAUT)]
    print(f"  {'zone':<32s} {'non-récup.':>11s} {'récup.':>8s} {'total':>7s} {'part':>6s} "
          f"{'risque observé':>24s}")
    print("  " + "-" * 90)
    detail = []
    for libelle, masque in zones:
        effectif, evenements = int(masque.sum()), int(y[masque].sum())
        ic = metriques.clopper_pearson(evenements, effectif)
        risque = evenements / effectif if effectif else float("nan")
        detail.append(dict(zone=libelle.split("(")[0].strip(), n=effectif,
                           evenements=evenements, risque=risque, ic=list(ic)))
        print(f"  {libelle:<32s} {evenements:>11d} {effectif - evenements:>8d} {effectif:>7d} "
              f"{effectif / n:>5.0%} {risque:>12.0%} [{ic[0]:.0%} ; {ic[1]:.0%}]")

    total = sum(d["n"] for d in detail)
    print(f"\n  contrôle des effectifs : {total} = {n} ✓" if total == n
          else f"\n  ⛔ {total} ≠ {n}")
    controle.verifier("effectif zone basse", detail[0]["n"],
                      reference.REGLE_TROIS_ZONES["bas_n"], tolerance=1)
    controle.verifier("effectif zone grise", detail[1]["n"],
                      reference.REGLE_TROIS_ZONES["grise_n"], tolerance=1)
    controle.verifier("effectif zone haute", detail[2]["n"],
                      reference.REGLE_TROIS_ZONES["haut_n"], tolerance=1)

    print(f"\n  Lecture clinique : {detail[0]['n']} patients ({detail[0]['n'] / n:.0%}) sont "
          f"rassurés d'emblée ; {detail[2]['n']} ({detail[2]['n'] / n:.0%}) relèvent d'une")
    print(f"  prise en charge de suppléance anticipée ; {detail[1]['n']} "
          f"({detail[1]['n'] / n:.0%}) restent en zone grise, et c'est là que")
    print("  la biopsie et le suivi rapproché gardent tout leur intérêt.")
    print(f"\n  ⚠️ Valeurs prédictives et risques par zone valent POUR UNE PRÉVALENCE DE "
          f"{y.mean():.0%}.")

    sortie = config.DOSSIER_RESULTATS / "05_regle_trois_zones.json"
    sortie.write_text(json.dumps(dict(seuil_bas=basse, seuil_haut=haute, zones=detail,
                                      prevalence=float(y.mean())),
                                 indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    controle.bilan()
    print(f"\nÉcrit : {sortie}")


if __name__ == "__main__":
    main()
