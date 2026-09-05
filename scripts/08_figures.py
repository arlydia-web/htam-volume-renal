#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08 — Les quatre figures du modèle parcimonieux.

    a. courbes ROC — créatininémie seule contre créatininémie + volume ;
    b. calibration — risque prédit contre risque observé, par quintiles ;
    c. courbe de décision — bénéfice net comparé à « traiter tout le monde » et « personne » ;
    d. règle à trois zones — où tombent les patients, et le risque observé de chaque zone.

Toutes sont tracées sur les prédictions HORS ÉCHANTILLON, jamais sur les probabilités
apparentes : une figure de calibration faite sur les patients d'entraînement est toujours
belle, et ne veut rien dire.

Toutes respectent le style du dépôt (module `htam/figures.py`) : virgule décimale, pas de
titre incrusté, largeur compatible avec une page A4, et une information jamais portée par la
seule couleur.

Lancer :  python scripts/08_figures.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htam import config, donnees, figures, metriques, modelisation
from htam.metriques import fr

import matplotlib.pyplot as plt


def figure_roc(y, p_creat, p_modele):
    fig, ax = plt.subplots(figsize=(figures.SIMPLE, 5.0))
    for p, libelle, couleur, style in (
            (p_creat, "créatininémie seule", figures.NEUTRE, "--"),
            (p_modele, "créatininémie + volume", figures.RECUPERATION, "-")):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, style, color=couleur, lw=2.2,
                label=f"{libelle} — AUC {fr(metriques.auc(y, p))}")
    ax.plot([0, 1], [0, 1], ":", color=figures.ENCRE, lw=1, alpha=0.6, label="hasard")
    ax.set_xlabel("1 − spécificité")
    ax.set_ylabel("Sensibilité")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right", frameon=False)
    figures.axes_virgule(ax)
    fig.tight_layout()
    figures.enregistrer(fig, "08a_courbes_roc")


def figure_calibration(y, p, n_groupes=5):
    ordre = np.argsort(p)
    groupes = np.array_split(ordre, n_groupes)
    predits = [p[g].mean() for g in groupes]
    observes = [y[g].mean() for g in groupes]
    # intervalle exact sur la proportion observée de chaque groupe
    bornes = [metriques.clopper_pearson(int(y[g].sum()), len(g)) for g in groupes]

    fig, ax = plt.subplots(figsize=(figures.SIMPLE, 5.0))
    ax.plot([0, 1], [0, 1], "--", color=figures.ENCRE, lw=1, alpha=0.6,
            label="calibration parfaite")
    bas = [o - b[0] for o, b in zip(observes, bornes)]
    haut = [b[1] - o for o, b in zip(observes, bornes)]
    ax.errorbar(predits, observes, yerr=[bas, haut], fmt="o", color=figures.RECUPERATION,
                ms=7, lw=1.6, capsize=3, label=f"quintiles de risque prédit (n = {len(y)})")
    pente = metriques.pente_calibration(y, p)
    ax.text(0.04, 0.93, f"pente {fr(pente, 2)}\nBrier {fr(metriques.brier(y, p))}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=figures.GRILLE))
    ax.set_xlabel("Risque prédit")
    ax.set_ylabel("Risque observé")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    figures.axes_virgule(ax)
    fig.tight_layout()
    figures.enregistrer(fig, "08b_calibration")


def figure_decision(y, p):
    seuils = np.arange(0.05, 0.90, 0.01)
    modele = [metriques.benefice_net(y, p, s) for s in seuils]
    tous = [metriques.benefice_net_tous(y, s) for s in seuils]

    fig, ax = plt.subplots(figsize=(figures.SIMPLE, 5.0))
    ax.axvspan(config.SEUIL_BAS, config.SEUIL_HAUT, color=figures.GRILLE, alpha=0.55, zorder=0)
    ax.plot(seuils, modele, "-", color=figures.RECUPERATION, lw=2.2,
            label="créatininémie + volume")
    ax.plot(seuils, tous, "--", color=figures.NEUTRE, lw=1.6, label="traiter tout le monde")
    ax.axhline(0, color=figures.ENCRE, ls=":", lw=1, label="ne traiter personne")
    ax.set_xlabel("Seuil de probabilité auquel on décide d'agir")
    ax.set_ylabel("Bénéfice net")
    ax.set_xlim(0.05, 0.89)
    ax.set_ylim(min(-0.1, min(modele) - 0.05), max(modele) + 0.10)
    # le libellé de la bande est placé APRÈS le cadrage, sinon il sort du graphique
    ax.text((config.SEUIL_BAS + config.SEUIL_HAUT) / 2, 0.04, "zone grise de la règle",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9.5,
            color=figures.NEUTRE)
    ax.legend(loc="upper right", frameon=False)
    figures.axes_virgule(ax)
    fig.tight_layout()
    figures.enregistrer(fig, "08c_courbe_de_decision")


def figure_trois_zones(y, p):
    zones = [("faible risque", 0.0, config.SEUIL_BAS),
             ("zone grise", config.SEUIL_BAS, config.SEUIL_HAUT),
             ("haut risque", config.SEUIL_HAUT, 1.0)]

    fig, ax = plt.subplots(figsize=(figures.SIMPLE, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.72, 1.75)

    # bandes de fond : une zone sur deux est teintée, la règle se lit sans légende
    for indice, (_, gauche, droite) in enumerate(zones):
        if indice % 2 == 1:
            ax.axvspan(gauche, droite, color=figures.GRILLE, alpha=0.5, zorder=0)

    rng = np.random.default_rng(config.GRAINE)
    for valeur, libelle, couleur, hauteur in (
            (0, "récupération", figures.RECUPERATION, 1.0),
            (1, "non-récupération", figures.NON_RECUPERATION, 0.0)):
        masque = y == valeur
        dispersion = hauteur + rng.uniform(-0.20, 0.20, int(masque.sum()))
        ax.plot(p[masque], dispersion, "o", color=couleur, ms=6, alpha=0.8, zorder=3,
                label=f"{libelle} (n = {int(masque.sum())})")

    for seuil in (config.SEUIL_BAS, config.SEUIL_HAUT):
        ax.axvline(seuil, color=figures.ENCRE, ls="--", lw=1.2, zorder=2)
        ax.text(seuil, -0.50, fr(seuil, 2), ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))

    for libelle, gauche, droite in zones:
        masque = (p >= gauche) & (p < droite) if droite < 1.0 else (p >= gauche)
        effectif = int(masque.sum())
        if not effectif:
            continue
        centre = (gauche + droite) / 2
        ax.text(centre, 1.62, libelle, ha="center", va="top", fontsize=10,
                color=figures.ENCRE)
        ax.text(centre, 1.42, f"{effectif} patients\nrisque observé {y[masque].mean():.0%}",
                ha="center", va="top", fontsize=9, color=figures.NEUTRE, linespacing=1.4)

    ax.set_xlabel("Probabilité prédite de non-récupération rénale (hors échantillon)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["non-\nrécupération", "récupération"], fontsize=9.5, linespacing=1.3)
    ax.grid(visible=False)
    figures.axes_virgule(ax, y=False)
    fig.tight_layout()
    figures.enregistrer(fig, "08d_regle_trois_zones")


def main():
    config.preparer_dossiers()
    figures.appliquer()
    df, synthetique = donnees.charger_cohorte()
    y, p_creat, p_modele, _ = modelisation.predictions_parcimonieux(df)
    print(f"\nFigures du modèle parcimonieux — n = {len(y)}, "
          f"{int(y.sum())} non-récupérations.\n")
    figure_roc(y, p_creat, p_modele)
    figure_calibration(y, p_modele)
    figure_decision(y, p_modele)
    figure_trois_zones(y, p_modele)
    print(f"\nÉcrit dans : {config.DOSSIER_FIGURES}")
    if synthetique:
        print("Rappel : données synthétiques — ces figures n'illustrent aucun résultat clinique.")


if __name__ == "__main__":
    main()
