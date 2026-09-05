# -*- coding: utf-8 -*-
"""Style commun à toutes les figures.

CE QU'IL GARANTIT
  · la VIRGULE décimale sur tous les axes et toutes les annotations — le manuscrit est en
    français, ses figures aussi, et le signe moins est le vrai (U+2212) ;
  · une palette où la couleur ne code JAMAIS seule : récupération et non-récupération se
    distinguent aussi par la position et par l'étiquette, la figure reste lisible imprimée en
    noir et blanc ou vue par un lecteur daltonien ;
  · deux largeurs seulement. Une figure dessinée à douze pouces entre dans une page A4 en
    étant réduite de moitié, et ses caractères de dix points s'y impriment à cinq :
    illisibles. `enregistrer()` refuse au-delà de la largeur maximale ;
  · aucun titre incrusté : la légende du manuscrit le porte déjà.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from . import config

# Palette sobre, contrastée aussi en niveaux de gris : le bleu est plus foncé que l'ocre.
RECUPERATION = "#2f6f8f"
NON_RECUPERATION = "#c1663a"
NEUTRE = "#5b6770"
ACCENT = "#8a4b6d"
GRILLE = "#dfe4e8"
ENCRE = "#1b2733"

LARGEUR_IMPRIMEE = 6.27      # pouces disponibles dans une page A4 à marges de 2,54 cm
SIMPLE = 6.6                 # un seul panneau
DOUBLE = 7.8                 # deux ou trois panneaux
LARGEUR_MAX = 8.2


def virgule(x, _=None):
    """1234.5 → « 1234,5 » ; -0.86 → « −0,86 »."""
    return f"{x:g}".replace("-", "−").replace(".", ",")


def appliquer():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "axes.edgecolor": ENCRE,
        "axes.labelcolor": ENCRE,
        "text.color": ENCRE,
        "xtick.color": ENCRE,
        "ytick.color": ENCRE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRILLE,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def axes_virgule(ax, x=True, y=True):
    if x:
        ax.xaxis.set_major_formatter(FuncFormatter(virgule))
    if y:
        ax.yaxis.set_major_formatter(FuncFormatter(virgule))


def enregistrer(fig, nom, formats=("png", "pdf", "svg")):
    """Écrit la figure dans `resultats/figures/`, en plusieurs formats."""
    largeur = fig.get_size_inches()[0]
    if largeur > LARGEUR_MAX:
        raise ValueError(
            f"{nom} fait {largeur:.1f} pouces de large : réduite à la largeur d'une page, ses "
            f"caractères deviendraient illisibles. Utiliser SIMPLE ({SIMPLE}) ou DOUBLE ({DOUBLE}).")
    config.preparer_dossiers()
    chemins = []
    for extension in formats:
        chemin = config.DOSSIER_FIGURES / f"{nom}.{extension}"
        fig.savefig(chemin, bbox_inches="tight", facecolor="white")
        chemins.append(chemin)
    plt.close(fig)
    print(f"  figure écrite : {nom}." + "/.".join(formats))
    return chemins
