# -*- coding: utf-8 -*-
"""Tableaux descriptifs ventilés selon le devenir rénal.

Deux règles tenues partout :
  · AUCUNE imputation dans les tableaux descriptifs. Chaque ligne porte son effectif réel ;
    imputer gonflerait les effectifs et déplacerait les médianes sans que rien ne le signale ;
  · variable continue → médiane [Q1 ; Q3] et test de Mann-Whitney ; variable binaire →
    n/N (%) et test exact de Fisher. Les effectifs sont trop faibles pour un test du khi-deux.
"""
import numpy as np
from scipy import stats

from . import config


def _fr(x, decimales=1):
    return f"{x:.{decimales}f}".replace(".", ",")


def _p(valeur):
    return "< 0,001" if valeur < 0.001 else _fr(valeur, 3)


def ligne_continue(df, libelle, colonne, decimales=0, transformation=None):
    """Médiane [Q1 ; Q3] par groupe et test de Mann-Whitney."""
    if colonne not in df.columns:
        return None
    critere = df[config.CRITERE]
    valeurs = df[colonne] if transformation is None else df[colonne].map(transformation)
    a = valeurs[(critere == 0) & valeurs.notna()].to_numpy(dtype=float)
    b = valeurs[(critere == 1) & valeurs.notna()].to_numpy(dtype=float)
    if len(a) < 3 or len(b) < 3:
        return None

    def resume(v):
        return (f"{_fr(np.median(v), decimales)} "
                f"[{_fr(np.percentile(v, 25), decimales)} ; {_fr(np.percentile(v, 75), decimales)}]")

    return (libelle, len(a) + len(b), resume(a), resume(b),
            _p(stats.mannwhitneyu(a, b)[1]))


def ligne_binaire(df, libelle, colonne, positif=1):
    """n/N (%) par groupe et test exact de Fisher."""
    if colonne not in df.columns:
        return None
    critere = df[config.CRITERE]
    presente = df[colonne].notna()
    groupes = []
    for valeur in (0, 1):
        masque = presente & (critere == valeur)
        total = int(masque.sum())
        if total == 0:
            return None
        atteints = int((df.loc[masque, colonne] == positif).sum())
        groupes.append((atteints, total))
    (a, na), (b, nb) = groupes
    p = stats.fisher_exact([[a, na - a], [b, nb - b]])[1]
    return (libelle, na + nb,
            f"{a}/{na} ({round(100 * a / na)} %)",
            f"{b}/{nb} ({round(100 * b / nb)} %)",
            _p(p))


def ligne_categorie(df, libelle, colonne, modalite):
    """Une modalité d'une variable qualitative, traitée comme binaire."""
    if colonne not in df.columns:
        return None
    travail = df.copy()
    travail[colonne] = (travail[colonne].astype(str).str.strip() == modalite).astype(float)
    travail.loc[df[colonne].isna(), colonne] = np.nan
    return ligne_binaire(travail, libelle, colonne, positif=1)


def afficher(titre, lignes, n_recuperation, n_non_recuperation):
    lignes = [l for l in lignes if l]
    print(f"\n{'=' * 104}")
    print(f"{titre}   (récupération n = {n_recuperation} · "
          f"non-récupération n = {n_non_recuperation})")
    print(f"{'':46s}{'n':>4s}  {'Récupération':>22s}  {'Non-récupération':>22s}  {'p':>9s}")
    print("-" * 104)
    for libelle, n, a, b, p in lignes:
        print(f"{libelle:46s}{n:>4d}  {a:>22s}  {b:>22s}  {p:>9s}")
    return lignes
