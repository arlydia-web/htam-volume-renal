# -*- coding: utf-8 -*-
"""Lecture et mise en forme des données.

Deux fichiers suffisent à tout reproduire :

  · `cohorte.csv`     — une ligne par patient, les variables cliniques et le critère de jugement ;
  · `radiomique.csv`  — une ligne par patient, les paramètres radiomiques extraits du parenchyme.

Les données de patients ne sont PAS distribuées avec ce dépôt (voir `donnees/README.md`).
Si elles sont absentes, ce module charge le jeu SYNTHÉTIQUE versionné et le signale à
chaque exécution : le code tourne, les chiffres n'ont aucune valeur clinique.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Colonnes indispensables : sans elles, aucune analyse n'est possible.
COLONNES_OBLIGATOIRES = ["id", config.CRITERE, "creat_admi", config.VOLUME]


class DonneesManquantes(FileNotFoundError):
    """Aucun fichier de données trouvé, pas même le jeu synthétique."""


def _chercher(nom_reel, nom_synthetique):
    """Rend (chemin, synthetique). Les données réelles ont la priorité."""
    reel = config.DOSSIER_DONNEES / nom_reel
    if reel.exists():
        return reel, False
    synth = config.DOSSIER_DONNEES / nom_synthetique
    if synth.exists():
        return synth, True
    raise DonneesManquantes(
        f"Ni {reel} ni {synth} n'existent.\n"
        f"→ Pour un jeu de démonstration : python outils/generer_donnees_synthetiques.py\n"
        f"→ Pour les données réelles : voir donnees/README.md"
    )


def _numerique(serie):
    """Convertit en nombre, y compris les valeurs CENSURÉES des comptes rendus.

    Dix-huit haptoglobines de la cohorte sont rendues « <0,01 », « <0,1 »… (sous le seuil de
    détection) et une LDH « >750 ». Les écarter retirerait les patients les PLUS hémolytiques
    et biaiserait les médianes ; une valeur censurée est donc remplacée par la moitié du seuil,
    usage courant qui préserve l'ordre — seul l'ordre compte pour un test de rang.
    """
    def conv(v):
        s = str(v).strip().replace(",", ".")
        if s in ("", "nan", "None", "NA", "NaN"):
            return np.nan
        if s.startswith("<"):
            try:
                return float(s[1:].strip()) / 2
            except ValueError:
                return np.nan
        if s.startswith(">"):
            try:
                return float(s[1:].strip())
            except ValueError:
                return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan
    return serie.map(conv)


def charger_cohorte(chemin=None, silencieux=False):
    """Rend (DataFrame, synthetique). Le critère de jugement est toujours entier 0/1."""
    if chemin is None:
        chemin, synthetique = _chercher(config.FICHIER_COHORTE, config.FICHIER_COHORTE_SYNTHETIQUE)
    else:
        chemin, synthetique = Path(chemin), False

    df = pd.read_csv(chemin, dtype={"id": str})
    manquantes = [c for c in COLONNES_OBLIGATOIRES if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"{chemin.name} : colonnes obligatoires absentes {manquantes}.\n"
            f"Colonnes attendues et leur signification : donnees/dictionnaire_variables.csv"
        )

    for c in df.columns:
        if c not in ("id", "ethnie"):
            converti = _numerique(df[c])
            # on ne remplace que si la conversion n'a rien détruit
            if converti.notna().sum() >= df[c].notna().sum():
                df[c] = converti

    df = df[df[config.CRITERE].notna()].copy()
    df[config.CRITERE] = df[config.CRITERE].astype(int)

    if synthetique and not silencieux:
        avertir_synthetique(chemin)
    return df, synthetique


def charger_radiomique(chemin=None):
    """Rend (DataFrame indexé par `id`, synthetique)."""
    if chemin is None:
        chemin, synthetique = _chercher(config.FICHIER_RADIOMIQUE,
                                        config.FICHIER_RADIOMIQUE_SYNTHETIQUE)
    else:
        chemin, synthetique = Path(chemin), False
    df = pd.read_csv(chemin, dtype={"id": str})
    if "id" not in df.columns:
        raise ValueError(f"{chemin.name} : la première colonne doit s'appeler « id ».")
    return df.set_index("id"), synthetique


LARGEUR_BANDEAU = 76


def avertir_synthetique(chemin):
    """Bandeau impossible à manquer dans la sortie console."""
    lignes = [
        "DONNÉES SYNTHÉTIQUES : les chiffres ci-dessous ne sont PAS des résultats",
        "cliniques. Ils servent uniquement à vérifier que le code s'exécute.",
        f"Fichier : {chemin.name}",
    ]
    print("╔" + "═" * LARGEUR_BANDEAU + "╗")
    for ligne in lignes:
        print("║ " + ligne.ljust(LARGEUR_BANDEAU - 2) + " ║")
    print("╚" + "═" * LARGEUR_BANDEAU + "╝")


def matrice(df, colonnes, imputer=True):
    """Rend (X, y, df_utilisé) pour les colonnes demandées.

    Les lignes dont une variable du modèle manque sont écartées, SAUF l'indice de masse
    corporelle, imputé par la médiane : trois valeurs manquantes sur quatre-vingt-six, et les
    écarter coûterait trois patients à tous les modèles cliniques.
    """
    travail = df.copy()
    if imputer and "IMC" in colonnes and "IMC" in travail.columns:
        travail["IMC"] = travail["IMC"].fillna(travail["IMC"].median())
    garde = travail[list(colonnes) + [config.CRITERE]].notna().all(axis=1)
    travail = travail[garde]
    X = travail[list(colonnes)].to_numpy(dtype=float)
    y = travail[config.CRITERE].to_numpy(dtype=int)
    return X, y, travail


def cohorte_radiomique(df_cohorte, df_radiomique):
    """Sous-cohorte à radiomique exploitable : rend (Xr, Xc, y, noms, ids).

    Xc = les huit variables de M2 (clinique + volume) ; Xr = les paramètres radiomiques.
    Un patient n'est retenu que s'il a un critère de jugement, un volume, et une extraction
    radiomique complète — c'est la définition de la sous-cohorte des modèles M1/M2/M3.
    """
    Xc_df = df_cohorte.copy()
    absentes = [c for c in config.VARIABLES_M2 if c not in Xc_df.columns]
    if absentes:
        raise ValueError(
            f"Les modèles M1/M2/M3 demandent des colonnes absentes du fichier : {absentes}.\n"
            f"Elles sont décrites dans donnees/dictionnaire_variables.csv. Les analyses 01 à 05 "
            f"n'en ont pas besoin et fonctionnent sans.")
    if "IMC" in Xc_df.columns:
        Xc_df["IMC"] = Xc_df["IMC"].fillna(Xc_df["IMC"].median())

    besoin = config.VARIABLES_M2 + [config.CRITERE]
    Xc_df = Xc_df[Xc_df[besoin].notna().all(axis=1)]

    radio = df_radiomique.apply(pd.to_numeric, errors="coerce")
    radio = radio[radio.notna().all(axis=1)]

    communs = [i for i in Xc_df["id"] if i in radio.index]
    if not communs:
        raise ValueError(
            "Aucun identifiant commun entre le fichier de cohorte et le fichier radiomique.\n"
            "La colonne « id » doit porter les mêmes valeurs dans les deux fichiers.")
    Xc_df = Xc_df[Xc_df["id"].isin(communs)].sort_values("id")
    radio = radio.loc[Xc_df["id"].tolist()]

    return (radio.to_numpy(dtype=float),
            Xc_df[config.VARIABLES_M2].to_numpy(dtype=float),
            Xc_df[config.CRITERE].to_numpy(dtype=int),
            list(radio.columns),
            Xc_df["id"].tolist())


def indices_famille(noms, famille):
    """Indices des paramètres radiomiques d'une famille (forme, premier ordre, texture)."""
    motifs = config.FAMILLES_RADIOMIQUES[famille]
    return [i for i, n in enumerate(noms) if any(m in n for m in motifs)]
