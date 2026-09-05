#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prépare les deux fichiers de travail à partir de la base source. À N'EXÉCUTER QUE LOCALEMENT.

CE QUE FAIT CE SCRIPT. Il lit le classeur maître de la cohorte et le fichier d'extraction
radiomique, en tire les seules colonnes dont les analyses ont besoin, traduit les variables
codées en texte (« Oui »/« Non », stades de rétinopathie…) en indicatrices 0/1, et écrit
`donnees/cohorte.csv` et `donnees/radiomique.csv`.

CE QU'IL NE FAIT PAS, ET C'EST VOULU
  · il ne recopie ni identifiant hospitalier, ni date, ni texte libre : l'identifiant est
    remplacé par un numéro d'ordre, et la date d'hospitalisation réduite à son ANNÉE, qui
    suffit à la validation croisée temporelle ;
  · il n'écrit rien hors de `donnees/`, dossier que `.gitignore` exclut du dépôt. Les fichiers
    produits contiennent des données de patients : ils ne doivent JAMAIS être publiés.

UNE RÈGLE DE CODAGE À CONNAÎTRE. L'œdème papillaire : deux nomenclatures coexistent dans la
base source. Dans l'une, le stade III de la classification en trois stades désigne la
rétinopathie maligne AVEC œdème papillaire ; dans celle de Keith-Wagener-Barker, l'œdème
papillaire est au stade IV. Les fonds d'œil codés « III » sont comptés comme œdèmes
papillaires, définition consensuelle du service dont vient la cohorte. La règle est appliquée
ici une fois pour toutes, et elle déplace un chiffre du manuscrit : elle doit donc être
explicite plutôt que dispersée dans les scripts.

Lancer :  python outils/exporter_depuis_master.py --classeur <chemin.xlsx> --radiomique <chemin.csv>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from htam import config

ONGLET = "01_Donnees_84"

# colonne de sortie → colonne source, copiée telle quelle
COLONNES_DIRECTES = {
    config.CRITERE: "outcome_M6",
    "creat_admi": "creat_admi",
    config.VOLUME: "VRT",
    "age": "age",
    "sexe": "sexe",
    "IMC": "IMC",
    "PAS_admission": "PAS_admission",
    "PAD_admission": "PAD_admission",
    "MAT": "MAT",
    "Hb_adm": "Hb_adm",
    "plaquettes_adm": "plaquettes_adm",
    "LDH_adm": "LDH_adm",
    "hapto_basse": "hapto_basse",
    "PU_admission_g_par_g": "PU_admission_g_par_g",
    "ethnie": "ethnie",
}


def texte(serie):
    return serie.astype(str).str.strip()


def oedeme_papillaire(serie):
    valeurs = texte(serie)
    return (valeurs.str.contains("œdème papillaire", case=False, na=False)
            | valeurs.isin(["III", "IV", "4", "4.0"])).astype(int)


def main():
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument("--classeur", required=True, help="classeur maître (.xlsx)")
    analyseur.add_argument("--radiomique", required=True, help="extraction radiomique (.csv)")
    analyseur.add_argument("--onglet", default=ONGLET)
    analyseur.add_argument("--colonne-identifiant", default="id_TNN")
    analyseur.add_argument("--conserver-identifiants", action="store_true",
                           help="conserve les identifiants d'origine au lieu de les remplacer "
                                "par un numéro d'ordre (déconseillé)")
    arguments = analyseur.parse_args()

    source = pd.read_excel(arguments.classeur, sheet_name=arguments.onglet)
    print(f"Lu : {Path(arguments.classeur).name}, onglet {arguments.onglet} "
          f"— {len(source)} lignes, {len(source.columns)} colonnes.")

    sortie = pd.DataFrame()
    identifiants = texte(source[arguments.colonne_identifiant])
    correspondance = {ancien: (ancien if arguments.conserver_identifiants else f"P{i + 1:03d}")
                      for i, ancien in enumerate(identifiants)}
    sortie["id"] = identifiants.map(correspondance)

    for cible, origine in COLONNES_DIRECTES.items():
        if origine in source.columns:
            sortie[cible] = source[origine]
        else:
            print(f"  ⚠️ colonne source absente, ignorée : {origine}")

    # année seule : la date exacte est une donnée identifiante, l'année suffit à l'IECV
    if "date_hospitalisation" in source.columns:
        sortie["annee_hospitalisation"] = pd.to_datetime(
            source["date_hospitalisation"], errors="coerce").dt.year

    # variables codées en texte → indicatrices 0/1
    if "dialyse_hosp" in source.columns:
        sortie["dialyse_hosp"] = (texte(source["dialyse_hosp"]) == "Oui").astype(int)
    if "retinopathie_HTA_stade" in source.columns:
        stade = texte(source["retinopathie_HTA_stade"])
        sortie["retinopathie"] = (~stade.isin(["", "nan", "Stade 0 - Absente"])).astype(int)
        sortie["oedeme_papillaire"] = oedeme_papillaire(source["retinopathie_HTA_stade"])
    if "cephalees" in source.columns:
        sortie["cephalees"] = pd.to_numeric(source["cephalees"], errors="coerce")
    for cible, origine in (("PRES", "PRES"), ("HVG", "HVG")):
        if origine in source.columns:
            valeurs = texte(source[origine])
            indicatrice = (valeurs == "Oui").astype(float)
            indicatrice[valeurs.isin(["", "nan"])] = np.nan
            sortie[cible] = indicatrice
    if "HTA_pre_existante" in source.columns:
        suivi = texte(source["HTA_pre_existante"])
        sortie["HTA_connue"] = (suivi == "Oui").astype(float)
        sortie["sans_suivi_anterieur"] = (suivi == "Pas de suivi antérieur").astype(float)
        sortie.loc[suivi.isin(["", "nan"]), ["HTA_connue", "sans_suivi_anterieur"]] = np.nan
    if "diabete" in source.columns:
        valeurs = texte(source["diabete"])
        sortie["diabete"] = valeurs.str.startswith("Diabète").astype(float)
        sortie.loc[valeurs.isin(["", "nan"]), "diabete"] = np.nan
    if "tabac" in source.columns:
        valeurs = texte(source["tabac"])
        sortie["tabac_actif"] = (valeurs == "Actif").astype(float)
        sortie.loc[valeurs.isin(["", "nan"]), "tabac_actif"] = np.nan

    sortie = sortie[sortie[config.CRITERE].notna()]
    config.DOSSIER_DONNEES.mkdir(parents=True, exist_ok=True)
    chemin_cohorte = config.DOSSIER_DONNEES / config.FICHIER_COHORTE
    sortie.to_csv(chemin_cohorte, index=False)
    print(f"Écrit : {chemin_cohorte} — {len(sortie)} patients, "
          f"{int(sortie[config.CRITERE].sum())} non-récupérations.")

    # ── radiomique ────────────────────────────────────────────────────────────
    radiomique = pd.read_csv(arguments.radiomique)
    premiere = radiomique.columns[0]
    radiomique[premiere] = texte(radiomique[premiere])
    radiomique = radiomique[radiomique[premiere].isin(correspondance)]
    radiomique[premiere] = radiomique[premiere].map(correspondance)
    radiomique = radiomique.rename(columns={premiere: "id"})
    chemin_radiomique = config.DOSSIER_DONNEES / config.FICHIER_RADIOMIQUE
    radiomique.to_csv(chemin_radiomique, index=False)
    print(f"Écrit : {chemin_radiomique} — {len(radiomique)} patients, "
          f"{radiomique.shape[1] - 1} paramètres.")

    print("\n⚠️ Ces deux fichiers contiennent des données de patients.")
    print("   `.gitignore` les exclut du dépôt. Ne jamais les forcer avec « git add -f ».")


if __name__ == "__main__":
    main()
