#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01 — Description de la cohorte, ventilée selon le devenir rénal à six mois.

Trois tableaux, dans l'ordre où un clinicien lit un dossier :
    1. les patients (terrain) ;
    2. leur présentation à l'admission ;
    3. leur biologie à l'admission.

Le critère de jugement est la NON-RÉCUPÉRATION rénale à six mois : dialyse chronique ou
débit de filtration glomérulaire estimé inférieur à 15 mL/min/1,73 m².

Aucune valeur n'est imputée ici : chaque ligne porte l'effectif réellement disponible, et
c'est voulu — une donnée manquante doit se voir dans un tableau descriptif.

Lancer :  python scripts/01_description_cohorte.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htam import config, descriptif, donnees


def main():
    config.preparer_dossiers()
    df, synthetique = donnees.charger_cohorte()

    recuperation = int((df[config.CRITERE] == 0).sum())
    non_recuperation = int((df[config.CRITERE] == 1).sum())
    print(f"\nCohorte : {len(df)} patients — {non_recuperation} non-récupérations "
          f"({non_recuperation / len(df):.1%}), {recuperation} récupérations.")

    tableaux = {}

    tableaux["Tableau 1. Caractéristiques des patients"] = [
        descriptif.ligne_continue(df, "Âge (années)", "age"),
        descriptif.ligne_binaire(df, "Sexe masculin", "sexe", positif=0),
        *[descriptif.ligne_categorie(df, f"    {origine}", "ethnie", origine)
          for origine in ("Afrique sub saharienne", "Caucasien", "Afrique du nord",
                          "Antilles", "Asie")],
        descriptif.ligne_continue(df, "Indice de masse corporelle (kg/m²)", "IMC", decimales=1),
        descriptif.ligne_binaire(df, "Hypertension artérielle connue", "HTA_connue"),
        descriptif.ligne_binaire(df, "Aucun suivi médical antérieur", "sans_suivi_anterieur"),
        descriptif.ligne_binaire(df, "Diabète", "diabete"),
        descriptif.ligne_binaire(df, "Tabagisme actif", "tabac_actif"),
    ]

    tableaux["Tableau 2. Caractéristiques à l'admission"] = [
        descriptif.ligne_continue(df, "Pression artérielle systolique (mmHg)", "PAS_admission"),
        descriptif.ligne_continue(df, "Pression artérielle diastolique (mmHg)", "PAD_admission"),
        descriptif.ligne_binaire(df, "Rétinopathie hypertensive documentée", "retinopathie"),
        descriptif.ligne_binaire(df, "    dont œdème papillaire", "oedeme_papillaire"),
        descriptif.ligne_binaire(df, "Céphalées", "cephalees"),
        descriptif.ligne_binaire(df, "Encéphalopathie postérieure réversible", "PRES"),
        descriptif.ligne_binaire(df, "Hypertrophie ventriculaire gauche", "HVG"),
        descriptif.ligne_binaire(df, "Microangiopathie thrombotique biologique", "MAT"),
        descriptif.ligne_binaire(df, "Épuration extra-rénale pendant le séjour", "dialyse_hosp"),
        descriptif.ligne_continue(df, "Volume rénal total (cm³)", config.VOLUME),
    ]

    tableaux["Tableau 3. Caractéristiques biologiques à l'admission"] = [
        descriptif.ligne_continue(df, "Créatininémie (µmol/L)", "creat_admi"),
        descriptif.ligne_continue(df, "Hémoglobine (g/dL)", "Hb_adm", decimales=1),
        descriptif.ligne_continue(df, "Plaquettes (G/L)", "plaquettes_adm"),
        descriptif.ligne_continue(df, "Lactate déshydrogénase (UI/L)", "LDH_adm"),
        # L'haptoglobine est traitée en BINAIRE et non en concentration : cliniquement, ce qui
        # compte est qu'elle soit effondrée, non sa valeur. Elle est d'ailleurs souvent rendue
        # « indosable », donc censurée — inexploitable comme variable continue.
        descriptif.ligne_binaire(df, "Haptoglobine basse", "hapto_basse"),
        descriptif.ligne_continue(df, "Protéinurie (g/g de créatininurie)",
                                  "PU_admission_g_par_g", decimales=2),
    ]

    sortie = config.DOSSIER_RESULTATS / "01_description_cohorte.csv"
    with open(sortie, "w", newline="", encoding="utf-8") as fichier:
        plume = csv.writer(fichier, delimiter=";")
        plume.writerow(["tableau", "variable", "n", "récupération", "non-récupération", "p"])
        for titre, lignes in tableaux.items():
            retenues = descriptif.afficher(titre, lignes, recuperation, non_recuperation)
            for ligne in retenues:
                plume.writerow([titre] + list(ligne))

    print(f"\nÉcrit : {sortie}")
    if synthetique:
        print("Rappel : données synthétiques — aucune de ces valeurs n'est clinique.")


if __name__ == "__main__":
    main()
