#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fabrique un jeu de données SYNTHÉTIQUE ayant la même forme que les données réelles.

À QUOI ÇA SERT. Les données de patients ne peuvent pas être publiées. Sans elles, un lecteur
qui télécharge ce dépôt ne peut même pas vérifier que le code s'exécute. Ce générateur produit
donc deux fichiers de la même structure exactement — mêmes colonnes, mêmes noms de paramètres
radiomiques, mêmes ordres de grandeur, mêmes taux de données manquantes — mais où AUCUNE ligne
ne correspond à un patient.

CE QUE CES DONNÉES SONT, ET CE QU'ELLES NE SONT PAS
  · elles reproduisent des ordres de grandeur publiés (moyennes, écarts-types, prévalences) et
    la corrélation négative connue entre créatininémie et volume rénal ;
  · elles sont tirées d'un modèle simple : elles ne contiennent donc RIEN de ce qui fait
    l'intérêt des vraies données, et les AUC qu'on en tire n'ont aucune valeur clinique ;
  · aucune ligne ne provient d'un patient réel, ni ne peut être rapprochée d'un patient réel.

Lancer :  python outils/generer_donnees_synthetiques.py
Écrit  :  donnees/cohorte_synthetique.csv et donnees/radiomique_synthetique.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from htam import config

N_PATIENTS = 86
N_RADIOMIQUE = 79          # patients dont l'extraction radiomique aboutit
GRAINE = 20260821

# Noms de paramètres PyRadiomics (nomenclature IBSI). Ce sont des NOMS DE MESURES, pas des
# données : les reprendre à l'identique permet aux scripts de reconnaître les familles.
PARAMETRES_RADIOMIQUES = [
    "original_firstorder_10Percentile",
    "original_firstorder_90Percentile",
    "original_firstorder_Energy",
    "original_firstorder_Entropy",
    "original_firstorder_InterquartileRange",
    "original_firstorder_Kurtosis",
    "original_firstorder_Maximum",
    "original_firstorder_Mean",
    "original_firstorder_MeanAbsoluteDeviation",
    "original_firstorder_Median",
    "original_firstorder_Minimum",
    "original_firstorder_Range",
    "original_firstorder_RobustMeanAbsoluteDeviation",
    "original_firstorder_RootMeanSquared",
    "original_firstorder_Skewness",
    "original_firstorder_TotalEnergy",
    "original_firstorder_Uniformity",
    "original_firstorder_Variance",
    "original_glcm_Autocorrelation",
    "original_glcm_ClusterProminence",
    "original_glcm_ClusterShade",
    "original_glcm_ClusterTendency",
    "original_glcm_Contrast",
    "original_glcm_Correlation",
    "original_glcm_DifferenceAverage",
    "original_glcm_DifferenceEntropy",
    "original_glcm_DifferenceVariance",
    "original_glcm_Id",
    "original_glcm_Idm",
    "original_glcm_Idmn",
    "original_glcm_Idn",
    "original_glcm_Imc1",
    "original_glcm_Imc2",
    "original_glcm_InverseVariance",
    "original_glcm_JointAverage",
    "original_glcm_JointEnergy",
    "original_glcm_JointEntropy",
    "original_glcm_MCC",
    "original_glcm_MaximumProbability",
    "original_glcm_SumAverage",
    "original_glcm_SumEntropy",
    "original_glcm_SumSquares",
    "original_gldm_DependenceEntropy",
    "original_gldm_DependenceNonUniformity",
    "original_gldm_DependenceNonUniformityNormalized",
    "original_gldm_DependenceVariance",
    "original_gldm_GrayLevelNonUniformity",
    "original_gldm_GrayLevelVariance",
    "original_gldm_HighGrayLevelEmphasis",
    "original_gldm_LargeDependenceEmphasis",
    "original_gldm_LargeDependenceHighGrayLevelEmphasis",
    "original_gldm_LargeDependenceLowGrayLevelEmphasis",
    "original_gldm_LowGrayLevelEmphasis",
    "original_gldm_SmallDependenceEmphasis",
    "original_gldm_SmallDependenceHighGrayLevelEmphasis",
    "original_gldm_SmallDependenceLowGrayLevelEmphasis",
    "original_glrlm_GrayLevelNonUniformity",
    "original_glrlm_GrayLevelNonUniformityNormalized",
    "original_glrlm_GrayLevelVariance",
    "original_glrlm_HighGrayLevelRunEmphasis",
    "original_glrlm_LongRunEmphasis",
    "original_glrlm_LongRunHighGrayLevelEmphasis",
    "original_glrlm_LongRunLowGrayLevelEmphasis",
    "original_glrlm_LowGrayLevelRunEmphasis",
    "original_glrlm_RunEntropy",
    "original_glrlm_RunLengthNonUniformity",
    "original_glrlm_RunLengthNonUniformityNormalized",
    "original_glrlm_RunPercentage",
    "original_glrlm_RunVariance",
    "original_glrlm_ShortRunEmphasis",
    "original_glrlm_ShortRunHighGrayLevelEmphasis",
    "original_glrlm_ShortRunLowGrayLevelEmphasis",
    "original_glszm_GrayLevelNonUniformity",
    "original_glszm_GrayLevelNonUniformityNormalized",
    "original_glszm_GrayLevelVariance",
    "original_glszm_HighGrayLevelZoneEmphasis",
    "original_glszm_LargeAreaEmphasis",
    "original_glszm_LargeAreaHighGrayLevelEmphasis",
    "original_glszm_LargeAreaLowGrayLevelEmphasis",
    "original_glszm_LowGrayLevelZoneEmphasis",
    "original_glszm_SizeZoneNonUniformity",
    "original_glszm_SizeZoneNonUniformityNormalized",
    "original_glszm_SmallAreaEmphasis",
    "original_glszm_SmallAreaHighGrayLevelEmphasis",
    "original_glszm_SmallAreaLowGrayLevelEmphasis",
    "original_glszm_ZoneEntropy",
    "original_glszm_ZonePercentage",
    "original_glszm_ZoneVariance",
    "original_ngtdm_Busyness",
    "original_ngtdm_Coarseness",
    "original_ngtdm_Complexity",
    "original_ngtdm_Contrast",
    "original_ngtdm_Strength",
    "original_shape_Elongation",
    "original_shape_Flatness",
    "original_shape_LeastAxisLength",
    "original_shape_MajorAxisLength",
    "original_shape_Maximum2DDiameterColumn",
    "original_shape_Maximum2DDiameterRow",
    "original_shape_Maximum2DDiameterSlice",
    "original_shape_Maximum3DDiameter",
    "original_shape_MeshVolume",
    "original_shape_MinorAxisLength",
    "original_shape_Sphericity",
    "original_shape_SurfaceArea",
    "original_shape_SurfaceVolumeRatio",
    "original_shape_VoxelVolume",
]


def tirer_cohorte(rng):
    """Cohorte clinique. Les paramètres sont des ordres de grandeur, arrondis."""
    n = N_PATIENTS

    # Créatininémie : très asymétrique, d'où la loi log-normale.
    creatinine = np.clip(np.exp(rng.normal(np.log(480), 0.70, n)), 60, 2800)
    # Volume rénal total, corrélé négativement à la créatininémie (r ≈ −0,4 sur log).
    z_creatinine = (np.log(creatinine) - np.log(creatinine).mean()) / np.log(creatinine).std()
    volume = np.clip(240 - 33 * z_creatinine + rng.normal(0, 73, n), 80, 520)

    # Critère de jugement : modèle logistique aux coefficients standardisés du modèle publié.
    z_volume = (volume - volume.mean()) / volume.std()
    logit = 0.22 + 1.57 * z_creatinine - 1.00 * z_volume
    critere = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    donnees = pd.DataFrame({
        "id": [f"SYNTH_{i + 1:03d}" for i in range(n)],
        config.CRITERE: critere,
        "creat_admi": creatinine.round(0),
        config.VOLUME: volume.round(2),
        "age": np.clip(rng.normal(39.6, 11.6, n), 16, 70).round(0),
        "sexe": rng.binomial(1, 0.26, n),                     # 0 = homme, 1 = femme
        "IMC": np.clip(rng.normal(25.8, 6.0, n), 16, 55).round(1),
        "PAS_admission": np.clip(rng.normal(211, 26, n), 140, 275).round(0),
        "PAD_admission": np.clip(rng.normal(131, 20, n), 85, 175).round(0),
        "MAT": rng.binomial(1, 0.70, n),
        "Hb_adm": np.clip(rng.normal(10.2, 2.7, n), 5, 18).round(1),
        "plaquettes_adm": np.clip(rng.normal(181, 68, n), 40, 410).round(0),
        "LDH_adm": np.clip(np.exp(rng.normal(np.log(470), 0.6, n)), 120, 5300).round(0),
        "hapto_basse": rng.binomial(1, 0.65, n),
        "PU_admission_g_par_g": np.clip(np.exp(rng.normal(np.log(2.6), 0.9, n)), 0, 22).round(2),
        "annee_hospitalisation": rng.choice(
            [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            n, p=np.array([4, 3, 2, 2, 5, 4, 12, 15, 10, 25, 4]) / 86),
        "dialyse_hosp": np.zeros(n, dtype=int),
        "retinopathie": rng.binomial(1, 0.93, n),
        "oedeme_papillaire": rng.binomial(1, 0.45, n),
        "cephalees": rng.binomial(1, 0.66, n),
        "PRES": rng.binomial(1, 0.17, n),
        "HVG": rng.binomial(1, 0.82, n),
        "HTA_connue": rng.binomial(1, 0.50, n),
        "sans_suivi_anterieur": rng.binomial(1, 0.28, n),
        "diabete": rng.binomial(1, 0.04, n),
        "tabac_actif": rng.binomial(1, 0.37, n),
        "ethnie": rng.choice(["Afrique sub saharienne", "Caucasien", "Afrique du nord",
                              "Antilles", "Asie", "Autre"], n,
                             p=[0.34, 0.22, 0.17, 0.13, 0.08, 0.06]),
    })
    # La dialyse pendant le séjour dépend fortement du devenir : on la tire conditionnellement,
    # sinon la cohorte simulée serait cliniquement incohérente.
    donnees["dialyse_hosp"] = rng.binomial(1, np.where(critere == 1, 0.55, 0.12))

    # Données manquantes, aux taux observés dans la cohorte réelle.
    for colonne, nombre in (("IMC", 3), ("plaquettes_adm", 8), ("LDH_adm", 10),
                            ("hapto_basse", 4), ("PU_admission_g_par_g", 5)):
        donnees.loc[rng.choice(n, nombre, replace=False), colonne] = np.nan
    return donnees


def tirer_radiomique(rng, cohorte):
    """Paramètres radiomiques : trois facteurs latents, dont un lié au volume rénal."""
    identifiants = list(cohorte["id"][:N_RADIOMIQUE])
    volume = cohorte[config.VOLUME].to_numpy()[:N_RADIOMIQUE]
    z_volume = (volume - volume.mean()) / volume.std()
    n = len(identifiants)

    facteurs = np.column_stack([z_volume, rng.normal(0, 1, n), rng.normal(0, 1, n)])
    colonnes = {}
    for indice, nom in enumerate(PARAMETRES_RADIOMIQUES):
        poids = rng.normal(0, 1, 3)
        # Les paramètres de forme suivent le volume de près : c'est ce qui rend la radiomique
        # largement redondante avec lui, et ce que la thèse met en évidence.
        if "_shape_" in nom:
            poids[0] = abs(poids[0]) + 1.2
        valeurs = facteurs @ poids + rng.normal(0, 0.6, n)
        echelle = float(10.0 ** rng.integers(-1, 4))          # ordres de grandeur variés, comme en vrai
        colonnes[nom] = (valeurs * echelle + 5 * echelle).round(6)
    return pd.DataFrame(colonnes, index=pd.Index(identifiants, name="id")).reset_index()


def main():
    rng = np.random.default_rng(GRAINE)
    config.DOSSIER_DONNEES.mkdir(parents=True, exist_ok=True)

    cohorte = tirer_cohorte(rng)
    radiomique = tirer_radiomique(rng, cohorte)

    chemin_cohorte = config.DOSSIER_DONNEES / config.FICHIER_COHORTE_SYNTHETIQUE
    chemin_radiomique = config.DOSSIER_DONNEES / config.FICHIER_RADIOMIQUE_SYNTHETIQUE
    cohorte.to_csv(chemin_cohorte, index=False)
    radiomique.to_csv(chemin_radiomique, index=False)

    evenements = int(cohorte[config.CRITERE].sum())
    print(f"Écrit : {chemin_cohorte}")
    print(f"        {len(cohorte)} patients simulés, {evenements} non-récupérations "
          f"({evenements / len(cohorte):.0%})")
    print(f"Écrit : {chemin_radiomique}")
    print(f"        {len(radiomique)} patients × {len(PARAMETRES_RADIOMIQUES)} paramètres")
    print("\nCes données sont SIMULÉES. Elles servent à vérifier que le code s'exécute ;")
    print("aucun chiffre qui en sort n'a de valeur clinique.")


if __name__ == "__main__":
    main()
