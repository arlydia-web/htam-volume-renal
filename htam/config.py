# -*- coding: utf-8 -*-
"""Chemins, constantes et réglages communs à toutes les analyses.

Un seul endroit fixe la graine aléatoire, le plan de validation croisée et le
nombre de rééchantillonnages : c'est ce qui garantit qu'une analyse relancée
deux ans plus tard redonne les mêmes chiffres, et qu'un lecteur n'a qu'un
fichier à ouvrir pour savoir comment tout a été réglé.
"""
import os
from pathlib import Path

# ── Emplacements ──────────────────────────────────────────────────────────────
RACINE = Path(__file__).resolve().parent.parent
DOSSIER_DONNEES = Path(os.environ.get("HTAM_DONNEES", RACINE / "donnees"))
DOSSIER_RESULTATS = Path(os.environ.get("HTAM_RESULTATS", RACINE / "resultats"))
DOSSIER_FIGURES = DOSSIER_RESULTATS / "figures"

# Les données réelles ne sont pas distribuées (voir donnees/README.md). En leur
# absence, les scripts se rabattent sur le jeu synthétique versionné, en
# l'annonçant clairement à chaque exécution.
FICHIER_COHORTE = "cohorte.csv"
FICHIER_RADIOMIQUE = "radiomique.csv"
FICHIER_COHORTE_SYNTHETIQUE = "cohorte_synthetique.csv"
FICHIER_RADIOMIQUE_SYNTHETIQUE = "radiomique_synthetique.csv"

# ── Réglages statistiques (identiques dans tout le travail) ───────────────────
GRAINE = 42
N_PLIS = 5                  # validation croisée stratifiée
N_REPETITIONS = 20          # répétitions du plan de validation croisée
N_BOOTSTRAP = 2000          # intervalles de confiance et comparaisons appariées
N_BOOTSTRAP_OPTIMISME = 2000

# ── Définition des modèles ────────────────────────────────────────────────────
CRITERE = "outcome_M6"      # 1 = non-récupération rénale à six mois (dialyse ou DFGe < 15)
VOLUME = "VRT"              # volume rénal total, cm³

# M1 : modèle clinico-biologique. M2 = M1 + volume rénal total.
VARIABLES_M1 = ["age", "sexe", "IMC", "PAS_admission", "PAD_admission", "creat_admi", "MAT"]
VARIABLES_M2 = VARIABLES_M1 + [VOLUME]

# Modèle parcimonieux, celui qui porte le résultat principal de la thèse.
VARIABLES_PARCIMONIEUX = ["creat_admi", VOLUME]

# ── Règle de décision à trois zones (§ 3.2.6 du manuscrit) ────────────────────
SEUIL_BAS = 0.31            # en deçà : non-récupération écartée
SEUIL_HAUT = 0.66           # au-delà : non-récupération retenue
SEUILS_NRI = (0.20, 0.50, 0.80)   # catégories du NRI catégoriel

# ── Familles de paramètres radiomiques (nomenclature PyRadiomics/IBSI) ────────
FAMILLES_RADIOMIQUES = {
    "forme": ("_shape_",),
    "premier_ordre": ("_firstorder_",),
    "texture": ("_glcm_", "_glrlm_", "_glszm_", "_gldm_", "_ngtdm_"),
}
K_FILTRE = 5                # paramètres retenus par les filtres à nombre fixé (§ 2.9.8)


def preparer_dossiers():
    """Crée les dossiers de sortie s'ils n'existent pas encore."""
    DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
