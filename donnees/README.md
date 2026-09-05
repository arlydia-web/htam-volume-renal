# Données

Ce dossier ne contient aucune donnée de patient. Les fichiers de travail réels
(`cohorte.csv`, `radiomique.csv`) sont exclus du dépôt par `.gitignore`.

## Contenu

| Fichier | Rôle |
|---|---|
| `dictionnaire_variables.csv` | signification, unité et type de chaque variable attendue |
| `cohorte_synthetique.csv` | jeu **simulé** de 86 patients, même structure que les données réelles |
| `radiomique_synthetique.csv` | jeu **simulé** de 79 × 107 paramètres radiomiques |
| `icc_perturbation_masque.csv` | pour chacun des 107 paramètres, coefficient de corrélation intraclasse entre le masque original et ses versions dilatées (2 et 4 mm) et érodée (2 mm) sur 80 patients ; aucune donnée individuelle, utilisé par le script 09 |

Les fichiers synthétiques sont tirés d'un modèle statistique simple
(`outils/generer_donnees_synthetiques.py`) : ils reproduisent les ordres de grandeur et les
taux de données manquantes des données réelles, et rien d'autre. Aucune ligne ne correspond
à un patient. Les chiffres obtenus en les analysant n'ont aucune valeur clinique ; chaque
script le rappelle dans sa sortie.

## Format attendu pour d'autres données

Deux fichiers, à déposer dans ce dossier :

- **`cohorte.csv`** — une ligne par patient. Colonnes indispensables : `id`, `outcome_M6`,
  `creat_admi`, `VRT`. Les autres sont facultatives ; les scripts sautent les lignes de
  tableau qu'ils ne peuvent pas calculer.
- **`radiomique.csv`** — une ligne par patient, une colonne `id` puis un paramètre par
  colonne. Les noms de colonnes suivent la nomenclature PyRadiomics
  (`original_shape_…`, `original_firstorder_…`, `original_glcm_…`), utilisée par les scripts
  pour reconnaître les familles de paramètres.

Dès que `cohorte.csv` est présent, il est utilisé à la place du fichier synthétique.
Séparateur : virgule ; encodage : UTF-8 ; virgule ou point décimal. Les valeurs censurées
(`<0,01`, `>750`) sont lues telles quelles : une valeur inférieure à un seuil de détection
est remplacée par la moitié du seuil.
