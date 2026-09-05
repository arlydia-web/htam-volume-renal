# Volume rénal et pronostic de l'hypertension artérielle maligne

Code d'analyse d'une thèse de médecine sur la valeur pronostique du **volume rénal total** mesuré au scanner, chez des patients hospitalisés pour **hypertension artérielle maligne avec insuffisance rénale aiguë**.

Critère de jugement : la **non-récupération rénale à six mois**, dialyse chronique ou débit
de filtration glomérulaire estimé inférieur à 15 mL/min/1,73 m².

Un jeu **synthétique** de même structure permet d'exécuter la totalité des analyses (voir [`donnees/README.md`](donnees/README.md)).

---

## Installation et exécution

```bash
conda env create -f environment.yml
conda activate htam
python outils/verifier_installation.py
python lancer_tout.py --rapide
```

`--rapide` exécute tout sauf les trois analyses radiomiques (06, 07 et 09), qui sont longues. Pour une seule analyse :

```bash
python scripts/03_modele_parcimonieux.py
```

Les sorties sont écrites dans `resultats/` (CSV, JSON, figures en PNG/PDF/SVG).

---

## Les analyses

| # | Script | Contenu | Durée |
|---|---|---|---|
| 01 | `01_description_cohorte.py` | tableaux descriptifs, ventilés selon le devenir rénal | quelques secondes |
| 02 | `02_analyse_univariee.py` | odds ratios univariés des prédicteurs candidats, graphique en forêt | quelques secondes |
| 03 | `03_modele_parcimonieux.py` | validation interne du modèle créatininémie + volume : discrimination, optimisme, calibration, courbe de décision, transportabilité temporelle | ~1 min |
| 04 | `04_reclassification.py` | NRI continu, NRI catégoriel, IDI, table de reclassification | < 1 min |
| 05 | `05_regle_trois_zones.py` | règle de décision à deux seuils et tables de contingence | quelques secondes |
| 06 | `06_modeles_radiomiques.py` | modèles emboîtés M1 / M2 / M3 | ~20 min |
| 07 | `07_robustesse_selection.py` | huit stratégies de sélection des paramètres radiomiques | 20–60 min |
| 08 | `08_figures.py` | courbes ROC, calibration, courbe de décision, règle à trois zones | quelques secondes |
| 09 | `09_redondance_et_stabilite.py` | dédoublonnage intra-pli des paramètres ; restriction aux paramètres stables sous perturbation du masque | ~10 min |

L'en-tête de chaque script décrit ce qu'il calcule et la méthode employée.

---

## Méthodes

Régressions logistiques sur variables centrées-réduites, évaluées par validation croisée
stratifiée à cinq plis, répétée vingt fois (graine 42). Toute étape apprise sur les données
(centrage, filtre de variance, sélection des paramètres radiomiques) est ajustée à
l'intérieur du pli d'apprentissage. Intervalles de confiance et comparaisons d'AUC par
bootstrap apparié (2 000 tirages) ; optimisme par la méthode de Harrell (2 000
rééchantillonnages) ; intervalles binomiaux exacts de Clopper-Pearson pour les proportions.
Les paramètres (graine, plan de validation, définition des modèles, seuils) sont regroupés
dans `htam/config.py`.

Sur les données réelles, les scripts vérifient qu'ils retrouvent les valeurs du manuscrit
(`htam/reference.py`) ; sur données synthétiques, cette vérification est sautée.

---

## Organisation du dépôt

```
htam-volume-renal/
├── htam/                    code partagé
│   ├── config.py            graine, plan de validation, définition des modèles, seuils
│   ├── donnees.py           lecture, schéma attendu, valeurs censurées
│   ├── modelisation.py      validation croisée, sélection intra-pli, optimisme, IECV
│   ├── metriques.py         AUC et son IC, DeLong, calibration, Clopper-Pearson, NRI/IDI
│   ├── descriptif.py        tableaux descriptifs
│   ├── figures.py           style commun des figures
│   └── reference.py         valeurs du manuscrit et vérification automatique
├── scripts/                 une analyse par fichier, numérotées dans l'ordre d'exécution
├── outils/
│   ├── verifier_installation.py          contrôle de l'environnement
│   ├── generer_donnees_synthetiques.py   génération du jeu synthétique
│   └── exporter_depuis_master.py         base source → CSV de travail (usage local)
├── donnees/                 dictionnaire des variables, jeu synthétique, ICC des paramètres radiomiques
├── resultats/               sorties régénérables (exclues du dépôt)
└── lancer_tout.py           exécute les analyses dans l'ordre
```

---

## Données

Aucune donnée de patient n'est publiée. `.gitignore` exclut le dossier `donnees/`, à
l'exception du dictionnaire des variables, des fichiers synthétiques et du fichier d'ICC,
ainsi que les formats `.xlsx`, `.dcm` et `.nii`. Un accès aux données peut être demandé à
l'autrice, sous réserve du cadre réglementaire et de l'accord du responsable de traitement.

---

## Licence et citation

Code sous licence [MIT](LICENSE). Pour citer ce code, voir [`CITATION.cff`](CITATION.cff).

---

## In English

Analysis code for a medical thesis on the prognostic value of CT-measured total kidney
volume in malignant hypertension with acute kidney injury (outcome: non-recovery of renal
function at six months, i.e. chronic dialysis or eGFR < 15 mL/min/1.73 m²). No patient data
is distributed; a synthetic dataset with the same structure allows every script to run
(see `donnees/README.md`). Comments and documentation are in French.

```bash
conda env create -f environment.yml && conda activate htam
python lancer_tout.py --rapide
```
