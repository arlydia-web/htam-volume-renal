# Volume rénal et pronostic de l'hypertension artérielle maligne

Code d'analyse d'une thèse de médecine portant sur la valeur pronostique du **volume rénal
total** mesuré au scanner, chez des patients hospitalisés pour **hypertension artérielle
maligne avec insuffisance rénale aiguë**.

Critère de jugement : la **non-récupération rénale à six mois** — dialyse chronique ou débit
de filtration glomérulaire estimé inférieur à 15 mL/min/1,73 m².

Ce dépôt contient les méthodes, pas les patients. Les données de santé ne sont pas
distribuées ; un jeu **synthétique** de même structure permet d'exécuter la totalité des
analyses sans rien demander à personne (voir [`donnees/README.md`](donnees/README.md)).

---

## Démarrer

```bash
conda env create -f environment.yml
conda activate htam
python outils/verifier_installation.py
python lancer_tout.py --rapide
```

`--rapide` exécute tout sauf les deux analyses radiomiques, qui sont longues. Sans cette
option, compter une heure environ. Pour une seule analyse :

```bash
python scripts/03_modele_parcimonieux.py
```

Un pas-à-pas complet, depuis l'installation de conda jusqu'à la mise en ligne sur GitHub, se
trouve dans [`GUIDE_DEMARRAGE.md`](GUIDE_DEMARRAGE.md).

---

## Les analyses

| # | Script | Ce qu'il établit | Durée |
|---|---|---|---|
| 01 | `01_description_cohorte.py` | les trois tableaux descriptifs, ventilés selon le devenir rénal | quelques secondes |
| 02 | `02_analyse_univariee.py` | odds ratios univariés des prédicteurs candidats + graphique en forêt | quelques secondes |
| 03 | `03_modele_parcimonieux.py` | **validation interne du modèle créatininémie + volume** : discrimination, optimisme, calibration, courbe de décision, transportabilité temporelle | ~1 min |
| 04 | `04_reclassification.py` | NRI continu, NRI catégoriel, IDI, table de reclassification | < 1 min |
| 05 | `05_regle_trois_zones.py` | la règle de décision à deux seuils et ses tables de contingence | quelques secondes |
| 06 | `06_modeles_radiomiques.py` | **M1 / M2 / M3 emboîtés, et ce que coûte une fuite de sélection** | ~20 min |
| 07 | `07_robustesse_selection.py` | huit stratégies de sélection : la conclusion dépend-elle du sélecteur ? | 20–60 min |
| 08 | `08_figures.py` | courbes ROC, calibration, courbe de décision, règle à trois zones | quelques secondes |

Chaque script est **autonome** : son en-tête explique ce qu'il calcule, pourquoi cette méthode
plutôt qu'une autre, et ce qu'il ne faut pas lui faire dire. Les sorties vont dans
`resultats/` (CSV, JSON, figures en PNG/PDF/SVG).

---

## Ce que le code reproduit

Lancés sur les données réelles, les scripts **vérifient d'eux-mêmes** qu'ils retrouvent les
valeurs publiées (module `htam/reference.py`) et signalent tout écart supérieur à la
tolérance. Sur données synthétiques, cette vérification est sautée.

**Modèle parcimonieux — créatininémie + volume rénal total** (n = 86, 44 non-récupérations)

| | valeur |
|---|---|
| AUC hors échantillon | **0,864** [0,78 ; 0,94] |
| AUC de la créatininémie seule | 0,876 |
| AUC apparente / optimisme / corrigée | 0,876 / 0,009 / 0,867 |
| Calibration : pente, Brier | 1,08 · 0,146 |
| Transportabilité temporelle (IECV) | 0,881 |
| NRI catégoriel | +0,40 [0,15 ; 0,64] — **17 patients** nets mieux classés |

**Règle à trois zones** — seuils **0,31** et **0,66** :

| zone | patients | risque observé | |
|---|---|---|---|
| p < 0,31 | 25 (29 %) | 16 % | sensibilité 0,91 · VPN 0,84 · rapport de vraisemblance négatif 0,18 |
| 0,31 – 0,66 | 33 (38 %) | 42 % | zone grise assumée |
| p ≥ 0,66 | 28 (33 %) | 93 % | spécificité 0,95 · VPP 0,93 · rapport de vraisemblance positif 12,4 |

**Modèles emboîtés** (sous-cohorte radiomique, n = 79, 41 non-récupérations)

| modèle | AUC | |
|---|---|---|
| M1 clinico-biologique | 0,818 | |
| M2 = M1 + volume | 0,847 | ΔAUC vs M1 +0,028 [−0,04 ; +0,10], p = 0,43 |
| **M3 = M2 + radiomique (LASSO intra-pli)** | **0,810** | ΔAUC vs M2 −0,037 [−0,084 ; +0,006], p = 0,09 |
| M3 avec fuite de sélection ⚠️ | 0,847 | reproduit pour montrer l'effet de la fuite |

> **Le point méthodologique du travail.** Les deux dernières lignes sont le même modèle, sur
> les mêmes données, avec le même plan de validation. La seule différence : dans la seconde,
> les cinq paramètres radiomiques ont été choisis sur **toute** la cohorte — critère de
> jugement compris — avant la validation croisée. Ce seul geste fait gagner 0,037 d'AUC, un
> gain qui n'existe que sur le papier et ne se retrouvera jamais chez un nouveau patient.
> Le script 06 calcule les deux versions côte à côte pour que ce coût soit vérifiable.
>
> L'écart −0,037 mesure un **coût en degrés de liberté** — douze paramètres supplémentaires
> par pli, sur un modèle qui en compte déjà huit pour quarante et un événements — et non une
> information qui irait à rebours de la texture. Il ne « renforce » aucune conclusion.

---

## La méthode, en un paragraphe

Tous les modèles sont des **régressions logistiques** sur variables centrées-réduites, évaluées
par **validation croisée stratifiée à cinq plis, répétée vingt fois, graine 42**. Chaque
patient reçoit vingt prédictions faites par des modèles qui ne l'avaient pas vu ; on en prend
la moyenne. Tout ce qui apprend quelque chose des données — centrage, filtre de variance,
**sélection des paramètres** — est ajusté à l'intérieur du seul pli d'apprentissage. Les
intervalles de confiance et les comparaisons de deux modèles reposent sur un **bootstrap
apparié** de 2 000 tirages ; l'optimisme suit la méthode de Harrell, également sur 2 000
rééchantillonnages, avec son erreur de Monte-Carlo. Les proportions sont accompagnées
d'intervalles binomiaux exacts de Clopper-Pearson.

---

## Organisation du dépôt

```
htam-volume-renal/
├── htam/                    le code partagé — à lire en premier
│   ├── config.py            graine, plan de validation, définition des modèles, seuils
│   ├── donnees.py           lecture, schéma attendu, valeurs censurées
│   ├── modelisation.py      validation croisée, sélection intra-pli, optimisme, IECV
│   ├── metriques.py         AUC et son IC, DeLong, calibration, Clopper-Pearson, NRI/IDI
│   ├── descriptif.py        tableaux descriptifs
│   ├── figures.py           style commun des figures
│   └── reference.py         valeurs publiées + vérification automatique
├── scripts/                 une analyse par fichier, numérotées dans l'ordre de lecture
├── outils/
│   ├── verifier_installation.py          contrôle l'environnement avant de commencer
│   ├── generer_donnees_synthetiques.py   fabrique le jeu de démonstration
│   └── exporter_depuis_master.py         base source → CSV de travail (usage local seulement)
├── donnees/                 dictionnaire + jeu synthétique (jamais de données réelles)
├── resultats/               sorties régénérables (exclues du dépôt)
└── lancer_tout.py           exécute les analyses dans l'ordre
```

---

## Données et confidentialité

Aucune donnée de patient n'est publiée, et `.gitignore` est écrit pour que cela reste vrai :
le dossier `donnees/` est exclu à l'exception du dictionnaire et des fichiers synthétiques,
et les formats susceptibles de contenir des patients (`.xlsx`, `.dcm`, `.nii`…) sont exclus
partout. Le script d'export local ne recopie ni identifiant hospitalier, ni date, ni texte
libre : l'identifiant devient un numéro d'ordre et la date d'admission est réduite à son
année.

Un accès aux données peut être demandé à l'autrice ; il suppose un cadre réglementaire et
l'accord du responsable de traitement.

---

## Citer ce code

Voir [`CITATION.cff`](CITATION.cff). Code sous licence [MIT](LICENSE) : réutilisable, y
compris commercialement, à condition de conserver la mention de copyright.

---

## In English

Analysis code for a medical thesis on the prognostic value of **total kidney volume** (CT) in
**malignant hypertension with acute kidney injury**; outcome is non-recovery of renal function
at six months (chronic dialysis or eGFR < 15 mL/min/1.73 m²).

No patient data is distributed. A **synthetic dataset** with the same structure ships with the
repository, so every script runs out of the box — the resulting numbers are meaningless
clinically, and each script says so. See [`donnees/README.md`](donnees/README.md) to plug in
your own data (two CSV files, schema in `donnees/dictionnaire_variables.csv`).

All models are logistic regressions evaluated by 5-fold stratified cross-validation repeated
20 times (seed 42), with **all feature selection performed inside the training fold**.
Script 06 deliberately reproduces the leaked variant — selection performed on the whole cohort
before cross-validation — to quantify what that single mistake buys: +0.037 AUC, entirely
illusory. Comments and documentation are in French, matching the thesis.

```bash
conda env create -f environment.yml && conda activate htam
python lancer_tout.py --rapide
```
