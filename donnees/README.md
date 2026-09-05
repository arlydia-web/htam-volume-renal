# Les données

## Pourquoi elles ne sont pas ici

Ce dossier ne contient **aucune donnée de patient**, et n'en contiendra jamais. La cohorte est
constituée de patients hospitalisés pour hypertension artérielle maligne : même
pseudonymisées, ces données restent des données de santé, et leur diffusion suppose une base
légale, un engagement de conformité et l'accord du responsable de traitement. Les publier sur
une plateforme comme GitHub reviendrait à s'en dessaisir.

Ce que ce dépôt publie, c'est le **code** : les méthodes, ligne à ligne, telles qu'elles ont
produit les chiffres. C'est ce qui permet à un lecteur de vérifier ce qui a été fait, de
critiquer un choix, ou d'appliquer la même méthode à sa propre cohorte.

## Ce que le dossier contient

| Fichier | Rôle |
|---|---|
| `dictionnaire_variables.csv` | la signification, l'unité et le type de chaque variable attendue |
| `cohorte_synthetique.csv` | jeu **simulé** de 86 patients, même structure que les données réelles |
| `radiomique_synthetique.csv` | jeu **simulé** de 79 × 107 paramètres radiomiques |

Les fichiers synthétiques sont tirés d'un modèle statistique simple (voir
`outils/generer_donnees_synthetiques.py`) : ils reproduisent les ordres de grandeur publiés et
les taux de données manquantes, **et rien d'autre**. Aucune ligne ne correspond à un patient.
Les scripts s'exécutent dessus de bout en bout, mais les chiffres obtenus n'ont **aucune
valeur clinique**, et chaque script le rappelle dans sa sortie.

## Faire tourner les analyses sur ses propres données

Deux fichiers suffisent, à déposer ici :

- **`cohorte.csv`** — une ligne par patient. Colonnes indispensables : `id`, `outcome_M6`,
  `creat_admi`, `VRT`. Toutes les autres sont facultatives ; les scripts sautent proprement
  les lignes de tableau qu'ils ne peuvent pas calculer.
- **`radiomique.csv`** — une ligne par patient, une colonne `id` puis un paramètre par
  colonne. Les noms de colonnes doivent suivre la nomenclature PyRadiomics
  (`original_shape_…`, `original_firstorder_…`, `original_glcm_…`), car les scripts s'en
  servent pour reconnaître les familles de paramètres.

Dès que `cohorte.csv` est présent, il est utilisé **à la place** du fichier synthétique, sans
autre réglage. Le séparateur est la virgule, l'encodage UTF-8, et la virgule décimale est
acceptée aussi bien que le point.

Les valeurs censurées des comptes rendus (`<0,01`, `>750`) sont lues telles quelles : une
valeur inférieure à un seuil de détection est remplacée par la moitié du seuil, ce qui
préserve l'ordre — le seul aspect qui compte pour les tests de rang employés ici.

## Sécurité

`.gitignore` exclut tout ce dossier à l'exception des trois fichiers ci-dessus. Avant de
publier ou de mettre à jour le dépôt, une vérification suffit :

```bash
git status --short
```

Si un fichier de `donnees/` autre que ceux-là apparaît, **ne pas valider** : il contient
probablement des données de patients.
