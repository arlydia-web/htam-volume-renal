# Guide de démarrage

Ce guide s'adresse à quelqu'un qui n'a jamais lancé de script Python depuis un terminal. Il va
de l'installation jusqu'à la mise en ligne du dépôt. Chaque bloc gris est une commande : on la
copie, on la colle dans le terminal, on appuie sur Entrée.

Sur macOS, le terminal s'ouvre avec **Cmd + Espace**, puis en tapant `Terminal`.

---

## 1. Installer conda (une seule fois)

Si `conda` répond déjà, passer à l'étape 2 :

```bash
conda --version
```

Sinon, installer **Miniforge** (la distribution conda libre, celle qui est déjà utilisée sur ce
poste) depuis <https://conda-forge.org/download/>, puis fermer et rouvrir le terminal.

---

## 2. Créer l'environnement (une seule fois)

Se placer dans le dossier du dépôt, puis créer l'environnement décrit par `environment.yml` :

```bash
cd ~/Desktop/htam-volume-renal
```

```bash
conda env create -f environment.yml
```

L'opération télécharge les bibliothèques et prend quelques minutes. Elle crée un environnement
nommé `htam` qui contient exactement les versions utilisées pour produire les résultats.

---

## 3. Activer l'environnement (à chaque nouvelle session de terminal)

```bash
conda activate htam
```

Le nom `(htam)` apparaît alors au début de la ligne de commande. C'est le signe que les
commandes `python` qui suivent utiliseront le bon environnement.

---

## 3 bis. Vérifier que tout est en place

```bash
python outils/verifier_installation.py
```

Le script contrôle la version de Python, chaque bibliothèque, la présence des fichiers de
données et leur bonne lecture. En cas de problème, il affiche la commande qui le corrige. Tant
qu'il ne se termine pas par « Tout est en place », inutile d'aller plus loin.

---

## 4. Lancer les analyses

Tout, sauf les deux analyses radiomiques qui sont longues :

```bash
python lancer_tout.py --rapide
```

Tout, sans exception — compter une heure :

```bash
python lancer_tout.py
```

Une analyse précise :

```bash
python scripts/03_modele_parcimonieux.py
```

Les résultats s'affichent dans le terminal **et** s'écrivent dans `resultats/` : un fichier
`.json` ou `.csv` par analyse, et les figures en PNG, PDF et SVG dans `resultats/figures/`.

### Ce qui s'affiche en tête de sortie

Sans données réelles, chaque script commence par un encadré :

```
╔════════════════════════════════════════════════════════════════════════════╗
║ DONNÉES SYNTHÉTIQUES : les chiffres ci-dessous ne sont PAS des résultats    ║
║ cliniques. Ils servent uniquement à vérifier que le code s'exécute.         ║
║ Fichier : cohorte_synthetique.csv                                          ║
╚════════════════════════════════════════════════════════════════════════════╝
```

C'est normal : c'est ce que verra un lecteur du dépôt. Les chiffres affichés sortent du jeu
simulé.

---

## 5. Travailler sur les vraies données (localement)

Les scripts lisent `donnees/cohorte.csv` et `donnees/radiomique.csv` dès que ces fichiers
existent, et se rabattent sinon sur le jeu synthétique. Pour les fabriquer depuis le classeur
maître :

```bash
python outils/exporter_depuis_master.py \
  --classeur "$HOME/Desktop/Thèse 24_06/Base de données 77 patients/Cohorte_HTAM_MASTER_2026-07-02.xlsx" \
  --radiomique "$HOME/Desktop/Thèse 24_06/Méthodes/reextraction_radiomique/features_reextracted.csv"
```

Puis relancer les analyses. Cette fois, chaque script termine par une ligne de contrôle :

```
✓ Contrôle : les 10 valeurs recalculées reproduisent celles du manuscrit.
```

(le nombre de valeurs contrôlées dépend du script)

Si un écart apparaît, il est affiché avec la valeur recalculée et la valeur publiée. Ce n'est
pas un arrondi : c'est un désaccord, et il demande examen.

> **Les deux fichiers produits contiennent des données de patients.** `.gitignore` les exclut
> du dépôt ; ils ne doivent jamais être publiés, ni envoyés par courriel, ni déposés sur un
> service de partage.

---

## 6. Publier le dépôt sur GitHub

### 6.0 Se présenter à git (une seule fois par ordinateur)

Git refuse d'enregistrer une version tant qu'il ne sait pas qui écrit. Deux commandes, à ne
faire qu'une fois :

```bash
git config --global user.name "Prénom Nom"
```

```bash
git config --global user.email "adresse@exemple.fr"
```

Cette adresse apparaîtra dans l'historique public du dépôt. GitHub en propose une de
substitution (`…@users.noreply.github.com`, dans *Settings → Emails*) pour ne pas exposer son
adresse personnelle.

### 6.1 Se placer DANS le dossier du dépôt

> ⚠️ **L'erreur à ne pas faire.** `git init` transforme en dépôt le dossier où l'on se trouve,
> et `git add` parcourt alors *tout* ce qu'il contient. Lancé depuis le dossier personnel, il
> se met à lire Photos, Musique, iCloud… et macOS ouvre une cascade de demandes d'autorisation.
> Ce n'est pas dangereux, mais ce n'est pas ce qu'on veut : il faut refuser ces demandes,
> supprimer le dépôt créé par erreur (`rm -rf ~/.git`), et recommencer au bon endroit.

Une commande à la fois, en attendant que chacune rende la main :

```bash
cd ~/Desktop/htam-volume-renal
```

Puis vérifier qu'on est bien là — la réponse doit se terminer par `htam-volume-renal` :

```bash
pwd
```

### 6.2 Vérifier qu'aucune donnée ne part avec

C'est la seule étape qu'il ne faut pas sauter.

```bash
git init
```

```bash
git add -A
```

```bash
git status --short
```

La liste affichée est **exactement** ce qui sera publié. Elle doit contenir le code, le
`README.md`, et de `donnees/` uniquement : `README.md`, `dictionnaire_variables.csv`,
`cohorte_synthetique.csv`, `radiomique_synthetique.csv`.

Si `donnees/cohorte.csv`, un `.xlsx` ou un fichier de `resultats/` apparaît :

```bash
git reset
```

…et vérifier `.gitignore` avant de recommencer. Ne jamais forcer un ajout avec `git add -f`.

### 6.3 Enregistrer une première version

```bash
git commit -m "Code d'analyse de la thèse — première version publiée"
```

### 6.4 Créer le dépôt en ligne

**Option A — depuis le site.** Sur <https://github.com/new>, créer un dépôt nommé
`htam-volume-renal`, **sans** cocher « Add a README file ». GitHub affiche ensuite deux
commandes `git remote add …` et `git push …` : les copier telles quelles.

**Option B — depuis le terminal**, si l'outil `gh` est installé (il ne l'est pas par défaut ;
`brew install gh` sur macOS) :

```bash
gh repo create htam-volume-renal --private --source=. --push
```

> **Conseil.** Créer le dépôt en **privé** d'abord (`--private`). Le passage en public se fait
> en un clic dans *Settings → General → Danger Zone → Change visibility*, le jour de la
> soutenance ou à la soumission de l'article. Un dépôt publié par erreur ne se dé-publie pas
> vraiment : il a pu être copié entre-temps.

### 6.5 Mettre à jour plus tard

```bash
cd ~/Desktop/htam-volume-renal
```

```bash
git add -A
```

```bash
git status --short
```

```bash
git commit -m "Ce qui a changé, en une phrase"
```

```bash
git push
```

Toujours regarder `git status --short` avant de valider. C'est la même vérification qu'au
premier envoi, et c'est là que se rattrapent les erreurs.

---

## 7. Autres hébergements possibles

| Où | Ce que ça apporte | Quand le choisir |
|---|---|---|
| **GitHub** | gratuit, connu de tous, historique des versions, lecture en ligne du code | par défaut |
| **Zenodo** | attribue un **DOI** citable et fige la version pour toujours ; se branche sur GitHub en deux clics | à faire au moment de la soutenance ou de la soumission — un DOI se cite dans un article, pas une adresse GitHub |
| **Software Heritage** | archivage patrimonial, moissonne GitHub automatiquement | rien à faire, mais bon à savoir |
| **GitLab / Codeberg** | équivalents à GitHub, hébergement européen pour Codeberg | si l'établissement le demande |
| **Hugging Face** | dépôts git gratuits, à côté des applications de démonstration | si une application interactive accompagne le code |

Le chemin habituel : GitHub pour le code vivant, puis Zenodo pour figer et citer la version de
la soutenance.

---

## 8. Que dire au jury ou à un relecteur

> Le code des analyses est public à l'adresse **<https://github.com/…/htam-volume-renal>**,
> sous licence MIT. Les données de patients ne sont pas diffusées ; un jeu synthétique de même
> structure est fourni, de sorte que tout le code s'exécute et se vérifie sans accès aux
> données. Les scripts contrôlent d'eux-mêmes qu'ils reproduisent les valeurs du manuscrit.

---

## En cas de problème

| Message | Ce qu'il faut faire |
|---|---|
| `conda: command not found` | conda n'est pas installé, ou le terminal n'a pas été rouvert après l'installation — étape 1 |
| `ModuleNotFoundError: No module named 'sklearn'` | l'environnement n'est pas activé : `conda activate htam` |
| `Ni donnees/cohorte.csv ni donnees/cohorte_synthetique.csv n'existent` | `python outils/generer_donnees_synthetiques.py` |
| `colonnes obligatoires absentes [...]` | le fichier CSV n'a pas les colonnes attendues — voir `donnees/dictionnaire_variables.csv` |
| macOS réclame l'accès aux Photos, à la Musique, au Bureau… | git est en train de parcourir le dossier personnel : un `git init` a été fait au mauvais endroit. Refuser les demandes, puis `rm -rf ~/.git`, puis reprendre à l'étape 6.1 |
| `fatal: not a git repository` | on n'est pas dans le dossier du dépôt : `cd ~/Desktop/htam-volume-renal` |
| Un script tourne depuis vingt minutes | c'est probablement `07`, qui est long par construction. Les analyses affichent leur avancement ligne à ligne. |
