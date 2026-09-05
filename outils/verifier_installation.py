#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vérifie que tout est en place avant de lancer les analyses.

Contrôle, dans l'ordre : la version de Python, la présence et la version de chaque
bibliothèque, l'existence des fichiers de données, et la bonne lecture de ces fichiers. Chaque
échec est accompagné de la commande qui le corrige.

Lancer :  python outils/verifier_installation.py
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

VERSIONS_DE_REFERENCE = {
    "numpy": "1.26.4", "scipy": "1.13.1", "pandas": "2.3.1",
    "sklearn": "1.6.1", "statsmodels": "0.14.6", "matplotlib": "3.9.4",
}
ECHECS = []


def controler(libelle, condition, remede=""):
    print(f"  {'✓' if condition else '✗'} {libelle}")
    if not condition:
        ECHECS.append((libelle, remede))
    return condition


def main():
    print("\nVérification de l'installation\n" + "─" * 60)

    print("\nPython")
    version = sys.version_info
    controler(f"version {version.major}.{version.minor}.{version.micro} "
              f"(3.9 ou plus récent)", version >= (3, 9),
              "installer l'environnement : conda env create -f environment.yml")
    print(f"    interpréteur : {sys.executable}")

    print("\nBibliothèques")
    for module, attendue in VERSIONS_DE_REFERENCE.items():
        try:
            importe = __import__(module)
            obtenue = getattr(importe, "__version__", "?")
            ecart = "" if obtenue == attendue else f"  (référence {attendue})"
            controler(f"{module} {obtenue}{ecart}", True)
        except ImportError:
            controler(f"{module} — absent", False,
                      "activer l'environnement : conda activate htam")

    print("\nDonnées")
    from htam import config
    reel = config.DOSSIER_DONNEES / config.FICHIER_COHORTE
    synthetique = config.DOSSIER_DONNEES / config.FICHIER_COHORTE_SYNTHETIQUE
    if reel.exists():
        print(f"  ✓ données réelles présentes ({reel.name}) — elles seront utilisées")
    elif synthetique.exists():
        print(f"  ✓ jeu synthétique présent ({synthetique.name}) — les analyses tourneront "
              f"dessus")
    else:
        controler("aucun fichier de données", False,
                  "python outils/generer_donnees_synthetiques.py")

    if reel.exists() or synthetique.exists():
        try:
            from htam import donnees
            cohorte, simule = donnees.charger_cohorte(silencieux=True)
            controler(f"lecture de la cohorte : {len(cohorte)} patients, "
                      f"{int(cohorte[config.CRITERE].sum())} non-récupérations"
                      + (" (synthétiques)" if simule else ""), True)
            radiomique, _ = donnees.charger_radiomique()
            controler(f"lecture de la radiomique : {len(radiomique)} patients × "
                      f"{radiomique.shape[1]} paramètres", True)
        except Exception as erreur:
            controler(f"lecture des données — {type(erreur).__name__} : {erreur}", False,
                      "voir donnees/README.md pour le format attendu")

    print("\n" + "─" * 60)
    if ECHECS:
        print(f"{len(ECHECS)} point(s) à corriger :\n")
        for libelle, remede in ECHECS:
            print(f"  · {libelle}")
            if remede:
                print(f"    → {remede}")
        return 1
    print("Tout est en place. Lancer :  python lancer_tout.py --rapide")
    return 0


if __name__ == "__main__":
    sys.exit(main())
