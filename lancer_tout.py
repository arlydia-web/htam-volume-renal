#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exécute toutes les analyses dans l'ordre, et s'arrête à la première qui échoue.

    python lancer_tout.py              toutes les analyses
    python lancer_tout.py --rapide     saute les trois plus longues (06, 07 et 09)
    python lancer_tout.py 03 04        seulement les analyses demandées

Chaque script est lancé dans un processus séparé : une analyse qui échoue n'emporte pas les
autres, et la sortie reste identique à celle d'un lancement à la main.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent
DOSSIER = RACINE / "scripts"

ANALYSES = [
    ("01", "Description de la cohorte", "quelques secondes"),
    ("02", "Analyse univariée", "quelques secondes"),
    ("03", "Modèle parcimonieux", "environ une minute"),
    ("04", "Reclassification", "moins d'une minute"),
    ("05", "Règle à trois zones", "quelques secondes"),
    ("06", "Modèles radiomiques M1/M2/M3", "une vingtaine de minutes"),
    ("07", "Robustesse de la sélection", "vingt à soixante minutes"),
    ("08", "Figures", "quelques secondes"),
    ("09", "Redondance et stabilité des paramètres radiomiques", "dix à vingt minutes"),
]
LONGUES = {"06", "07", "09"}


def script(numero):
    trouves = sorted(DOSSIER.glob(f"{numero}_*.py"))
    if not trouves:
        raise FileNotFoundError(f"aucun script ne commence par « {numero}_ » dans {DOSSIER}")
    return trouves[0]


def main():
    analyseur = argparse.ArgumentParser(description=__doc__,
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    analyseur.add_argument("numeros", nargs="*", help="numéros d'analyse (01 à 09)")
    analyseur.add_argument("--rapide", action="store_true",
                           help="saute les analyses radiomiques, les plus longues")
    arguments = analyseur.parse_args()

    if arguments.numeros:
        choisies = [a for a in ANALYSES if a[0] in arguments.numeros]
    elif arguments.rapide:
        choisies = [a for a in ANALYSES if a[0] not in LONGUES]
    else:
        choisies = ANALYSES

    print(f"\n{len(choisies)} analyse(s) à exécuter, avec {sys.executable}\n")
    depart_total = time.time()
    for numero, titre, duree in choisies:
        chemin = script(numero)
        print("━" * 78)
        print(f"▶ {numero} — {titre}   ({duree})")
        print("━" * 78, flush=True)
        depart = time.time()
        resultat = subprocess.run([sys.executable, str(chemin)], cwd=RACINE)
        if resultat.returncode != 0:
            print(f"\n⛔ {chemin.name} s'est arrêté sur une erreur (code "
                  f"{resultat.returncode}). Rien n'est lancé ensuite.")
            return resultat.returncode
        print(f"\n✓ {numero} terminé en {time.time() - depart:.0f} s\n", flush=True)

    print("━" * 78)
    print(f"Toutes les analyses ont abouti — {time.time() - depart_total:.0f} s au total.")
    print(f"Résultats dans : {RACINE / 'resultats'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
