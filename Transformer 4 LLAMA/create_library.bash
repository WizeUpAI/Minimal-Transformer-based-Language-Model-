#!/bin/bash

# 🔧 Script pour builder et installer la bibliothèque llama_from_scratch

set -e  # Quitte en cas d'erreur

echo "🔹 Étape 1 : Installer les outils nécessaires"
pip install --upgrade pip setuptools wheel

echo "🔹 Étape 2 : Nettoyer les anciens fichiers de build"
rm -rf build/ dist/ llama_from_scratch.egg-info/

echo "🔹 Étape 3 : Construire le package (whl + tar.gz)"
python setup.py sdist bdist_wheel

echo "🔹 Étape 4 : Installer la bibliothèque localement"
pip install dist/llama_from_scratch-0.1-py3-none-any.whl

echo "✅ Terminé ! Vous pouvez maintenant utiliser : from llama.model import LLaMAModel"
