# Recherche d incohérences dans un texte

Dans de nombreuses phrases, il est possible d'identifier une prémisse (condition logique sur laquelle l'argument s'appuie) et une hypothèse (conjecture réaliste s'appuyant sur la prémisse) par des méthodes d'apprentissage automatique.

Les méthodes d'inférence à partir de langage naturel (NLI) peuvent être utilisées pour déterminer automatiquement les couples prémisse/hypothèse de différentes phrases d'un texte (voir la compétition Kaggle en source). Il est ainsi possible d'analyser et d'annoter un texte ou une suite de messages, et donc d'identifier les nouvelles phrases qui seraient en contradictions (ou redondantes) avec les phrases précédentes.

Il vous est demandé de mettre en place une application d'analyse de texte, d'annotation du texte et d'identifications et d'élimination de contradictions/redondances. L'application pourra être utilisé pour aider lors de l'écriture de textes ou de messages sur les réseaux sociaux (type Twitter).

## Objectifs :

Le projet a pour but d'identifier et d'éliminer les contradictions/redondances dans un texte type paragraphe (paragraphe de roman par exemple) à l'aide d'un modèle NLI performant. Le logiciel prend un texte en entrée. Dans le texte créé en sortie, les contradictions/redondances seront annontées en rouge et on laisse le choix à l'utilisateur de les supprimer ou non (le texte en sortie reste inchangé de base). Il y a plusieurs fonctionnalités disponible sur deux interfaces bien distinctes. Je vous laisse vous référer à la section 7 de notre mémoire qui s'intitule "Présentation des outils développés" afin d'obtenir une description détaillée des différentes fonctionnalités disponible.

## Installation :     

Il est tout d'abord important de noter que l'ensemble de ces modules est importé au début de chaque fichier python correspondant aux interfaces :    
tkinter, tkinter.ttk, re, numpy, platform, nltk, nltk.tokenize, pandas et warnings    
Il sera alors éventuellement nécessaire d'installer python 3 et ces modules sur votre système d'exploitation, à l'aide de commandes du type *pip install \<nomDuModule\>*.

Autrement, les deux versions de notre interface sont utilisables en tapant ces commandes, dans le dossier source (*recherche-d-incoherences-dans-un-texte*) :    
- Version 1 : *python interface_tkinter\interface_14.py*
- Version 2 : *python interface_tkinter_2\interface_v2.py*

## Source :
- https://www.kaggle.com/competitions/contradictory-my-dear-watson/overview/description
- http://nlpprogress.com/english/natural_language_inference.html
- https://www.kaggle.com/code/anasofiauzsoy/tutorial-notebook/notebook
- https://paperswithcode.com/sota/natural-language-inference-on-multinli
