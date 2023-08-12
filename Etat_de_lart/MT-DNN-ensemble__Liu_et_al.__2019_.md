# MT-DNN-ensemble (Liu et al., 2019)

#### <u>Qu'est ce que GLUE ?</u>

GLUE (General Language Understanding Evaluation) est une plate-forme de référence et d'analyse multi-tâches pour la compréhension du langage naturel.

Il faut obtenir ainsi le meilleur score possible dans les 11 tâches d'évaluation (voir lien ci-dessous) imposé par la plate-forme.

- https://gluebenchmark.com/leaderboard
- https://gluebenchmark.com/tasks

**Classement GLUE de MT-DNN-ensemble (Liu et al., 2019) :** 20

**Score obtenus :** 87.6

L'implémentation de MT-DNN est basée sur les implémentations PyTorch de MT-DNN3 et BERT4 .De plus, MT-DNN utilise Adamax comme optimiseur.

#### <u>Architecture du modèle MT-DNN</u>

Les chercheurs de Microsoft ont publié les détails techniques d'un système d'IA qui combine l'apprentissage multi-tâches ainsi que la pré-formation de modèles linguistiques . Le nouveau réseau de neurones profonds multitâches (MT-DNN) est un modèle de traitement du langage naturel (NLP) qui surpasse Google BERT dans neuf des onze tâches NLP de référence.

MT-DNN s'appuie sur un modèle proposé par Microsoft en 2015 et intègre l'architecture réseau de BERT, un modèle de langage de transformateur bidirectionnel pré-entraîné proposé par Google l'année dernière.

![Architecture d'un Modèle MT-DNN](/Users/amorce/Desktop/Capture d’écran 2022-09-27 à 14.01.32.png)

Comme le montre la figure ci-dessus, les couches de bas niveau du réseau (c'est-à-dire les couches d'encodage de texte) sont partagées entre toutes les tâches, tandis que les couches supérieures sont spécifiques aux tâches, combinant différents types de tâches NLU (Natural language understanding). Comme le modèle BERT, MT-DNN est formé en deux phases : pré-formation et mise au point. Mais contrairement à BERT, MT-DNN ajoute l'apprentissage multitâche (MTL) dans les phases de réglage fin avec plusieurs couches spécifiques aux tâches dans son architecture de modèle.

L'entrée X, qui est une séquence de mots (soit une phrase soit un ensemble de phrases regroupées) est d'abord représentée comme une séquence de vecteurs d'intégration, un pour chaque mot, dans *l1*. Ensuite, le codeur transformateur capture les informations contextuelles pour chaque mot et génère les vecteurs d'intégration contextuelle partagés dans l2. C'est la représentation sémantique partagée qui est entraînée par nos objectifs multitâches.

Enfin, pour chaque tâche, des couches supplémentaires spécifiques à la tâche génèrent des représentations spécifiques à la tâche, suivies des opérations nécessaires à la classification, à la notation de similarité ou au classement de pertinence.

<u>Lexicon Encoder</u> : L'entrée X = {x1, ..., xm} est une séquence de jetons de longueur m. Si X est emballé par un ensemble de phrases (X1, X2), nous séparons ces phrases avec des jetons spéciaux [SEP]. L'encodeur de lexique mappe X en une séquence de vecteurs d'intégration d'entrée, un pour chaque jeton, construit en additionnant les intégrations de mot, de segment et de position correspondantes.

<u>Transformer Encoder</u> : Le modèle utilise un "multilayer bidirectional Transformer encoder" pour mapper les vecteurs de représentation d'entrée (l1) en une séquence de vecteurs d'intégration contextuels C. Il s'agit de la représentation partagée entre différentes tâches.

<u>Couches de sortie spécifiques à la tâche</u> : Le modèle incorpore des tâches arbitraires en langage naturel, chacune avec ses couches de sortie spécifiques à la tâche. Par exemple, nous implémentons les couches de sortie comme un décodeur neuronal pour la génération de texte, un classement neuronal pour le classement par pertinence, une régression logistique pour la classification du texte, etc.

#### Processus de distillation des connaissances pour l'apprentissage multi-tâches

![](/Users/amorce/Desktop/Capture d’écran 2022-09-27 à 14.58.35.png)

Un ensemble de tâches où il y a des données de formation étiquetées spécifiques à la tâche est sélectionné. Ensuite, pour chaque tâche, un ensemble de réseaux de neurones différents (enseignant) est formé. L'enseignant est utilisé pour générer pour chaque échantillon de formation spécifique à une tâche un ensemble de cibles souples. Compte tenu des cibles souples des ensembles de données de formation sur plusieurs tâches, un seul MT-DNN (étudiant) est formé à l'aide de l'apprentissage multitâche et de la rétropropagation comme décrit dans l'algorithme, sauf que si la tâche *t* a un enseignant, la perte spécifique à la tâche dans la ligne 3 (voir algorithme ci-dessous) est la moyenne de deux fonctions objectives, l'une pour les cibles correctes et l'autre pour les cibles souples assignées par l'enseignant.

Une fois que MT-DNN est formé via MTL, il peut être affiné (ou adapté) à l'aide de données de formation étiquetées spécifiques à la tâche pour effectuer une prédiction sur n'importe quelle tâche individuelle, qui peut être une tâche utilisée dans l'étape MTL ou une nouvelle tâche qui est liés à ceux utilisés dans MTL. Liu et al. (2019) ont montré que les couches partagées de MT-DNN produisent des représentations textuelles plus universelles que celles de BERT. En conséquence, MT-DNN permet un réglage fin ou une adaptation avec beaucoup moins d'étiquettes spécifiques aux tâches.

Lorsque les cibles correctes sont connues, les performances du modèle peuvent être considérablement améliorées en entraînant le modèle distillé sur une combinaison de cibles souples et dures. Nous le faisons en définissant une fonction de perte pour chaque tâche qui prend une moyenne pondérée entre la perte d'entropie croisée avec les cibles correctes.

Ce modèle a pour but de montrer la distillation de connaissances étendu à MTL pour former un MT-DNN pour la compréhension du langage naturel. Nous avons montré que la distillation fonctionne très bien pour transférer les connaissances d'un ensemble de modèles (enseignants) vers un seul MT-DNN distillé (étudiant). 

#### Algorithme

![](/Users/amorce/Desktop/Capture d’écran 2022-09-28 à 15.02.22.png)

#### Résultat

<u>MT-DNNKD</u> : Il s'agit du modèle MT-DNN entraîné en utilisant la distillation des connaissances. MT-DNNKD utilise la même architecture de modèle que celle de MT-DNN. Mais il est entraîné avec l'aide de quatre ensembles spécifiques à la tâche (enseignants). Le MT-DNNKD est optimisé pour les objectifs multitâches qui sont basés sur les cibles correctes dures, ainsi que sur les cibles douces produites par les enseignants si elles sont disponibles.

![](/Users/amorce/Desktop/Capture d’écran 2022-09-27 à 15.00.28.png)

Le tableau montre que MT-DNNKD surpasse significativement MT-DNN non seulement dans le score global mais aussi sur 7 des 9 tâches GLUE, y compris les tâches sans enseignant. Puisque MT-DNNKD et MT-DNN utilisent la même architecture de réseau, et sont entraînés avec la même initialisation et sur les mêmes jeux de données, l'amélioration de MT-DNNKD est uniquement attribuée à l'utilisation de la distillation des connaissances dans MTL.

Nous montrons que le MT-DNN distillé conserve presque toutes les améliorations obtenues par les modèles d'ensemble, tout en gardant la même taille de modèle comme le modèle MT-DNN de base.

### Liens

- https://arxiv.org/pdf/1904.09482.pdf
- https://medium.com/syncedreview/microsofts-new-mt-dnn-outperforms-google-bert-b5fa15b1a03e
- https://medium.com/syncedreview/improved-microsoft-mt-dnn-tops-glue-rankings-a22575b63f24
