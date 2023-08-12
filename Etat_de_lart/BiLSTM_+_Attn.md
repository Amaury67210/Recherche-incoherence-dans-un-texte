

#  Multi-task BiLSTM + Attn (Wang et al., 2018)

#### Qu'est ce que GLUE ?

GLUE (General Language Understanding Evaluation) est une plate-forme de référence et d'analyse multi-tâches pour la compréhension du langage naturel.

Il faut obtenir ainsi le meilleur score possible dans les 11 tâches d'évaluation (voir lien ci-dessous) imposé par la plate-forme.

- https://gluebenchmark.com/leaderboard
- https://gluebenchmark.com/tasks

**Classement GLUE de Multi-task BiLSTM + Attn (Wang et al., 2018) :** 78

**Score obtenus :** 65.6

Ce modèle resemble très forte au modèle Multi-task BiLSTM + Attn en ce qui concerne l'architecture, les domaines où il obtient de bons résultats ainsi que le score obtenu sur GLUE. Le papier qui le présente est une introduction au référentiel GLUE où le modèle a été un des premiers modèles utilisé pour tester ce référentiel. Depuis la création de ce référentiel, une cinquantaine de modèle obtiennent de meilleure résultats que lui sur tout les plans donc il est devenu obsolète.

### GLUE

La capacité humaine à comprendre le langage est générale, flexible et robuste. En revanche, la plupart des modèles NLU au-dessus du niveau du mot sont conçus pour une tâche spécifique et se débattent avec des données hors du domaine. Si nous aspirons à développer des modèles dont la compréhension va au-delà de la détection de correspondances superficielles entre les entrées et les sorties, il est essentiel de développer un modèle plus unifié qui puisse apprendre à exécuter une gamme de tâches linguistiques différentes dans différents domaines.

Pour faciliter la recherche dans cette direction, nous présentons le référentiel General Language Understanding Evaluation (GLUE) : une collection de tâches NLU comprenant la réponse à des questions, l'analyse de sentiments et l'implication textuelle, et une plateforme en ligne associée pour l'évaluation, la comparaison et l'analyse de modèles. GLUE n'impose pas de contraintes sur l'architecture des modèles au-delà de la capacité à traiter des entrées d'une seule phrase ou d'une paire de phrases et à faire les prédictions correspondantes. Pour certaines tâches GLUE, les données d'entraînement sont abondantes, mais pour d'autres, elles sont limitées ou ne correspondent pas au genre de l'ensemble de test. GLUE favorise donc les modèles qui peuvent apprendre à représenter les connaissances linguistiques d'une manière qui facilite l'apprentissage efficace des échantillons et le transfert efficace des connaissances entre les tâches. Aucun des jeux de données de GLUE n'a été créé de toutes pièces pour le benchmark ; nous nous appuyons sur des jeux de données préexistants parce qu'ils ont été implicitement reconnus par la communauté NLP comme étant difficiles et intéressants. Quatre des jeux de données contiennent des données de test privées, qui seront utilisées pour s'assurer que le benchmark est utilisé de manière équitable.

### Architecture

L' architecture de base la plus simple est basée sur des codeurs de phrases en vecteurs, et met de côté la capacité de GLUE à évaluer des modèles avec des structures plus complexes. Le modèle utilise un BiLSTM (LSTM bidirectionnels) à deux couches, 1500D avec max pooling et 300D GloVe word embeddings. Pour les tâches à une seule phrase, nous encodons la phrase et passons le vecteur résultant à un classificateur. Pour les tâches de paires de phrases, nous encodons les phrases indépendamment pour produire des vecteurs u, v, et passons [u , v ; |u - v| ; u ∗ v] à un classificateur (détail de cette phase dans un autre état de l'art). Le classifieur est un MLP avec une couche cachée de 512D.

En gros dans ce papier, pour introduire et présenter GLUE, nous avons fait une série de test sur différents modèles où le modèle Multi-task BiLSTM + Attn était présent et nous avons obtenus les résultats ci-dessous :

Chaque colonne du tableau corrpespond à une tâche spécifique (que je peux décrire et ajouter au besoin ) :

![](/Users/amorce/Desktop/Capture d’écran 2022-10-03 à 15.06.22.png)

Nous constatons que l'entraînement multi-tâches donne de meilleurs résultats globaux que l'entraînement monotâche pour les modèles utilisant l'attention ou ELMo.

L'attention a généralement un effet global négligeable ou négatif dans l'entraînement à une seule tâche, mais elle est utile dans l'entraînement multi-tâches.

Nous constatons une amélioration constante dans l'utilisation des incorporations ELMo à la place des incorporations GloVe ou CoVe, en particulier pour les tâches à une seule phrase. 

![](/Users/amorce/Desktop/Capture d’écran 2022-10-03 à 15.06.46.png)

En conclusion, GLUE est une plateforme et une collection de ressources pour évaluer et analyser les systèmes de compréhension du langage naturel. On peut remarquer que, dans l'ensemble, les modèles formés conjointement à nos tâches obtiennent de meilleures performances que les performances combinées des modèles formés pour chaque tâche séparément.

On constate aussi l'utilité des mécanismes d'attention et des méthodes d'apprentissage par transfert telles que ELMo dans les systèmes NLU, qui se combinent pour surpasser les meilleurs modèles de représentation de phrases sur le benchmark GLUE, mais laissent encore une marge d'amélioration. 

En revanche, lorsqu'on évalue ces modèles sur notre jeu de données de diagnostic, nous constatons qu'ils échouent (souvent de manière spectaculaire) sur de nombreux phénomènes linguistiques, ce qui suggère des pistes de travail pour l'avenir. En résumé, la question de savoir comment concevoir des modèles NLU à usage général reste sans réponse, et nous pensons que GLUE peut fournir un terrain fertile pour relever ce défi.

