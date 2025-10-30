# **Projet GNN - Détection d'Anomalies sur le Graphe des Aéroports (Amine ITJI & Youssef BOUAMAMA)**



## 1. Introduction et Question de Recherche

### 1.1. Contexte du Projet

Ce projet vise à appliquer les techniques de Graphes de Neurones (GNN) sur un jeu de données représentant le réseau mondial des aéroports. L'objectif est de formuler une question de recherche pertinente et d'y répondre en utilisant une méthodologie de Machine Learning rigoureuse, incluant la comparaison de modèles et une analyse d'ablation, pour aboutir à un rapport scientifique de 6 pages maximum.

Le jeu de données fourni est un graphe où les nœuds sont des aéroports et les arêtes représentent les liaisons aériennes. Les nœuds possèdent des caractéristiques (features) : latitude, longitude, et population de la ville associée, ainsi qu'une étiquette sémantique, le pays.

### 1.2. Choix de la Question de Recherche

Les consignes du projet proposent plusieurs "Questions possibles" : prédiction de liens, classification, ou détection d'anomalies.

Une information cruciale est explicitement donnée : **"Certaines des informations de pays, et beaucoup d'information de population sont erronnées"**.

* Choisir la **Classification** comme objectif principal (prédire le pays) serait biaisé, car nous nous entraînerions sur des étiquettes potentiellement fausses.
* Choisir la **Prédiction de Liens** est une tâche structurelle intéressante, mais elle ignore les riches caractéristiques nodales (population, pays) qui sont au cœur de la problématique des "données erronnées".

Par conséquent, la question de recherche la plus pertinente et la plus alignée avec la description du jeu de données est la **Détection d'Anomalies sur les Nœuds**.

**Question de Recherche (formulée) :**
> "Est-il possible de développer un modèle GNN qui, en apprenant les relations structurelles (liaisons), géographiques (coordonnées) et démographiques (population) normales du réseau d'aéroports, peut identifier les nœuds (aéroports) contenant des informations de population ou de pays erronées ?"

L'hypothèse sous-jacente est que les anomalies se manifesteront comme des *outliers* : des nœuds que le modèle aura du mal à prédire correctement car leurs caractéristiques contredisent ce que le modèle a appris de leurs voisins "normaux".

---

## 2. Approche Méthodologique et Démarche

Notre démarche s'articule autour d'une approche de détection d'anomalies non-supervisée (ou "auto-supervisée"). Nous construisons un modèle GNN pour une tâche *prétexte* (prédire les propriétés des nœuds), puis nous utilisons l'erreur de ce modèle comme score d'anomalie.

### 2.1. Pré-traitement et Caractéristiques (Features)

Les données initiales proviennent du fichier `airportsAndCoordAndPop.graphml.xml`. Chaque nœud représente un aéroport et contient ses coordonnées géographiques, sa population associée et son pays.

#### Statistiques générales

| Élément | Valeur |
| :--- | :--- |
| Nombre de nœuds (aéroports) | **3 363** |
| Nombre d’arêtes (connexions) | **13 547** |
| Nombre de classes (pays) | **212** |

L'important nombre de pays (212) montre la forte diversité géographique, mais **impliquera une classification complexe**, car certaines classes (pays) sont très peu représentées.

#### Top 10 des pays les plus représentés

| Pays | Nombre d’aéroports |
| :--- | :--- |
| USA | 650 |
| CANADA | 243 |
| AUSTRALIA | 207 |
| BRAZIL | 105 |
| CHINA | 93 |
| PAPUA\_NEW\_GUINEA | 92 |
| RUSSIA | 70 |
| JAPAN | 69 |
| MEXICO | 59 |
| ARGENTINA | 57 |

On remarque un **fort déséquilibre de classes** : les 10 premiers pays concentrent à eux seuls près de la moitié des 3 363 aéroports. Ce déséquilibre rendra la prédiction du pays plus difficile.

#### Préparation des données

Les données brutes sont chargées à l'aide de la classe `AirportDataLoader`. Pour chaque nœud, nous extrayons et utilisons quatre caractéristiques fondamentales :
1.  **Latitude** (normalisée)
2.  **Longitude** (normalisée)
3.  **Population Logarithmique** : La population est une caractéristique clé, mais sa distribution est fortement asymétrique. Nous la transformons donc avec `np.log1p` pour la rendre plus gaussienne, ce qui stabilise l'entraînement.
4.  **Degré du Nœud** : Le nombre de connexions d'un aéroport, une mesure de centralité fondamentale.

Les caractéristiques de latitude et longitude sont normalisées (Standard Scaler) pour être centrées autour de 0. Les données sont ensuite divisées en ensembles d'entraînement (60%), de validation (20%) et de test (20%).

### 2.2. Une Tâche d'Apprentissage Multi-Objectifs

Pour que notre modèle apprenne une représentation riche (`embedding`) d'un nœud, nous lui demandons de prédire *simultanément* deux propriétés :
1.  **Tâche de Régression (Population)** : Prédire la population (log-transformée). La perte est la **MSE (Mean Squared Error)**.
2.  **Tâche de Classification (Pays)** : Prédire le pays de l'aéroport. La perte est la **Cross-Entropy Loss**.

La fonction de perte (loss) totale pour nos modèles standards est une somme pondérée, contrôlée par un hyperparamètre $\alpha \in [0, 1]$ :

$$
\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{MSE}(\text{pop})} + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}(\text{country})}
$$

Afin d’enrichir encore la représentation, nous avons étendu cette approche à une **troisième tâche** pour notre modèle `GAT + Anomaly-Guided` : la **détection d’anomalies de liens**.

L’idée est que certaines connexions peuvent être *incohérentes* (ex: une petite ville connectée directement à un grand hub étranger). Pour cela, nous introduisons une composante de perte $\mathcal{L}_{\text{link}}$, donnant une loss totale à trois composantes :

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{pop}} + \beta \cdot \mathcal{L}_{\text{country}} + \gamma \cdot \mathcal{L}_{\text{link}}$$

Cette extension permet au réseau d’apprendre non seulement à prédire les attributs des nœuds, mais aussi à identifier les connexions improbables, renforçant sa robustesse.

### 2.3. Métrique d'Évaluation (Score d'Anomalie)

Après l'entraînement, la classe `AnomalyEvaluator` calcule un score d'anomalie pour *chaque* nœud.

1.  **Erreur de Population ($e_{pop}$)** : $\lvert \log(\hat{y}_{pop}) - \log(y_{pop}) \rvert$
2.  **Score de Pays ($s_{country}$)** : $1 - P(y_{\text{vrai}} | z)$ (l'incertitude du modèle). Un score proche de 1 signifie que le modèle est très peu confiant.

Notre **`combined_score`** final pour les modèles standards est fixé avec $\alpha=0.6$ :

$$
\text{Score}_{\text{Std}} = 0.6 \cdot e_{pop} + 0.4 \cdot s_{country}
$$

Pour le modèle **GAT + Anomaly-Guided**, une **troisième composante** est ajoutée pour évaluer la cohérence des **liens** du nœud :

* **Score de Lien ($s_{link}$)** : évalue à quel point les connexions d'un nœud sont jugées anormales par le détecteur de liens.

Le score global devient alors (avec les poids $\alpha, \beta, \gamma$ fixés à $0.5, 0.3, 0.2$ dans notre code) :

$$
\text{Score}_{\text{AG}} = 0.5 \cdot e_{pop} + 0.3 \cdot s_{country} + 0.2 \cdot s_{link}
$$

Pour comparer les modèles, nous utilisons le **95ème percentile (Q95)** du score d'anomalie sur l'ensemble de test. Un Q95 plus **bas** est meilleur.

---

## 3. Recherche d'un Modèle Adapté

Conformément aux "Attendus" du projet, nous devons comparer plusieurs modèles candidats. Notre démarche a consisté à comparer des baselines GCN simples à une architecture GAT plus complexe, puis à optimiser cette dernière par un Grid Search systématique.

### 3.1. Modèles Candidats

Nous avons implémenté quatre architectures :

1.  **`BaselineGCN` (Référence)** : Un GCN simple à 2 couches. Il sert de référence minimale.
2.  **`AnomalyDetectorGCN` (Référence)** : Un GCN plus profond (3 couches) avec des décodeurs MLP plus complexes (incluant `ReLU` et `Dropout`).
3.  **`ImprovedGAT` (Notre Proposition)** : Nous avons émis l'hypothèse qu'un mécanisme d'**attention** (GATConv) serait supérieur. L'attention permet au modèle de pondérer l'importance des voisins, et pourrait ainsi apprendre à *ignorer* l'influence d'un voisin lui-même anormal.
4.  **`GAT + Anomaly-Guided` (Amélioration)** : Ce modèle étend le `ImprovedGAT` en intégrant un **apprentissage guidé par la structure des liens** du graphe (voir Section 2.2). Il apprend non seulement les propriétés des nœuds, mais aussi à identifier les *connexions* anormales.

Grâce à notre Grid Search complet, nous avons pu comparer rigoureusement ces quatre architectures.

### 3.2. Grid Search (Recherche Systématique)

Pour justifier notre "proposition finale", nous avons implémenté un Grid Search complet pour explorer l'espace des hyperparamètres. L'objectif était de tester :
* **`hidden_channels` [16, 32, 64]** : La dimension de l'embedding.
* **`num_layers` [2, 3, 4]** : La profondeur du modèle (voisinage k-hops).
* **`num_heads` [1, 2, 4]** : Le nombre de têtes d'attention.

Nous avons mené deux campagnes de tests parallèles :
1.  **Grid Search Standard (135 exécutions)** : Sur le `ImprovedGAT`, en testant également le poids de la loss `train_alpha` [0.1, 0.3, 0.5, 0.7, 0.9].
2.  **Grid Search AG (27 exécutions)** : Sur le `GAT + Anomaly-Guided`, en gardant des poids de loss fixes ($\alpha=0.5, \beta=0.3, \gamma=0.2$) pour se concentrer sur l'architecture.

Cette recherche de **(135 + 27) = 162 exécutions** au total nous a fourni une analyse robuste pour déterminer le meilleur modèle global.

---

## 4. Résultats et Interprétations

### 4.1. Validation de l'Entraînement

Avant d'analyser les performances, il est crucial de valider que les modèles s'entraînent correctement. L'image ci-dessous montre les courbes d'apprentissage de notre meilleur modèle global, identifié par le Grid Search : **`GAT_AG_h64_l2_head4`**.

![Courbes d'entraînement du meilleur modèle GAT](results/run_20251029_224928_GridSearch_Full/training_GAT_AG_h64_l2_head4_BEST_OVERALL.png)

On observe une convergence stable : la `train_loss` et la `val_loss` (ici, la MSE de validation) diminuent et se stabilisent. L'Accuracy de validation (`val_acc`) augmente conjointement. Le mécanisme d'Early Stopping s'est déclenché lorsque la `val_loss` n'a plus diminué, empêchant le sur-apprentissage.

### 4.2. Attendu 1 : Tableau Comparatif et Analyse

Le Grid Search a identifié que l'architecture `GAT + Anomaly-Guided` surpasse toutes les autres configurations.

![Comparaison des modèles de base et du meilleur GAT](results/run_20251029_224928_GridSearch_Full/comparison_01_main_models.png)

**Interprétations (Attendu 1) :**

1.  **GAT-AG > GAT > GCN** : Le graphique de comparaison est sans appel. Notre proposition `GAT + Anomaly-Guided` domine l'ensemble des tests. Le meilleur modèle est **`GAT_AG_h64_l2_head4`**, qui obtient un score Q95 de **1.6496**.

2.  **L'utilité de la supervision des liens** : Le meilleur GAT *standard* (`GAT_h64_l3_head2_a0.5`) obtient un score de **1.8385**. Le fait de passer à une architecture `Anomaly-Guided` (1.6496) apporte une amélioration significative, prouvant que l'apprentissage de la structure des liens améliore la détection des anomalies de nœuds.

3.  **L'échec des Baselines** : Les modèles GCN de référence sont loin derrière, avec des scores Q95 supérieurs à 2.0, validant notre choix d'abandonner GCN au profit de GAT.

**Tableau 3 : Résultats Détaillés du Grid Search (Top 5 + Baselines)**
*(Basé sur le score Q95 `combined_score` sur l'ensemble de test. Plus bas = Meilleur)*

| Modèle | Alpha (Train) | Score (Type) | **Q95 (Test)** | Q99 (Test) |
| :--- | :---: | :--- | :---: | :---: |
| **`GAT_AG_h64_l2_head4`** | **0.5** | **combined_score** | **1.6496** | **2.2778** |
| `GAT_AG_h64_l3_head4` | 0.5 | combined_score | 1.6548 | 2.1583 |
| `GAT_AG_h64_l2_head2` | 0.5 | combined_score | 1.6587 | 2.3023 |
| `GAT_AG_h32_l2_head4` | 0.5 | combined_score | 1.6852 | 2.2727 |
| `GAT_AG_h64_l3_head2` | 0.5 | combined_score | 1.7046 | 2.2238 |
| ... | ... | ... | ... | ... |
| `GAT_h64_l3_head2_a0.5` | 0.5 | (Meilleur GAT Std) | 1.8385 | 2.5020 |
| ... | ... | ... | ... | ... |
| `AnomalyDetector GCN (Ref)`| 0.6 | combined_score | 2.0034 | 2.6334 |
| `Baseline GCN (Ref)` | 0.6 | combined_score | 2.0688 | 2.7402 |

*(Données extraites de `results/run_20251029_224928_GridSearch_Full/comparison_full_results.csv`)*

### 4.3. Attendu 2 : Étude d'Ablation

Conformément aux consignes, nous avons effectué une étude d'ablation sur le meilleur *modèle GAT standard* identifié (`GAT_h64_l3_head2_a0.5`). L'élément "customisé" que nous testons est notre fonction de perte combinée (Pop vs Country).

Nous comparons la performance de ce modèle (entraîné avec $\alpha=0.5$) à deux versions d'ablation entraînées sur la même architecture :
1.  **Pop-Only** : Entraîné uniquement sur la population ($\alpha=1.0$).
2.  **Country-Only** : Entraîné uniquement sur le pays ($\alpha=0.0$).

![Graphique de l'étude d'Ablation](results/run_20251029_224928_GridSearch_Full/comparison_02_ablation_study.png)

**Interprétations (Attendu 2) :**

Cette étude d'ablation démontre de manière spectaculaire l'utilité de notre approche multi-tâches :

1.  **`Country-Only` ($\alpha=0.0$) est un échec complet.** Le score Q95 explose à **8.4116**. En analysant son "Top 10" (voir log), la raison est évidente : `1. Shanghai, 2. Beijing, 3. Shenzhen...` Ce modèle n'ayant *jamais* appris la notion de population, il signale à tort les plus grandes villes du monde comme des anomalies. Cela prouve que **la prédiction de population est indispensable**.

2.  **L'Ablation `Pop-Only` ($\alpha=1.0$) est moins performante.** Le modèle entraîné uniquement sur la population obtient un score de **1.9446**, ce qui est nettement moins bon que notre modèle combiné à $\alpha=0.5$ (1.8385).

3.  **Conclusion de l'Ablation** : **Toutes les composantes de notre modèle sont utiles**. Le meilleur modèle n'est ni $\alpha=0$ ni $\alpha=1$, mais un équilibre.

### 4.4. Validation Manuelle (Analyse Qualitative)

La validation finale, comme suggéré par les consignes, est l'observation manuelle des anomalies détectées.

![Distribution des Scores d'Anomalie](results/run_20251029_224928_GridSearch_Full/anomaly_dist.png)

Le graphique de distribution des scores de notre meilleur modèle montre que la grande majorité des nœuds ont un score d'anomalie très faible (proche de 0). Cependant, une longue traîne (long tail) est visible, représentant les nœuds anormaux que le modèle a identifiés.

![Visualisation t-SNE des Embeddings](results/run_20251029_224928_GridSearch_Full/tsne.png)

La visualisation t-SNE est très révélatrice :
* **Graphique de Gauche (par Pays)** : Le modèle a appris des embeddings sémantiquement pertinents. On voit des clusters clairs se former (ex: les nœuds "USA" en violet, "JAPAN" en rouge, "AUSTRALIA" en marron), prouvant que le GNN a capturé la structure géo-politique du réseau.
* **Graphique de Droite (par Score)** : En colorant ces mêmes points par leur score d'anomalie, on voit que les anomalies (en rouge vif) ne sont pas des clusters entiers, mais des **points individuels** qui se détachent de leur propre cluster de pays.

---
#### **Anomalies Emblématiques (Détectées par les modèles standards)**

L'analyse des "Top 10" de nos modèles GAT standards a systématiquement révélé des erreurs de données factuelles :

**Anomalie 1 : Sydney, CANADA (Nœud 2488)**
* **Constat** : Ce nœud est l'anomalie N°1 ou N°2 dans la quasi-totalité de nos runs GAT standards (ex: score 3.567 pour le Baseline GCN).
* **Analyse des données** : Le fichier attribue à "Sydney, CANADA" une population de **5 231 147**. En recherchant le nœud `54` ("Sydney", "AUSTRALIA"), on constate qu'il a la *exactement la même* population. La population réelle de Sydney, Canada, est d'environ 30 000 habitants.
* **Conclusion** : Il s'agit d'une **erreur de copier-coller évidente**. Notre modèle a parfaitement réussi à la détecter, en signalant qu'un aéroport avec une faible connectivité au Canada ne pouvait pas avoir une population de 5.2 millions.

**Anomalie 2 : Xi An, CHINA (Nœud 285)**
* **Constat** : Également un "Top 10" récurrent (ex: score 3.333 pour l'AnomalyDetector GCN).
* **Analyse des données** : Le fichier de données attribue au nœud `285` ("Xi An", "CHINA") une population de **10 000**.
* **Conclusion** : Il s'agit d'une autre **erreur factuelle flagrante**. Xi'an est une métropole de plus de 10 millions d'habitants. Le modèle a appris qu'un nœud avec un tel degré et de telles connexions ne pouvait pas avoir la population d'un petit village.

---
#### **Anomalies du Meilleur Modèle (GAT + Anomaly-Guided)**

Notre **meilleur modèle global** (`GAT_AG_h64_l2_head4`) identifie un ensemble d'anomalies différent, souvent plus subtil, car il pénalise aussi les nœuds ayant des *liens suspects* (voir Section 4.5). Voici son "Top 3" (issu des logs `BaseRun` qui détaillent les anomalies de ce type de modèle, car le log du GridSearch est abrégé) :

**Anomalie 1 : Taipei, TAIWAN**
* **Population observée :** 7 871 900
* **Population prédite :** 110 104
* **Pays prédit :** THAILAND ❌
* **Score Liens :** 0.5266 (Élevé)
* **Interprétation :** C'est l'anomalie parfaite. Le modèle échoue sur *tous les plans* : il sous-estime massivement la population, se trompe de pays, et signale que ses connexions sont suspectes.

**Anomalie 2 : Pusan, SOUTH\_KOREA**
* **Population observée :** 10 000
* **Population prédite :** 649 863
* **Pays prédit :** CHINA ❌
* **Score Liens :** 0.4732 (Élevé)
* **Interprétation :** Similaire à Xi An, c'est une **erreur de population sous-estimée**. Mais le GAT-AG ajoute deux dimensions : il se trompe de pays (probablement à cause de la forte connectivité avec la Chine) et signale des liens anormaux.

**Anomalie 3 : Kuwait, KUWAIT**
* **Population observée :** 10 000
* **Population prédite :** 616 492
* **Pays prédit :** INDIA ❌
* **Score Liens :** 0.5677 (Très élevé)
* **Interprétation :** Encore une **population sous-estimée**. Le modèle prédit "INDIA" et signale des liens très anormaux, ce qui est logique : l'aéroport de Koweït sert de hub majeur pour les vols vers l'Inde, une connexion que le modèle a jugée anormale pour un "village" de 10 000 habitants.

*(Les autres anomalies comme Frankfurt, Los Angeles (Chili) et Melbourne (USA) sont également systématiquement détectées, confirmant des erreurs de données flagrantes.)*

### 4.5 Analyse des Liens Anormaux Détectés

Après l’analyse des **10 liens présentant les scores d’anomalie les plus élevés**, notre modèle **GAT + Anomaly-Guided** a mis en évidence plusieurs connexions **hautement improbables** du point de vue géographique et structurel.

**Top 10 liens les plus suspects (scores d’anomalie les plus élevés)**

| Rang | Source (Pays) | Population source | Destination (Pays) | Population destination | Score | Distance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | Barrow *(Australie)* | 10 000 | Cape Lisburne *(USA)* | 10 000 | **0.9970** | 4.68 |
| #2 | Barrow *(Australie)* | 10 000 | Kivalina *(USA)* | 10 000 | **0.9970** | 4.64 |
| #3 | Buckland *(USA)* | 10 000 | Barrow *(Australie)* | 10 000 | **0.9964** | 4.57 |
| #4 | Port Macquarie *(Australie)* | 45 692 | Wewak *(Canada)* | 18 230 | **0.9942** | 3.96 |
| #5 | Sitka *(USA)* | 10 000 | Edna Bay *(Australie)* | 10 000 | **0.9937** | 4.08 |
| #6 | Montréal *(Canada)* | 10 000 | Casablanca *(Australie)* | 3 144 909 | **0.9934** | 3.46 |
| #7 | Barrow *(Australie)* | 10 000 | Fairbanks *(USA)* | 32 325 | **0.9932** | 4.45 |
| #8 | Northway *(USA)* | 10 000 | Denham *(Australie)* | 10 000 | **0.9930** | 4.02 |
| #9 | Craig Cove *(Vanuatu)* | 10 000 | Sara *(USA)* | 10 000 | **0.9927** | 4.03 |
| #10 | Nadi *(Fidji)* | 42 284 | Honolulu *(USA)* | 371 657 | **0.9927** | 3.72 |

---

#### Analyse des Liens Détectés

Parmi les liens détectés, on remarque d’abord que la **majorité des populations sources sont fixées à 10 000 habitants**, ce qui indique que certaines données du graphe sont **artificielles**.

* **Liens #1, #2, #3 et #7 — Barrow (Australie) → USA :**
    Ces liaisons sont incohérentes. Selon le dataset, *Barrow* serait une ville australienne, mais d’après les sources réelles, Barrow correspond à **Utqiaġvik**, une ville située en **Alaska (USA)**.
    Il existe effectivement des vols reliant Barrow à d’autres villes américaines comme Fairbanks.
    Cette erreur révèle une **mauvaise labellisation du pays** entre l’Australie et les États-Unis.

* **Lien #4 — Port Macquarie (Australie) → Wewak (Canada) :**
    Ce lien est géographiquement impossible, reliant deux petites villes très éloignées.
    Après vérification, Wewak se situe en **Papouasie-Nouvelle-Guinée**, et non au Canada.
    Le modèle a ainsi permis de détecter une **erreur de label** entre pays voisins.

* **Lien #5 — Sitka (USA) → Edna Bay (Australie) :**
    Les deux villes se trouvent en **Alaska (USA)**, et non en Australie.
    Ce cas illustre une **encore une erreur de pays dans les données** du graphe.

* **Lien #8 — Northway (USA) → Denham (Australie) :**
    Aucun vol commercial n’existe entre ces deux lieux isolés, faiblement peuplés et très lointains.
    Le lien ne provient pas d’une erreur de pays, mais d’une **connexion qui ne devrait pas exister**.

* **Lien #9 — Craig Cove (Vanuatu) → Sara (USA) :**
    La ville *Sara* est introuvable sur internet.
    Selon ses coordonnées lon et lan des données, elle se situerait en Alaska.
    Le modèle a ainsi identifié un **lien qui est anormal entre ces deux endroits**.

* **Lien #10 — Nadi (Fidji) → Honolulu (USA) :**
    Ce lien correspond à un **vol commercial réel**, opéré de manière saisonnière entre les Fidji et Hawaï.
    Bien que valide, il a été détecté comme atypique du fait de la **distance élevée** et de l’appartenance à **deux pays différents**.
    Il s’agit du **seul lien réel** parmi les dix premiers détectés.

---

#### Conclusion (Analyse des Liens)

Cette analyse montre que le modèle **GAT + Anomaly-Guided** a permis de :
* trouver des **erreurs de pays** (labels incohérents entre Alaska et Australie),
* identifier des **connexions géographiquement improbables**,
* et de repérer également des liaisons bien réelles, mais jugées anormales en raison de leur caractère atypique par rapport aux routes aériennes les plus fréquentes (ex. Fidji ↔ Hawaï).

---

## 5. Conclusion Générale

En réponse à la question de recherche sur la détection d'anomalies, nous avons développé, testé et validé avec succès une méthodologie basée sur les GNN.

Notre démarche a consisté en une exploration systématique (Grid Search de 162 exécutions) qui a non seulement prouvé la supériorité d'un modèle **`ImprovedGAT`** sur les baselines GCN, mais a surtout démontré que l'architecture la plus performante est le **`GAT + Anomaly-Guided`**.

Notre meilleur modèle global, **`GAT_AG_h64_l2_head4`**, a atteint le score Q95 le plus bas (**1.6496**), prouvant qu'en apprenant à identifier les *liens anormaux* (partie de Youssef), il devient également meilleur pour identifier les *nœuds anormaux* (ma partie).

L'**étude d'ablation** (Attendu 2) a confirmé que l'approche multi-tâches (prédiction de population et de pays) était indispensable et non "inutilement complexe", car les modèles entraînés sur une seule tâche étaient largement sous-performants.

Enfin, le **tableau comparatif** (Attendu 1) et l'**analyse qualitative** ont validé la réussite du projet : nos modèles ont non seulement obtenu les meilleurs scores quantitatifs, mais ils ont aussi prouvé leur efficacité pratique en identifiant avec succès les erreurs de données les plus flagrantes (comme "Sydney, CANADA") et des anomalies structurelles plus complexes (comme "Taipei" ou "Kuwait").