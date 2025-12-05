# Optimisation Black-Box avec Processus Gaussiens  
Projet réalisé dans le cadre du NeurIPS 2020 BBO Post-Challenge

## Présentation générale
Ce projet s’inscrit dans le benchmark NeurIPS 2020 Black-Box Optimization Challenge, consacré à l’évaluation d’algorithmes d’optimisation en boîte noire appliqués à des tâches réelles de sélection d’hyperparamètres.  
L’optimiseur doit interroger une fonction dont la structure interne est inconnue et proposer progressivement de nouveaux hyperparamètres à tester. La plateforme appelle la fonction `suggest(...)` pour obtenir un point à évaluer, puis `observe(hp, R)` pour mettre à jour l’état interne de l’optimiseur avec le score obtenu.

Dans ce cadre, plusieurs variantes d’optimiseurs basés sur la régression par processus gaussiens (Gaussian Process Regression, GPR) ont été conçues et évaluées afin d’analyser l’impact de différentes stratégies d’acquisition et de gestion de l’incertitude.

---

## Méthodes développées

### 1. Gaussian Process Regression (GPR) – Baseline  
**Submission ID : 893732 — Score : 91.28**

Cette version repose sur un modèle de processus gaussiens associé à une fonction d’acquisition Expected Improvement.  
Les hyperparamètres sont transformés en fonction de leur nature (catégoriels, booléens, entiers, réels avec éventuelle échelle logarithmique).  
L’optimiseur mène une phase d’exploration initiale avant de s’orienter progressivement vers les régions les plus prometteuses, sur la base des observations accumulées.

### 2. GPR avec exploration adaptative  
**Submission ID : 893739 — Score : 83.24**

Cette variante modifie l’intensité de l’exploration en fonction de la variance des observations récentes.  
Une variance faible est interprétée comme un signe de stagnation et entraîne une augmentation de l’exploration.  
Une variance plus élevée conduit au contraire à la réduire pour concentrer la recherche sur les zones les mieux notées.  
Dans le cadre du challenge, cette stratégie n’a pas amélioré les résultats et a eu tendance à générer une exploration trop importante.

### 3. GPR avec seuil minimal de variance  
**Submission ID : 893749 — Score : 92.65**

Cette version introduit un seuil minimal de variance dans le calcul de la fonction d’acquisition.  
Lorsque la variance prédite devient extrêmement faible, la GPR peut se montrer trop confiante et réduire exagérément son exploration.  
En imposant un seuil plancher, le modèle reste plus prudent, continue d’explorer quand c’est nécessaire et évite de se figer prématurément.  
Cette approche est celle qui a produit les meilleures performances.

---

## Résultats

### Performance globale  
Le modèle intégrant un seuil minimal de variance montre une progression stable du score moyen au fil des itérations, jusqu’à atteindre un plateau élevé.  
Cette dynamique reflète un bon équilibre entre exploration et exploitation sur l’ensemble du benchmark.

<img width="716" height="537" alt="672b267a-a19f-478f-90d0-ca8b21b6c429" src="https://github.com/user-attachments/assets/6e3f9c0f-2cc4-4d4d-b136-df922b5bb4ef" />


### Performance par tâche  
L’évolution sur la tâche « gina » met en évidence une amélioration rapide durant les premières itérations, suivie d’un affinement progressif.  
Ce comportement indique que l’optimiseur identifie rapidement des zones prometteuses avant de stabiliser la recherche autour des meilleures régions.

<img width="713" height="533" alt="3b2b3fc1-cd30-4aa0-a34c-b934060af6ac" src="https://github.com/user-attachments/assets/6e155b14-adaf-4f37-a9a3-3a0bdc0ec6d0" />


---

## Conclusion
Les expérimentations montrent que la maîtrise de la variance prédite par le modèle joue un rôle déterminant dans la stabilité de l’optimisation.  
Le seuil minimal de variance apparaît comme un mécanisme efficace pour éviter la sur-confiance du modèle et maintenir une exploration utile, ce qui se traduit par les meilleurs résultats obtenus dans ce projet.  
À l’inverse, l’exploration adaptative s’est révélée moins performante dans ce contexte précis, probablement en raison d’ajustements trop agressifs du niveau d’exploration.

---

## Implémentations incluses
- Optimiseur GPR : `optimizer_GPR.py`  
- Optimiseur GPR avec exploration adaptative : `optimizer_GPR_adaptive.py`  
- Optimiseur GPR avec seuil de variance : `optimizer_GPR_threshold.py`


