Ce projet vise à analyser les performances d'un modèle de classification SVM (Support Vector Machine) appliqué au jeu de données Iris,
en se concentrant sur les classes 1 et 2 pour simplifier le problème. Après avoir importé et filtré les données, nous séparons celles-ci
en ensembles d'entraînement et de test. Le projet compare les performances d'un SVM avec un noyau linéaire et d'un SVM avec un noyau polynomial,
en ajustant des paramètres comme la régularisation C, le degré du polynôme, et le paramètre gamma. Des tests de précision (accuracy) sont réalisés pour chaque modèle,
et une validation croisée à 5 plis est utilisée pour évaluer la robustesse du modèle linéaire. Les résultats permettent
d'étudier l'impact de chaque configuration sur la précision de la classification et d'optimiser le choix des hyperparamètres pour les futurs modèles.
