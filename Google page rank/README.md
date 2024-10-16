L'une des applications célèbres du modèle des Chaînes de Markov est l'algorithme PageRank de Google. PageRank est l'algorithme d'analyse des liens concourant 
au système de classement des pages Web utilisé par le moteur de recherche Google, il affiche les réponses dans un ordre reposant sur un indice de popularité
des pages web. Le PageRank n'est qu'un indicateur parmi d'autres dans l'algorithme qui permet de classer les pages du Web dans les résultats de recherche de Google.
Son principe de base est d'attribuer à chaque page une valeur (ou score) proportionnelle au nombre de fois que passerait par cette page un utilisateur parcourant
le graphe du Web en cliquant aléatoirement, sur un des liens apparaissant sur chaque page. Ainsi, une page a un PageRank d'autant plus important qu'est grande
la somme des PageRanks des pages qui pointent vers elle (elle comprise, s'il y a des liens internes) . 
Le but de ce projet est d'implémenter le calcul de l'indice: il s'agit d'un calcul de vecteur propre par la méthode de la puissance itérée.
