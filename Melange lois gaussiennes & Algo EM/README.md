On se place dans le cas d’un modèle de mélange de deux variables gaussiennes : l’une de moyenne 0 et de variance $\sigma^2$ donnée (avec probabilité $1 - \theta$) 
et l’autre de moyenne $\mu$ positive et de même variance (avec probabilité $\theta$). On considère la modélisation de l’effet d’un traitement,
qui a un effet nul sur une fraction $\theta$ de patients et un effet positif sur le restant de la population. 
On observe un échantillon i.i.d. de cette variable aléatoire et l’objectif est d’estimer la proportion $\theta$, la moyenne $\mu$, et la variance $\sigma^2$.
Pour se faire, on utilise un algorithme de type Expectation-Maximization (abrégé EM).
