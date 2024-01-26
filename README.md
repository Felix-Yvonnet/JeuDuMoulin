# Jeu Du Moulin

## Le jeu
Ce jeu se décompose en 2 phases :
- poser tour à tour chacun de ses 9 pions sur une case vide.
- déplacer ses pions vers une case vide.

Avoir trois pions alignés (selon le graphe) permet de supprimer un pion de l'adversaire.

Si l'un des joueurs à moins de trois pions ou ne peut pas jouer, il perd.

## Le problème

Il existe une stratégie optimale (théorique et pratique) mais elle est trop complexe pour être réellement implémentée. On va donc se rabbatre sur de l'intélligence artificielle.

### Utilisation
Entrainer :
```sh
python3 train.py --epoch 13 --trains 5000 --iter-fun 5 --name modelQuiteWorking.pt --tests 100
```

Tester contre une politique random intelligente :
```sh
python3 tests.py
```

Observer
```sh
python3 game.py --load-model models/model.pt
```

### Makefile
Pour aller (un peu) plus vite vous pouvez aussi utiliser le Makefile.

Pour lancer le jeu
```
make
```


