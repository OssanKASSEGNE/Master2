/*--------------------------------------------------------------------------------*/
/*                TP diagnostic                      			*/
/* Calcul des diagnostics minimaux a partir d'observations  	*/
/* diag(L) renvoie dans L la liste des diagnostics minimaux 	*/
/* diag(L) nécessite:					*/
/*   - un fait observations(Obs): la liste des observations	 	*/
/*   - un fait composants(Comp): la liste des composants    	*/
/*--------------------------------------------------------------------------------*/

diag(Lcomp):-
	observations(Obs),
	findall(E,etat_systeme(E,Obs),Ldiag),
	filtrer(Ldiag,Ldiag,[],LdiagMin),
	convertir(LdiagMin,Lcomp).

/* Ote de Ldiag les diags non minimaux et renvoie LdiagMin */
filtrer([],_,L,L).
filtrer([E|L],Ltot,L1,L2):-
	nonMin(E,Ltot), !,
	filtrer(L,Ltot,L1,L2).
filtrer([E|L],Ltot,L1,L2):-
	filtrer(L,Ltot,[E|L1],L2).

/* Réussit si X n'est pas un diag minimal */
nonMin(X,[E|_]):-
	dif(X,E),
	inclut(X,E), !.
nonMin(X,[_|L]):-
	nonMin(X,L).

/* Réussit si le premier ensemble inclut le second */
inclut([],[]).
inclut([1|L1],[_|L2]):-
	inclut(L1,L2).
inclut([0|L1],[0|L2]):-
	inclut(L1,L2).

/* Produit des listes de composants en panne à partir d'états */
convertir(L1,L2):-
	composants(Comp),
	convertirL(L1,Comp,L2).

/* Parcourt la liste des états */
convertirL([],_,[]).
convertirL([E1|L1],Comp,[E2|L2]):-
	convertirD(E1,Comp,E2),
	convertirL(L1,Comp,L2).

/* Produit une liste de composants en panne à partir d'un état*/
convertirD([],[],[]).
convertirD([1|L1],[Co|L2],[Co|R]):-
	convertirD(L1,L2,R).
convertirD([0|L1],[_|L2],R):-
	convertirD(L1,L2,R).

dif(X,X):- !,fail.
dif(_,_).


/*------------------------------------------------------------------*/
/* La liste des états Ldiag doit être dans le même ordre 	*/
/* que la liste des composants (A1, A2, M1, M2, M3)   	*/
/* Idem pour les observations (A, B, C, D, E, F, G)        	*/
/* Codage des états :	  			*/
/* 0 = ok, 1 = panne 			*/
/*------------------------------------------------------------------*/

etat_systeme([EtatA1, EtatA2,EtatM1, EtatM2,EtatM3],[A, B, C, D, E, F, G]) :-

	%Domains definition
	fd_domain(EtatM1,0,1),
	fd_domain(EtatM2,0,1),
	fd_domain(EtatM3,0,1),
	fd_domain(EtatA1,0,1),
	fd_domain(EtatA2,0,1),


	%Definition des contraintes Etat0 => bonfonctionnement
	%M1
	(EtatM1 #= 0) #==> OutM1 #= A*C,
	%M2
	(EtatM2 #= 0) #==> OutM2 #= B*D,
	%M3
	(EtatM3 #= 0) #==> OutM3 #= C*E,
	%A1
	(EtatA1 #= 0)  #==> OutA1 #= OutM1 + OutM2,
	%A2
	(EtatA2 #= 0) #==> OutA2 #= OutM2 + OutM3,
	
	F #= OutA1,
	G #= OutA2,
	
	fd_labeling([EtatA1,EtatA2,EtatM1,EtatM2,EtatM3]).



/* Exemple d'observations dans l'ordre [A, B, C, D, E, F, G] */
/* Sur cet exemple (obs1) on doit trouver : (a1),(a2,m2), (m1), (m2,m3)*/
/*observations([3, 2, 2, 3, 3, 10, 12]).

/* Sur cet exemple (obs2) on doit trouver :  (a1,m2),(a2),(m1,m2),(m3) */
observations([3, 2, 2, 3, 3, 12, 10]).

/* Sur cet exemple (obs3) on doit trouver :  (a1,a2), (a1,m3), (a2,m1), (m2), (m1, m3) */
/* observations([3, 2, 2, 3, 3, 10, 10]).*/

/* Noms des composants (pour afficher le diagnostic) */
composants([a1, a2, m1, m2, m3]).

	

	