/*
*@author Ossan KASSEGNE M2 INFO
*
*/

%Prerequisites
%count number of solutions
count(P,Count) :-
        findall(1,P,L),
        length(L,Count).

  /*************************************/
 /*		       Training			      */
/*************************************/

%Part1 With all constraints
voitures([X1,X2,X3,X4,X5,X6]) :-

	% Domains definition
	fd_domain(X1,[1]),   
	fd_domain(X2,[3]),  
	fd_domain(X3,2,3),
	fd_domain(X4,1,4),
	fd_domain(X5,2,4),
	fd_domain(X6,2,4),
	/* définition des contraintes */  
	X4#=X5,
	X5#=X6,
	X4#=X6,
	X3#<X4,
	X1#<X4,
	X2#<X4,
	
	fd_labeling([X1,X2,X3,X4,X5,X6]).

% Results
% voitures(L).
% L = [1,3,2,4,4,4] 
% L = [1,3,3,4,4,4]

%Part2 With less constraints
voitures2([X1,X2,X3,X4,X5,X6],NbContraintes) :-

	% Domains definition
	fd_domain(X1,[1]),   
	fd_domain(X2,[3]),  
	fd_domain(X3,2,3),
	fd_domain(X4,1,3),
	fd_domain(X5,2,3),
	fd_domain(X6,2,3),

	% Constraints 
	fd_cardinality([X4#=X5,X4#=X6,X3#<X4,X1#<X4,X2#<X4,X5#=X6],NbContraintes),
	NbContraintes#>=4,
	

	fd_labeling([X1,X2,X3,X4,X5,X6]).

% Results
% voitures2(L,N).  
% L = [1,3,2,2,2,2]
% N = 4 
%
% L = [1,3,2,3,3,3]
% N = 5
%
% L = [1,3,3,2,2,2]
% N = 4
%
% L = [1,3,3,3,3,3]
% N = 4


  /*************************************/
 /*		       EXO1 Reines			  */
/*************************************/

reines([X1,X2,X3,X4]) :-  

	% Domains definition
	fd_domain([X1,X2,X3,X4],1,4), 

	% Constraints 
	fd_all_different([X1,X2,X3,X4]),
	
	fd_relation([[1,3],[1,4],[2,4],[3,1],[4,1],[4,2]],[X1,X2]),
	fd_relation([[1,2],[1,4],[2,1],[2,3],[3,2],[3,4],[4,1],[4,3]],[X1,X3]),
	fd_relation([[1,2],[1,3],[2,1],[2,3],[2,4],[3,1],[3,2],[4,2],[4,3]],[X1,X4]),
	fd_relation([[1,3],[1,4],[2,4],[3,1],[4,1],[4,2]],[X2,X3] ),
	fd_relation([[1,2],[1,4],[2,1],[2,3],[3,2],[3,4],[4,1],[4,3]],[X2,X4]),
	fd_relation([[1,3],[1,4],[2,4],[3,1],[4,1],[4,2]],[X3,X4]),
	
	fd_labeling([X1,X2,X3,X4]).

% Results
% reines(L).
% L = [2,4,1,3] 
% L = [3,1,4,2]

/***************************************/
/*		        EXO2				  */
/**************************************/
% version 1.0
money([S,E,N,D,M,O,R,Y]) :-

	% Domains definition
	fd_domain([E,N,D,O,R,Y],0,9),
	fd_domain([S,M],1,9),
	
	% Constraints 
	1000*(S+M)+100*(O+E)+10*(N+R) +D +E  #= 10000*M +1000*O + 100*N + 10* E + Y,
	fd_all_different([S,E,N,D,M,O,R,Y]),
	
	fd_labeling([S,E,N,D,M,O,R,Y]).

% Results
% money(R)
% R = [9,5,6,7,1,0,8,2]

%version2.0	
money2([S,E,N,D,M,O,R,Y,R1,R2,R3]) :-

	% Domains definition
	fd_domain([E,N,D,O,R,Y],0,9),
	fd_domain([S,M],1,9),
	fd_domain([R1,R2,R3],0,1),
	
	% Constraints 
	fd_all_different([S,E,N,D,M,O,R,Y]),
	D + E #= Y + 10* R1,
	R1 + N + R #= E +R2*10,
	R2  + E + 0 #= N + R3*10,
	S + R3 + M #= O+ M*10,
	
	fd_labeling([S,E,N,D,M,O,R,Y]).

% Results
% money2(R2).
% R2 = [9,5,6,7,1,0,8,2,1,1,0]




/***************************************/
/*		        EXO3				  */
/**************************************/

% cinema = 1
% theatre = 2
% concert = 3
% pub = 4


sortie([A,B,C,D]):-
	fd_domain([A,B,C,D],1,4),
	
	/**Contraintes**/
	(A#=1) #==> (D#=1),
	(C#=2) #<=> (B#=3),
	(D#=B) #<=> (B#\=3),
	(B#=4) #==> (A#=4),
	(A#=3) #<=> ((B#=3) #/\ (C#=3)),
	(D#\=4) #==> ((A#=2) #/\ (B#=2)),
	(C#=2) #<=> (D#=2),
	((A#=4) #\/ (D#=4)) #==> (C#=4),
	
	fd_labeling([A,B,C,D]).

	% Results
	% sortie(R).
 	% R = [4,4,4,4] On va tous au pub




/***************************************/
/*		        EXO4				  */
/**************************************/


	%% Part 1 CSP
monnaie(T, P, [E2,E1,C50,C20,C10], [XE2,XE1,XC50,XC20,XC10]):-

	% Domains definition
	fd_domain(XE2,0,E2),
	fd_domain(XE1,0,E1),
	fd_domain(XC50,0,C50),
	fd_domain(XC20,0,C20),
	fd_domain(XC10,0,C10),

	% Constraints
	T-P #= 200*XE2 + 100*XE1 + 50*XC50 + 20*XC20 + 10*XC10,

	fd_labeling([XE2,XE1,XC50,XC20,XC10]).

	%% Part 2 Tests (findall)

	% Cas 1
	% count(monnaie(200,90,[10,10,10,10,10],L),L2).
	% L2 = 11
	% On a 11 solutions

	% Cas 2 
	% count(monnaie(200,90,[10,10,10,10,1],L),L2). 
	% L2 = 4
	% On a 4 solutions

	% Cas 3
	% count(monnaie(200,90,[10,10,0,10,0],L),L2). 
	% L2 = 0
	% On a aucune solution (CSP inconsistant)


/***************************************/
/*		        EXO5				  */
/**************************************/

	% Variables
	% CF = "creuser fondation",
	% CM = "Construire les murs",
	% PPF = "Poser portes et fenêtres"
	% PT = "Poser le toit",
	% PC = "Poser la cheminée",
	% PM = "Peindre les murs",
	% IE = "Installer l'électricité"
	% JR = "Jour de repos"

travaux(Duration,[CF,CM,PPF,PT,PC,PM,IE,JR]):-

	% Domains definition
	
	% Part 1 
	fd_domain([CF,CM,PPF,PT,PC,PM,IE,JR],1, Duration),

	%Constraints
	CF #< CM,
	CM #< PPF,
	CM #< PT,
	PM #>= CM + 3,
	PC #> PT,
	PC #\=PM,
	IE #> PM,
	(JR #\= CF) #/\ (JR #\= CM) #/\ (JR #\= PPF) #/\ (JR #\= PT)  #/\ (JR #\= IE) #/\ (JR #\= PC) #/\ (JR #\= PM),
	
	fd_labeling([CF,CM,PPF,PT,PC,PM,IE,JR]).

	% Part 2 Tests (Changement de taille du domaine)

	% count(travaux(6,L),C).
	% C = 12
	% CSP est consistant à partir d'une durée minimum de 6 jours (12 solutions)

	% Part 3 (Variables ayant même valeurs, Duration = 6)
	% CF = "creuser fondation" = jour 1
	% CM = "Construire les murs" = jour 2
	% PM = "Peindre les murs" = jour 5
	% PC = "Poser la cheminée" = jour 6
	% IE = "Installer l'électricité" = jour 6