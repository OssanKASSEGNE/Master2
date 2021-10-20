strips :- 	objectif(B), initial(S),
	format("Taper entree apres chaque etape \n\n",[]),
	strips([but(B)],S,[]), ligne.

strips(Pile,_S,_P) :- ligne, format("Pile : ",[]), write(Pile), ligne, get0(_), fail.

/*   Inserer ici strips/3   */


strips([],_,P) :- afficher_plan(P).

strips([but(B)|Pr],S,P) :- filtre(B,S), message(1,B), strips(Pr,S,P). 


strips([but([B])|Pr],S,P) :- non_filtre(B,S), 
				action(N,Pred,_,L_ajout), 
				membre(B,L_ajout),
				message(2,N),
				strips([but(Pred),act(N)|Pr],S,P).
				

strips([but(B)|Pr],S,P) :- non_filtre(B,S),
				compose(B), 
				message(3,B), 
				decomposer(B,B_simple), 
				conc(B_simple,[but(B)|Pr],Res), 
				strips(Res,S,P). 
				

strips([act(A)|Pr],S,P) :- message(4,A), appliquer(A,S,Sn), 
				message(5,Sn), 
				strips(Pr,Sn,[A|P]).
     
     
          
afficher_plan([]) :- 
	!, ligne, format("Plan solution : ",[]), ligne.
afficher_plan([A1|A2]) :- 
	afficher_plan(A2), format("   -",[]), write(A1), ligne.

filtre([],_).
filtre([B1|B2],S) :- membre(B1,S), filtre(B2,S).

non_filtre(B,S) :- filtre(B,S), !, fail.
non_filtre(_B,_S).

membre(E, [E|_]).
membre(E,[_|L]) :- membre(E,L).

compose([_,_|_]).

decomposer([],[]).
decomposer([B1|B2],[but([B1])|B3]) :- decomposer(B2,B3).

conc([],L2,L2).
conc([X|L1],L2,[X|L3]) :- conc(L1,L2,L3).

appliquer(N_action,S_courante,S_nouvelle) :-
	action(N_action,_P,S,A),
	retirer(S,S_courante,S_inter),
	conc(A,S_inter,S_nouvelle).

retirer([],S,S).
retirer([R1|R2],S,S2) :- oter(R1,S,S1), retirer(R2, S1, S2).

oter(_E,[],[]).
oter(E,[E|L],L) :- !.
oter(E,[A|L],[A|L1]) :- oter(E,L,L1).

ligne :- format("\n",[]).

message(1,B) :- 
	format("Le but ",[]), write(B), format(" filtre la situation",[]), ligne.
message(2,A) :-
	format("On empile l'action ",[]), write(A), ligne.
message(3,B) :- format("On empile les sous-buts ",[]), write(B), ligne.
message(4,A) :- format("*** On execute l'action ",[]), write(A), format(" ***",[]), ligne.
message(5,S) :- format("Nouvelle situation ",[]), write(S), ligne.


/*
initial([lasagnes]).
objectif([lasagnes,rassasie]).
action(manger_lasagnes,[lasagnes],[lasagnes],[sans_lasagnes,rassasie]).
action(cuire_lasagnes,[sans_lasagnes],[sans_lasagnes],[lasagnes]). */


initial([singe(a),caisse(b),bananes(c),sursol]).
objectif([avoir_bananes]).
action(aller(X,Y),[singe(X),sursol],[singe(X)],[singe(Y)]).
action(pousser(X,Y),[caisse(X),singe(X),sursol],[caisse(X),singe(X)],[caisse(Y),singe(Y)]).
action(monter,[caisse(X),singe(X),sursol],[sursol],[surcaisse]).
action(descendre,[surcaisse],[surcaisse],[sursol]).
action(attraper,[bananes(X),caisse(X),surcaisse],[],[avoir_bananes]). 










