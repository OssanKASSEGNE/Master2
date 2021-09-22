%% Exo 3 

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
