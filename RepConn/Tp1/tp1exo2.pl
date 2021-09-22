%%%% Exo 2


	
money2([S,E,N,D,M,O,R,Y,R1,R2,R3]) :-
	fd_domain([E,N,D,O,R,Y],0,9),
	fd_domain([S,M],1,9),
	fd_domain([R1,R2,R3],0,1),
	
	fd_all_different([S,E,N,D,M,O,R,Y]),
	
	D + E #= Y + 10* R1,
	R1 + N + R #= E +R2*10,
	R2  + E + 0 #= N + R3*10,
	S + R3 + M #= O+ M*10,
	
	fd_labeling([S,E,N,D,M,O,R,Y]).
