money([S,E,N,D,M,O,R,Y]) :-
	fd_domain([E,N,D,O,R,Y],0,9),
	fd_domain([S,M],1,9),
	
	1000*(S+M)+100*(O+E)+10*(N+R) +D +E  #= 10000*M +1000*O + 100*N + 10* E + Y,
	fd_all_different([S,E,N,D,M,O,R,Y]),
	
	fd_labeling([S,E,N,D,M,O,R,Y]).
