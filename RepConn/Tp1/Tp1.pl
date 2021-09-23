%%%% Exo 


voitures([X1,X2,X3,X4,X5,X6]) :-  /* définition des domaines */  
	fd_domain(X1,[1]),   
	fd_domain(X2,[3]),  
	fd_domain(X3,2,3),
	fd_domain(X4,1,3),
	fd_domain(X5,2,3),
	fd_domain(X6,2,3),
	/* définition des contraintes */  
	fd_cardinality([X4#=X5,X4#=X6,X3#<X4,X1#<X4,X2#<X4,X5#=X6],Nbc),
	Nbc#>2,
	
	

	fd_labeling([X1,X2,X3,X4,X5,X6]).
	

