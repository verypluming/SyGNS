
:- module(fol2nltk,[fol2nltk/2]).

:- use_module(comsemPredicates,[basicFormula/1]).

/*========================================================================
   Translates formula to NLTK syntax on Stream
========================================================================*/

fol2nltk(Formula,Stream):-
   numbervars(Formula,0,_),printnltk(Formula,Stream).

   % write(Stream,'input_formula(comsem,conjecture,'),
   % \+ \+ ( numbervars(Formula,0,_),printNLTK(Formula,Stream) ),
   % write(Stream,').'),
   % numbervars(Formula,0,_),printnltk(Formula,Stream), !,
   % nl(Stream).


/*========================================================================
   Print nltk formulas
========================================================================*/

printnltk(some(X,Formula),Stream):- !,
   write(Stream,'(exists '),
   write_term(Stream,X,[numbervars(true)]),
   write(Stream,'.'),
   printnltk(Formula,Stream),
   write(Stream,')').

printnltk(all(X,Formula),Stream):- !,
   write(Stream,'(forall '),
   write_term(Stream,X,[numbervars(true)]),
   write(Stream,'.'),
   printnltk(Formula,Stream),
   write(Stream,')').

printnltk(and(Phi,Psi),Stream):- !,
   write(Stream,'('),
   printnltk(Phi,Stream),
   write(Stream,' & '),
   printnltk(Psi,Stream),
   write(Stream,')').

printnltk(or(Phi,Psi),Stream):- !,
   write(Stream,'('),
   printnltk(Phi,Stream),
   write(Stream,' | '),
   printnltk(Psi,Stream),
   write(Stream,')').

printnltk(imp(Phi,Psi),Stream):- !,
   write(Stream,'('),
   printnltk(Phi,Stream),
   write(Stream,' -> '),
   printnltk(Psi,Stream),
   write(Stream,')').

printnltk(not(Phi),Stream):- !,
   write(Stream,'- '),
   printnltk(Phi,Stream).

printnltk(Phi,Stream):-
   basicFormula(Phi),
   write_term(Stream,Phi,[numbervars(true)]).
