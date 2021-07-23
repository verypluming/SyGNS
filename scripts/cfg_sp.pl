% A CFG to generate schematic sentences for Monotonicity Semantics Parsing

:- use_module(betaConversion,[betaConvert/2]).
:- use_module(fol2nltk,[fol2nltk/2]).
:- use_module(fol2tptp,[fol2tptp/2]).
:- use_module(comsemPredicates,[infix/0,
                                prefix/0,
                                printRepresentations/1]).

/* ==============================
   Main Rules
============================== */

%%% Sentence %%%

s([depth:D,sel:K]) -->
    np([depth:D,sel:K]),
    iv([form:fin]).

s([depth:D,sel:K]) -->
    np([depth:D,sel:K]),
    iv([form:fin]),
    vmod([form:fin]).

s([depth:0,sel:_]) -->
    pn,
    iv([form:fin]).

s([depth:0,sel:_]) -->
    pn,
    iv([form:fin]),
    vmod([form:fin]).

s([depth:0,sel:_]) -->
    pn,
    tv([form:fin]),
    pn.

s([depth:D,sel:K]) -->
    pn,
    tv([form:fin]),
    np([depth:D,sel:K]).

s([depth:D,sel:K]) -->
    np([depth:D,sel:K]),
    tv([form:fin]),
    pn.

% Negation
s([depth:D,sel:K]) -->
    np([depth:D,sel:K]),
    neg,
    iv([form:inf]).

s([depth:D,sel:K]) -->
    np([depth:D,sel:K]),
    neg,
    iv([form:inf]),
    vmod([form:inf]).

s([depth:0,sel:_]) -->
    pn,
    neg,
    iv([form:inf]).

s([depth:0,sel:_]) -->
    pn,
    neg,
    iv([form:inf]),
    vmod([form:inf]).

s([depth:0,sel:_]) -->
    pn,
    neg,
    tv([form:fin]),
    pn.

s([depth:D,sel:K]) -->
    pn,
    neg,
    tv([form:inf]),
    np([depth:D,sel:K]).

s([depth:D,sel:K]) -->
    np([depth:D,sel:K]),
    neg,
    tv([form:inf]),
    pn.

%%% Noun Phrase %%%

np([depth:0,sel:_]) -->
    det([num:NUM]),
    n([num:NUM]).

np([depth:0,sel:_]) -->
    !,
    det([num:NUM]),
    adj,
    n([num:NUM]).

np([depth:D1,sel:K]) -->
    {D2 is D1 - 1},
    det([num:NUM]),
    n([num:NUM]),
    sbar([depth:D2,sel:K]).


%%% Sbar %%%

sbar([depth:D,sel:K]) -->
    whnp_sbj,
    tv([form:fin]),
    np([depth:D,sel:K]),
    {selector(K)}.

sbar([depth:D,sel:K]) -->
    whnp_obj,
    np([depth:D,sel:K]),
    tv([form:fin]),
    {selector(K)}.

sbar([depth:_,sel:K]) -->
    whnp_sbj,
    iv([form:fin]),
    {selector(K)}.

sbar([depth:_,sel:K]) -->
    whnp_sbj,
    iv([form:fin]),
    vmod([form:inf]),
    {selector(K)}.

sbar([depth:D,sel:K]) -->
    whnp_sbj,
    neg,
    tv([form:inf]),
    np([depth:D,sel:K]),
    {selector(K)}.

sbar([depth:D,sel:K]) -->
    whnp_obj,
    np([depth:D,sel:K]),
    neg,
    tv([form:inf]),
    {selector(K)}.

sbar([depth:_,sel:K]) -->
    whnp_sbj,
    neg,
    iv([form:inf]),
    {selector(K)}.

sbar([depth:_,sel:K]) -->
    whnp_sbj,
    neg,
    iv([form:inf]),
    vmod([form:inf]),
    {selector(K)}.

%%% Verbal Modifier %%%

vmod([form:_]) --> adv.
vmod([form:F]) --> conn, iv([form:F]).


/* ==============================
   Lexicon
============================== */

%%% Noun %%%
n([num:Num]) -->
    {lex(n,[surf:Surf,num:Num])},
    Surf.

%%% Proper Noun %%%
pn -->
    {lex(pn,[surf:Surf])},
    Surf.

%%% Negation %%%
neg -->
    {lex(neg,[surf:Surf])},
    Surf.

%%% Wh-NP %%%
whnp_sbj -->
    {lex(whnp_sbj,[surf:Surf])},
    Surf.

whnp_obj -->
    {lex(whnp_obj,[surf:Surf])},
    Surf.

%%% Determiner %%%
det([num:Num]) -->
    {lex(det,[surf:Surf,num:Num])},
    Surf.

%%% Intransitive Verb %%%
iv([form:Form]) -->
    {lex(iv,[surf:Surf,form:Form])},
    Surf.

%%% Transitive Verb %%%
tv([form:Form]) -->
    {lex(tv,[surf:Surf,form:Form])},
    Surf.

%%% Adjectives %%%
adj -->
    {lex(adj,[surf:Surf])},
    Surf.

%%% Adverbs %%%
adv -->
    {lex(adv,[surf:Surf])},
    Surf.


%%% Connectives %%%
conn -->
    {lex(conn,[surf:Surf])},
    Surf.


% /* ==============================
%   Lexical Entries
% ============================== */

% Noun
lex(n,[surf:[nounSing],num:sing]).
lex(n,[surf:[nounPlur],num:plur]).

% Proper Noun
lex(pn,[surf:[pn]]).

% Negation
lex(neg,[surf:[did,not]]).

% WH-NP
% lex(whnp_sbj,[surf:[whSubj]]).
% lex(whnp_obj,[surf:[whObj]]).
lex(whnp_sbj,[surf:[that]]).
lex(whnp_obj,[surf:[that]]).
% lex(whnp_sbj,[surf:[which]]).
% lex(whnp_obj,[surf:[which]]).

% Determienr
lex(det,[surf:[detSing],num:sing]).
lex(det,[surf:[detPlur],num:plur]).

% lex(det,[surf:[some],num:sing]).
% lex(det,[surf:[a],num:sing]).
% lex(det,[surf:[every],num:sing]).
% lex(det,[surf:[each],num:sing]).
% lex(det,[surf:[no],num:sing]).
% lex(det,[surf:[several],num:plur]).
% lex(det,[surf:[all],num:plur]).
% lex(det,[surf:[few],num:plur]).
% lex(det,[surf:[a,few],num:plur]).
% lex(det,[surf:[at,least,three],num:plur]).
% lex(det,[surf:[less,than,three],num:plur]).
% lex(det,[surf:[more,than,three],num:plur]).
% lex(det,[surf:[at,most,three],num:plur]).

% Intransitive Verb
lex(iv,[surf:[ivFin],form:fin]).
lex(iv,[surf:[ivInf],form:inf]).

% Transitive Verb
lex(tv,[surf:[tvFin],form:fin]).
lex(tv,[surf:[tvInf],form:inf]).

% Adjective
lex(adj,[surf:[adj]]).

% Adverb
lex(adv,[surf:[adv]]).

% ConneAdverb
% lex(conn,[surf:[and]]).
% lex(conn,[surf:[or]]).
lex(conn,[surf:[conn]]).


/* ==============================
  Auxiliary predicates
============================== */

yield([]).
yield([X|List]) :-
    write(X), write(' '), yield(List).


% leq(N,N).
% leq(_,0) :- !, fail.
% leq(N1,N2):-
%     M is N2 - 1, leq(N1,M).

% le(N,M) :- leq(N,M), N =\= M.

selector(N) :- random_between(1,N,1).

% compose(Term,Symbol,ArgList):-
%     Term =.. [Symbol|ArgList].

% nicePrint(SR):-
%    \+ \+ (numbervars(SR,0,_), print(SR)).


/* ==============================
   Main Predicates
============================== */

% Generate a plain sentence with depth N and selector K
plain(N,K) :-
   s([depth:N,sel:K],Sentence,[]),
   % selector(K),
   yield(Sentence),nl,
   fail.

