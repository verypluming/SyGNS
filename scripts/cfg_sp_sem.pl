/************************************************************************

 A CFG to convert sentences to first-order logic and variable-free form

*************************************************************************/

:- use_module(betaConversion,[betaConvert/2]).
:- use_module(fol2nltk,[fol2nltk/2]).
:- use_module(comsemPredicates,[infix/0,
                                prefix/0,
                                printRepresentations/1]).
:- use_module(morphDic,[convertSymbol/2]).


/* ==============================
   Main Rules
============================== */

%%% Sentence %%%

s([sem:SR,sys:Sys,depth:D,sel:K,phen:[sbj_quant:yes,obj_quant:no,neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]) -->
    np([sem:SR1,sys:Sys,depth:D,sel:K,phen:[neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]),
    iv([sem:SR2,sys:Sys,form:fin]),
    {combine(Sys,s:SR,[np:SR1,iv:SR2])}.

s([sem:SR,sys:Sys,depth:D,sel:K,phen:[sbj_quant:yes,obj_quant:no,neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]) -->
    np([sem:SR1,sys:Sys,depth:D,sel:K,phen:[neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv1,conj:Conj1,disj:Disj1]]),
    iv([sem:SR2,sys:Sys,form:fin]),
    vmod([sem:SR3,sys:Sys,form:fin,adv:Adv2,conj:Conj2,disj:Disj2]),
    {labelJudge(Adv1,Adv2,Adv)},
    {labelJudge(Conj1,Conj2,Conj)},
    {labelJudge(Disj1,Disj2,Disj)},
    {combine(Sys,s:SR,[np:SR1,iv:SR2,vmod:SR3])}.

s([sem:SR,sys:Sys,depth:0,sel:_,phen:[sbj_quant:no,obj_quant:no,neg:no,adj:no,per_emb:no,cen_emb:no,adv:no,conj:no,disj:no]]) -->
    pn([sem:SR1,sys:Sys]),
    iv([sem:SR2,sys:Sys,form:fin]),
    {combine(Sys,s:SR,[np:SR1,iv:SR2])}.

s([sem:SR,sys:Sys,depth:0,sel:_,phen:[sbj_quant:no,obj_quant:no,neg:no,adj:no,per_emb:no,cen_emb:no,adv:Adv,conj:Conj,disj:Disj]]) -->
    pn([sem:SR1,sys:Sys]),
    iv([sem:SR2,sys:Sys,form:fin]),
    vmod([sem:SR3,sys:Sys,form:fin,adv:Adv,conj:Conj,disj:Disj]),
    {combine(Sys,s:SR,[np:SR1,iv:SR2,vmod:SR3])}.

%%%%%
s([sem:SR,sys:Sys,depth:0,sel:_,phen:[sbj_quant:no,obj_quant:no,neg:no,adj:no,per_emb:no,cen_emb:no,adv:no,conj:no,disj:no]]) -->
    pn([sem:SR1,sys:Sys]),
    tv([sem:SR2,sys:Sys,form:fin]),
    pn([sem:SR3,sys:Sys]),
    {combine(Sys,s:SR,[np:SR1,tv:SR2,np:SR3])}.
%%%%

s([sem:SR,sys:Sys,depth:D,sel:K,phen:[sbj_quant:no,obj_quant:yes,neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]) -->
    pn([sem:SR1,sys:Sys]),
    tv([sem:SR2,sys:Sys,form:fin]),
    np([sem:SR3,sys:Sys,depth:D,sel:K,phen:[neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]),
    {combine(Sys,s:SR,[np:SR1,tv:SR2,np:SR3])}.

s([sem:SR,sys:Sys,depth:D,sel:K,phen:[sbj_quant:yes,obj_quant:no,neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]) -->
    np([sem:SR1,sys:Sys,depth:D,sel:K,phen:[neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]),
    tv([sem:SR2,sys:Sys,form:fin]),
    pn([sem:SR3,sys:Sys]),
    {combine(Sys,s:SR,[np:SR1,tv:SR2,np:SR3])}.

% Negation

s([sem:SR,sys:Sys,depth:D,sel:K,phen:[sbj_quant:yes,obj_quant:no,neg:yes,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]) -->
    np([sem:SR1,sys:Sys,depth:D,sel:K,phen:[neg:_,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]),
    neg([sem:SR2,sys:Sys]),
    iv([sem:SR3,sys:Sys,form:inf]),
    {combine(Sys,s:SR,[np:SR1,neg:SR2,iv:SR3])}.

s([sem:SR,sys:Sys,depth:D,sel:K,phen:[sbj_quant:yes,obj_quant:no,neg:yes,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]) -->
    np([sem:SR1,sys:Sys,depth:D,sel:K,phen:[neg:_,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv1,conj:Conj1,disj:Disj1]]),
    neg([sem:SR2,sys:Sys]),
    iv([sem:SR3,sys:Sys,form:inf]),
    vmod([sem:SR4,sys:Sys,form:inf,adv:Adv2,conj:Conj2,disj:Disj2]),
    {labelJudge(Adv1,Adv2,Adv)},
    {labelJudge(Conj1,Conj2,Conj)},
    {labelJudge(Disj1,Disj2,Disj)},
    {combine(Sys,s:SR,[np:SR1,neg:SR2,iv:SR3,vmod:SR4])}.

s([sem:SR,sys:Sys,depth:0,sel:_,phen:[sbj_quant:no,obj_quant:no,neg:yes,adj:no,per_emb:no,cen_emb:no,adv:no,conj:no,disj:no]]) -->
    pn([sem:SR1,sys:Sys]),
    neg([sem:SR2,sys:Sys]),
    iv([sem:SR3,sys:Sys,form:inf]),
    {combine(Sys,s:SR,[np:SR1,neg:SR2,iv:SR3])}.

s([sem:SR,sys:Sys,depth:0,sel:_,phen:[sbj_quant:no,obj_quant:no,neg:yes,adj:no,per_emb:no,cen_emb:no,adv:Adv,conj:Conj,disj:Disj]]) -->
    pn([sem:SR1,sys:Sys]),
    neg([sem:SR2,sys:Sys]),
    iv([sem:SR3,sys:Sys,form:inf]),
    vmod([sem:SR4,sys:Sys,form:inf,adv:Adv,conj:Conj,disj:Disj]),
    {combine(Sys,s:SR,[np:SR1,neg:SR2,iv:SR3,vmod:SR4])}.

%%%%%
s([sem:SR,sys:Sys,depth:0,sel:_,phen:[sbj_quant:no,obj_quant:no,neg:yes,adj:no,per_emb:no,cen_emb:no,adv:no,conj:no,disj:no]]) -->
    pn([sem:SR1,sys:Sys]),
    neg([sem:SR2,sys:Sys]),
    tv([sem:SR3,sys:Sys,form:fin]),
    pn([sem:SR4,sys:Sys]),
    {combine(Sys,s:SR,[np:SR1,neg:SR2,tv:SR3,np:SR4])}.
%%%%

s([sem:SR,sys:Sys,depth:D,sel:K,phen:[sbj_quant:no,obj_quant:yes,neg:yes,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]) -->
    pn([sem:SR1,sys:Sys]),
    neg([sem:SR2,sys:Sys]),
    tv([sem:SR3,sys:Sys,form:inf]),
    np([sem:SR4,sys:Sys,depth:D,sel:K,phen:[neg:_,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]),
    {combine(Sys,s:SR,[np:SR1,neg:SR2,tv:SR3,np:SR4])}.

s([sem:SR,sys:Sys,depth:D,sel:K,phen:[sbj_quant:yes,obj_quant:no,neg:yes,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]) -->
    np([sem:SR1,sys:Sys,depth:D,sel:K,phen:[neg:_,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]),
    neg([sem:SR2,sys:Sys]),
    tv([sem:SR3,sys:Sys,form:inf]),
    pn([sem:SR4,sys:Sys]),
    {combine(Sys,s:SR,[np:SR1,neg:SR2,tv:SR3,np:SR4])}.

%%% Noun Phrase %%%

np([sem:SR,sys:Sys,depth:0,sel:_,phen:[neg:no,adj:no,per_emb:no,cen_emb:no,adv:no,conj:no,disj:no]]) -->
    det([sem:SR1,sys:Sys,num:NUM]),
    n([sem:SR2,sys:Sys,num:NUM]),
    {combine(Sys,np:SR,[det:SR1,n:SR2])}.

np([sem:SR,sys:Sys,depth:0,sel:_,phen:[neg:no,adj:yes,per_emb:no,cen_emb:no,adv:no,conj:no,disj:no]]) -->
    % !,
    det([sem:SR1,sys:Sys,num:NUM]),
    adj([sem:SR2,sys:Sys]),
    n([sem:SR3,sys:Sys,num:NUM]),
    {combine(Sys,np:SR,[det:SR1,adj:SR2,n:SR3])}.

np([sem:SR,sys:Sys,depth:D1,sel:K,phen:[neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]) -->
    {D2 is D1 - 1},
    det([sem:SR1,sys:Sys,num:NUM]),
    n([sem:SR2,sys:Sys,num:NUM]),
    sbar([sem:SR3,sys:Sys,depth:D2,sel:K,phen:[neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]),
    {combine(Sys,np:SR,[det:SR1,n:SR2,sbar:SR3])}.

%%% Sbar %%%

sbar([sem:SR,sys:Sys,depth:D,sel:K,phen:[neg:Neg,adj:Adj,per_emb:yes,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]) -->
    whnp_sbj([sem:SR1,sys:Sys]),
    tv([sem:SR2,sys:Sys,form:fin]),
    np([sem:SR3,sys:Sys,depth:D,sel:K,phen:[neg:Neg,adj:Adj,per_emb:_,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]),
    {combine(Sys,sbar:SR,[whnp_sbj:SR1,tv:SR2,np:SR3])},
    {selector(K)}.

sbar([sem:SR,sys:Sys,depth:D,sel:K,phen:[neg:Neg,adj:Adj,per_emb:Per,cen_emb:yes,adv:Adv,conj:Conj,disj:Disj]]) -->
    whnp_obj([sem:SR1,sys:Sys]),
    np([sem:SR2,sys:Sys,depth:D,sel:K,phen:[neg:Neg,adj:Adj,per_emb:Per,cen_emb:_,adv:Adv,conj:Conj,disj:Disj]]),
    tv([sem:SR3,sys:Sys,form:fin]),
    {combine(Sys,sbar:SR,[whnp_obj:SR1,np:SR2,tv:SR3])},
    {selector(K)}.

sbar([sem:SR,sys:Sys,depth:_,sel:K,phen:[neg:no,adj:no,per_emb:no,cen_emb:no,adv:no,conj:no,disj:no]]) -->
    whnp_sbj([sem:SR1,sys:Sys]),
    iv([sem:SR2,sys:Sys,form:fin]),
    {combine(Sys,sbar:SR,[whnp_sbj:SR1,iv:SR2])},
    {selector(K)}.

sbar([sem:SR,sys:Sys,depth:_,sel:K,phen:[neg:no,adj:no,per_emb:no,cen_emb:no,adv:Adv,conj:Conj,disj:Disj]]) -->
    whnp_sbj([sem:SR1,sys:Sys]),
    iv([sem:SR2,sys:Sys,form:fin]),
    vmod([sem:SR3,sys:Sys,form:inf,adv:Adv,conj:Conj,disj:Disj]),
    {combine(Sys,sbar:SR,[whnp_sbj:SR1,iv:SR2,vmod:SR3])},
    {selector(K)}.

sbar([sem:SR,sys:Sys,depth:D,sel:K,phen:[neg:yes,adj:Adj,per_emb:yes,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]) -->
    whnp_sbj([sem:SR1,sys:Sys]),
    neg([sem:SR2,sys:Sys]),
    tv([sem:SR3,sys:Sys,form:inf]),
    np([sem:SR4,sys:Sys,depth:D,sel:K,phen:[neg:_,adj:Adj,per_emb:_,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]]),
    {combine(Sys,sbar:SR,[whnp_sbj:SR1,neg:SR2,tv:SR3,np:SR4])},
    {selector(K)}.

sbar([sem:SR,sys:Sys,depth:D,sel:K,phen:[neg:yes,adj:Adj,per_emb:Per,cen_emb:yes,adv:Adv,conj:Conj,disj:Disj]]) -->
    whnp_obj([sem:SR1,sys:Sys]),
    np([sem:SR2,sys:Sys,depth:D,sel:K,phen:[neg:_,adj:Adj,per_emb:Per,cen_emb:_,adv:Adv,conj:Conj,disj:Disj]]),
    neg([sem:SR3,sys:Sys]),
    tv([sem:SR4,sys:Sys,form:inf]),
    {combine(Sys,sbar:SR,[whnp_obj:SR1,np:SR2,neg:SR3,tv:SR4])},
    {selector(K)}.

sbar([sem:SR,sys:Sys,depth:_,sel:K,phen:[neg:yes,adj:no,per_emb:no,cen_emb:no,adv:no,conj:no,disj:no]]) -->
    whnp_sbj([sem:SR1,sys:Sys]),
    neg([sem:SR2,sys:Sys]),
    iv([sem:SR3,sys:Sys,form:inf]),
    {combine(Sys,sbar:SR,[whnp_sbj:SR1,neg:SR2,iv:SR3])},
    {selector(K)}.

sbar([sem:SR,sys:Sys,depth:_,sel:K,phen:[neg:yes,adj:no,per_emb:no,cen_emb:no,adv:Adv,conj:Conj,disj:Disj]]) -->
    whnp_sbj([sem:SR1,sys:Sys]),
    neg([sem:SR2,sys:Sys]),
    iv([sem:SR3,sys:Sys,form:inf]),
    vmod([sem:SR4,sys:Sys,form:inf,adv:Adv,conj:Conj,disj:Disj]),
    {combine(Sys,sbar:SR,[whnp_sbj:SR1,neg:SR2,iv:SR3,vmod:SR4])},
    {selector(K)}.

%%% Verbal Modifier %%%

vmod([sem:SR,sys:Sys,form:_,adv:yes,conj:no,disj:no]) -->
    adv([sem:SR1,sys:Sys]),
    {combine(Sys,vmod:SR,[adv:SR1])}.
vmod([sem:SR,sys:Sys,form:F,adv:no,conj:Conj,disj:Disj]) -->
    conn([sem:SR1,sys:Sys,conj:Conj,disj:Disj]),
    iv([sem:SR2,sys:Sys,form:F]),
    {combine(Sys,vmod:SR,[conn:SR1,iv:SR2])}.


/* ==============================
   Lexicon
============================== */

%%% Noun %%%
n([sem:SR,sys:Sys,num:Num]) -->
    {lex(n,[surf:Surf,num:Num])},
    Surf,
    {semlex(Sys,n,[symbol:Surf,sem:SR])}.

%%% Proper Noun %%%
pn([sem:SR,sys:Sys]) -->
    {lex(pn,[surf:Surf])},
    Surf,
    {semlex(Sys,pn,[symbol:Surf,sem:SR])}.

%%% Negation %%%
neg([sem:SR,sys:Sys]) -->
    {lex(neg,[surf:Surf])},
    Surf,
    {semlex(Sys,neg,[sem:SR])}.

%%% Wh-NP %%%
whnp_sbj([sem:SR,sys:Sys]) -->
    {lex(whnp_sbj,[surf:Surf])},
    Surf,
    {semlex(Sys,whnp_sbj,[symbol:Surf,sem:SR])}.

whnp_obj([sem:SR,sys:Sys]) -->
    {lex(whnp_obj,[surf:Surf])},
    Surf,
    {semlex(Sys,whnp_obj,[symbol:Surf,sem:SR])}.

%%% Determiner %%%
det([sem:SR,sys:Sys,num:Num]) -->
    {lex(det,[surf:Surf,num:Num])},
    Surf,
    {semlex(Sys,det,[symbol:Surf,sem:SR])}.

%%% Intransitive Verb %%%
iv([sem:SR,sys:Sys,form:Form]) -->
    {lex(iv,[surf:Surf,form:Form])},
    Surf,
    {semlex(Sys,iv,[symbol:Surf,sem:SR])}.

%%% Transitive Verb %%%
tv([sem:SR,sys:Sys,form:Form]) -->
    {lex(tv,[surf:Surf,form:Form])},
    Surf,
    {semlex(Sys,tv,[symbol:Surf,sem:SR])}.

%%% Adjectives %%%
adj([sem:SR,sys:Sys]) -->
    {lex(adj,[surf:Surf])},
    Surf,
    {semlex(Sys,adj,[symbol:Surf,sem:SR])}.

%%% Adverbs %%%
adv([sem:SR,sys:Sys]) -->
    {lex(adv,[surf:Surf])},
    Surf,
    {semlex(Sys,adv,[symbol:Surf,sem:SR])}.

%%% Connectives %%%
conn([sem:SR,sys:Sys,conj:Conj,disj:Disj]) -->
    {lex(conn,[surf:Surf],conj:Conj,disj:Disj)},
    Surf,
    {semlex(Sys,conn,[symbol:Surf,sem:SR])}.


% /* ==============================
%   Lexical Entries
% ============================== */

%%% Noun %%%

% lex(n,[surf:[nounSing],num:sing]).
% lex(n,[surf:[nounPlur],num:plur]).

lex(n,[surf:[dog],num:sing]).
lex(n,[surf:[dogs],num:plur]).

lex(n,[surf:[rabbit],num:sing]).
lex(n,[surf:[rabbits],num:plur]).

lex(n,[surf:[lion],num:sing]).
lex(n,[surf:[lions],num:plur]).

lex(n,[surf:[cat],num:sing]).
lex(n,[surf:[cats],num:plur]).

lex(n,[surf:[bear],num:sing]).
lex(n,[surf:[bears],num:plur]).

lex(n,[surf:[tiger],num:sing]).
lex(n,[surf:[tigers],num:plur]).

lex(n,[surf:[fox],num:sing]).
lex(n,[surf:[foxes],num:plur]).

lex(n,[surf:[monkey],num:sing]).
lex(n,[surf:[monkeys],num:plur]).

lex(n,[surf:[bird],num:sing]).
lex(n,[surf:[birds],num:plur]).

lex(n,[surf:[horse],num:sing]).
lex(n,[surf:[horses],num:plur]).

%%% Proper Noun %%%

% lex(pn,[surf:[pn]]).

lex(pn,[surf:[ann]]).
lex(pn,[surf:[bob]]).
lex(pn,[surf:[chris]]).
lex(pn,[surf:[daniel]]).
lex(pn,[surf:[elliot]]).
lex(pn,[surf:[fred]]).
lex(pn,[surf:[greg]]).
lex(pn,[surf:[henry]]).
lex(pn,[surf:[tom]]).
lex(pn,[surf:[john]]).

%%% Negation %%%

lex(neg,[surf:[did,not]]).

%%% WH-NP %%%

% lex(whnp_sbj,[surf:[whSubj]]).
% lex(whnp_obj,[surf:[whObj]]).

lex(whnp_sbj,[surf:[that]]).
lex(whnp_obj,[surf:[that]]).
% lex(whnp_sbj,[surf:[which]]).
% lex(whnp_obj,[surf:[which]]).

%%% Determienr %%%

% lex(det,[surf:[detSing],num:sing]).
% lex(det,[surf:[detPlur],num:plur]).

lex(det,[surf:[every],num:sing]).
lex(det,[surf:[a],num:sing]).
lex(det,[surf:[one],num:sing]).
lex(det,[surf:[all],num:plur]).
lex(det,[surf:[two],num:plur]).
lex(det,[surf:[three],num:plur]).

% lex(det,[surf:[some],num:sing]).
% lex(det,[surf:[each],num:sing]).
% lex(det,[surf:[no],num:sing]).
% lex(det,[surf:[several],num:plur]).
% lex(det,[surf:[few],num:plur]).
% lex(det,[surf:[a,few],num:plur]).
% lex(det,[surf:[at,least,three],num:plur]).
% lex(det,[surf:[less,than,three],num:plur]).
% lex(det,[surf:[more,than,three],num:plur]).
% lex(det,[surf:[at,most,three],num:plur]).

%%% Intransitive Verb %%%

% lex(iv,[surf:[ivFin],form:fin]).
% lex(iv,[surf:[ivInf],form:inf]).

lex(iv,[surf:[ran],form:fin]).
lex(iv,[surf:[walked],form:fin]).
lex(iv,[surf:[came],form:fin]).
lex(iv,[surf:[swam],form:fin]).
lex(iv,[surf:[danced],form:fin]).
lex(iv,[surf:[rushed],form:fin]).
lex(iv,[surf:[escaped],form:fin]).
lex(iv,[surf:[cried],form:fin]).
lex(iv,[surf:[slept],form:fin]).
lex(iv,[surf:[moved],form:fin]).

lex(iv,[surf:[run],form:inf]).
lex(iv,[surf:[walk],form:inf]).
lex(iv,[surf:[come],form:inf]).
lex(iv,[surf:[swim],form:inf]).
lex(iv,[surf:[dance],form:inf]).
lex(iv,[surf:[rush],form:inf]).
lex(iv,[surf:[escape],form:inf]).
lex(iv,[surf:[cry],form:inf]).
lex(iv,[surf:[sleep],form:inf]).
lex(iv,[surf:[move],form:inf]).

%%% Transitive Verb %%%

% lex(tv,[surf:[tvFin],form:fin]).
% lex(tv,[surf:[tvInf],form:inf]).

lex(tv,[surf:[kissed],form:fin]).
lex(tv,[surf:[cleaned],form:fin]).
lex(tv,[surf:[touched],form:fin]).
lex(tv,[surf:[loved],form:fin]).
lex(tv,[surf:[accepted],form:fin]).
lex(tv,[surf:[licked],form:fin]).
lex(tv,[surf:[followed],form:fin]).
lex(tv,[surf:[kicked],form:fin]).
lex(tv,[surf:[chased],form:fin]).
lex(tv,[surf:[liked],form:fin]).

lex(tv,[surf:[kiss],form:inf]).
lex(tv,[surf:[clean],form:inf]).
lex(tv,[surf:[touch],form:inf]).
lex(tv,[surf:[love],form:inf]).
lex(tv,[surf:[accept],form:inf]).
lex(tv,[surf:[lick],form:inf]).
lex(tv,[surf:[follow],form:inf]).
lex(tv,[surf:[kick],form:inf]).
lex(tv,[surf:[chase],form:inf]).
lex(tv,[surf:[like],form:inf]).

%%% Adjective %%%

% lex(adj,[surf:[adj]]).

lex(adj,[surf:[small]]).
lex(adj,[surf:[large]]).
lex(adj,[surf:[crazy]]).
lex(adj,[surf:[polite]]).
lex(adj,[surf:[wild]]).

%%% Adverb %%%

% lex(adv,[surf:[adv]]).

lex(adv,[surf:[slowly]]).
lex(adv,[surf:[quickly]]).
lex(adv,[surf:[seriously]]).
lex(adv,[surf:[suddenly]]).
lex(adv,[surf:[lazily]]).

%%% Connectives %%%

% lex(conn,[surf:[conn]]).

lex(conn,[surf:[and]],conj:yes,disj:no).
lex(conn,[surf:[or]],conj:no,disj:yes).


/* ========================================
  Semantic Composition: First-Order Logic
=========================================== */

%%% Sentence %%%

combine(fol,s:S,[np:NP,iv:IV]) :-
    S = app(NP,IV).

combine(fol,s:S,[np:NP,iv:IV,vmod:Vmod]) :-
    S = app(NP,lam(X,app(app(Vmod,IV),X))).

combine(fol,s:S,[np:NP1,tv:TV,np:NP2]) :-
    S = app(NP1,lam(X,app(NP2,lam(Y,app(app(TV,Y),X))))).

combine(fol,s:S,[np:NP,neg:Neg,iv:IV]) :-
    S = app(NP,lam(X,app(Neg,app(IV,X)))).

combine(fol,s:S,[np:NP,neg:Neg,iv:IV,vmod:Vmod]) :-
    S = app(NP,lam(X,app(Neg,app(app(Vmod,IV),X)))).

combine(fol,s:S,[np:NP1,neg:Neg,tv:TV,np:NP2]) :-
    S = app(NP1,lam(X,app(Neg,app(NP2,lam(Y,app(app(TV,Y),X)))))).

%%% Noun Phrase %%%

combine(fol,np:NP,[det:Det,n:N]) :-
    NP = app(Det,N).

combine(fol,np:NP,[det:Det,adj:ADJ,n:N]) :-
    NP = app(Det,lam(X,and(app(N,X),app(ADJ,X)))).

combine(fol,np:NP,[det:Det,n:N,sbar:Sbar]) :-
    NP = app(Det,lam(X,and(app(N,X),app(Sbar,X)))).

%%% Sbar %%%

combine(fol,sbar:Sbar,[whnp_sbj:WH,tv:TV,np:NP]) :-
    Sbar = app(WH,lam(X,app(NP,lam(Y,app(app(TV,Y),X))))).

combine(fol,sbar:Sbar,[whnp_obj:WH,np:NP,tv:TV]) :-
    Sbar = app(WH,lam(Y,app(NP,lam(X,app(app(TV,Y),X))))).

combine(fol,sbar:Sbar,[whnp_sbj:WH,iv:IV]) :-
    Sbar = app(WH,lam(X,app(IV,X))).

combine(fol,sbar:Sbar,[whnp_sbj:WH,iv:IV,vmod:Vmod]) :-
    Sbar = app(WH,app(Vmod,IV)).
    % Sbar = app(WH,lam(X,app(app(Vmod,IV),X))).

combine(fol,sbar:Sbar,[whnp_sbj:WH,neg:Neg,tv:TV,np:NP]) :-
    Sbar = app(WH,lam(X,app(Neg,app(NP,lam(Y,app(app(TV,Y),X)))))).
    % Sbar = app(WH,lam(Y,app(NP,lam(X,app(Neg,app(app(TV,Y),X)))))).

combine(fol,sbar:Sbar,[whnp_obj:WH,np:NP,neg:Neg,tv:TV]) :-
    Sbar = app(WH,lam(Y,app(NP,lam(X,app(Neg,app(app(TV,Y),X)))))).
    % Sbar = app(WH,lam(Y,app(Neg,app(NP,lam(X,app(app(TV,Y),X)))))).

combine(fol,sbar:Sbar,[whnp_sbj:WH,neg:Neg,iv:IV]) :-
    Sbar = app(WH,lam(X,app(Neg,app(IV,X)))).

combine(fol,sbar:Sbar,[whnp_sbj:WH,neg:Neg,iv:IV,vmod:Vmod]) :-
    Sbar = app(WH,lam(X,app(Neg,app(app(Vmod,IV),X)))).

%%% Verbal Modifier %%%

combine(fol,vmod:Vmod,[adv:ADV]) :-
    Vmod = lam(F,lam(X,and(app(F,X),app(ADV,X)))).
combine(fol,vmod:Vmod,[conn:Conn,iv:IV]) :-
    Vmod = lam(F,lam(X,app(app(Conn,app(F,X)),app(IV,X)))).


/* ========================================
  Semantic Composition: Variable-Free Form
=========================================== */

%%% Sentence %%%

combine(vf,s:S,[np:NP,iv:IV]) :-
    S = app(NP,IV).

combine(vf,s:S,[np:NP,iv:IV,vmod:Vmod]) :-
    S = app(NP,app(Vmod,IV)).

combine(vf,s:S,[np:NP1,tv:TV,np:NP2]) :-
    S = app(NP1,app(NP2,TV)).

combine(vf,s:S,[np:NP,neg:Neg,iv:IV]) :-
    S = app(NP,app(Neg,IV)).

combine(vf,s:S,[np:NP,neg:Neg,iv:IV,vmod:Vmod]) :-
    S = app(NP,app(Neg,app(Vmod,IV))).

combine(vf,s:S,[np:NP1,neg:Neg,tv:TV,np:NP2]) :-
    S = app(NP1,app(Neg,app(NP2,TV))).

%%% Noun Phrase %%%

combine(vf,np:NP,[det:Det,n:N]) :-
    NP = app(Det,N).

combine(vf,np:NP,[det:Det,adj:ADJ,n:N]) :-
    NP = app(Det,and(N,ADJ)).

combine(vf,np:NP,[det:Det,n:N,sbar:Sbar]) :-
    NP = app(Det,and(N,Sbar)).

%%% Sbar %%%

combine(vf,sbar:Sbar,[whnp_sbj:WH,tv:TV,np:NP]) :-
    Sbar = app(WH,app(NP,TV)).

combine(vf,sbar:Sbar,[whnp_obj:WH,np:NP,tv:TV]) :-
    Sbar = app(WH,app(NP,inv(TV))).

combine(vf,sbar:Sbar,[whnp_sbj:WH,iv:IV]) :-
    Sbar = app(WH,IV).

combine(vf,sbar:Sbar,[whnp_sbj:WH,iv:IV,vmod:Vmod]) :-
    Sbar = app(WH,app(Vmod,IV)).

combine(vf,sbar:Sbar,[whnp_sbj:WH,neg:Neg,tv:TV,np:NP]) :-
    Sbar = app(WH,app(Neg,app(NP,TV))).
    % Sbar = app(WH,app(Neg,app(NP,TV))).

combine(vf,sbar:Sbar,[whnp_obj:WH,np:NP,neg:Neg,tv:TV]) :-
    Sbar = app(WH,app(NP,app(Neg,inv(TV)))).
    % Sbar = app(WH,app(Neg,app(NP,inv(TV)))).

combine(vf,sbar:Sbar,[whnp_sbj:WH,neg:Neg,iv:IV]) :-
    Sbar = app(WH,app(Neg,IV)).

combine(vf,sbar:Sbar,[whnp_sbj:WH,neg:Neg,iv:IV,vmod:Vmod]) :-
    Sbar = app(WH,app(Neg,app(Vmod,IV))).

%%% Verbal Modifier %%%

combine(vf,vmod:Vmod,[adv:ADV]) :-
    Vmod = lam(F,and(F,ADV)).
combine(vf,vmod:Vmod,[conn:Conn,iv:IV]) :-
    Vmod = lam(F,app(app(Conn,F),IV)).


/* ===================================
  Semantic Lexicon: First-Order Logic
====================================== */

semlex(fol,n,[symbol:[Surf],sem:SR]) :-
    convertSymbol(Surf,N),
    SR = lam(X,F),
    compose(F,N,[X]).

semlex(fol,pn,[symbol:[Surf],sem:SR]) :-
    SR = lam(G,some(X,and(app(F,X),app(G,X)))),
    F = lam(X,P),
    compose(P,Surf,[X]).

% semlex(fol,pn,[symbol:[Surf],sem:SR]) :-
%     SR = lam(F,app(F,Surf)).

semlex(fol,neg,[sem:SR]) :-
    SR = lam(P,not(P)).

semlex(fol,whnp_sbj,[symbol:_,sem:SR]) :-
    SR = lam(F,F).

semlex(fol,whnp_obj,[symbol:_,sem:SR]) :-
    SR = lam(F,F).

semlex(fol,det,[symbol:[every],sem:SR]) :-
    SR = lam(F,lam(G,all(X,imp(app(F,X),app(G,X))))).

semlex(fol,det,[symbol:[all],sem:SR]) :-
    SR = lam(F,lam(G,all(X,imp(app(F,X),app(G,X))))).

semlex(fol,det,[symbol:[a],sem:SR]) :-
    SR = lam(F,lam(G,some(X,and(app(F,X),app(G,X))))).

semlex(fol,det,[symbol:[one],sem:SR]) :-
    SR = lam(F,lam(G,some(X,and(app(F,X),app(G,X))))).

semlex(fol,det,[symbol:[two],sem:SR]) :-
    SR = lam(F,lam(G,some(X,and(two(X),and(app(F,X),app(G,X)))))).

semlex(fol,det,[symbol:[three],sem:SR]) :-
    SR = lam(F,lam(G,some(X,and(three(X),and(app(F,X),app(G,X)))))).

semlex(fol,iv,[symbol:[Surf],sem:SR]) :-
    convertSymbol(Surf,V),
    SR = lam(X,F),
    compose(F,V,[X]).

semlex(fol,tv,[symbol:[Surf],sem:SR]) :-
    convertSymbol(Surf,V),
    SR = lam(Y,lam(X,F)),
    compose(F,V,[X,Y]).

semlex(fol,adj,[symbol:[Surf],sem:SR]) :-
    SR = lam(X,F),
    compose(F,Surf,[X]).

semlex(fol,adv,[symbol:[Surf],sem:SR]) :-
    SR = lam(X,F),
    compose(F,Surf,[X]).

semlex(fol,conn,[symbol:[and],sem:SR]) :-
    SR = lam(P,lam(Q,and(P,Q))).

semlex(fol,conn,[symbol:[or],sem:SR]) :-
    SR = lam(P,lam(Q,or(P,Q))).


/* ===================================
  Semantic Lexicon: Variable-Free Form
====================================== */

semlex(vf,n,[symbol:[Surf],sem:SR]) :-
    convertSymbol(Surf,SR).

semlex(vf,pn,[symbol:[Surf],sem:SR]) :-
    SR = lam(F,exist(Surf,F)).

% semlex(vf,pn,[symbol:[Surf],sem:SR]) :-
%     SR = lam(F,pred(Surf,F)).

semlex(vf,neg,[sem:SR]) :-
    SR = lam(P,not(P)).

semlex(vf,whnp_sbj,[symbol:_,sem:SR]) :-
    SR = lam(F,F).

semlex(vf,whnp_obj,[symbol:_,sem:SR]) :-
    SR = lam(F,F).

semlex(vf,det,[symbol:[every],sem:SR]) :-
    SR = lam(F,lam(G,Det)),
    compose(Det,forall,[F,G]).

semlex(vf,det,[symbol:[all],sem:SR]) :-
    SR = lam(F,lam(G,Det)),
    compose(Det,forall,[F,G]).

semlex(vf,det,[symbol:[a],sem:SR]) :-
    SR = lam(F,lam(G,Det)),
    compose(Det,exist,[F,G]).

semlex(vf,det,[symbol:[one],sem:SR]) :-
    SR = lam(F,lam(G,Det)),
    compose(Det,exist,[F,G]).

semlex(vf,det,[symbol:[two],sem:SR]) :-
    SR = lam(F,lam(G,Det)),
    compose(Det,two,[F,G]).

semlex(vf,det,[symbol:[three],sem:SR]) :-
    SR = lam(F,lam(G,Det)),
    compose(Det,three,[F,G]).

% semlex(vf,det,[symbol:[Surf],sem:SR]) :-
%     SR = lam(F,lam(G,Det)),
%     compose(Det,Surf,[F,G]).

% semlex(vf,det,[symbol:[Surf],sem:SR]) :-
%     SR = lam(F,lam(G,Det)),
%     atom_concat(#,Surf,Sym),
%     write(Sym),
%     compose(Det,Sym,[F,G]).
% semlex(vf,det,[symbol:[Det],sem:SR]) :-
%     SR = lam(F,lam(G,[Det,G,F])).
% semlex(vf,det,[symbol:[Surf],sem:SR]) :-
%     SR = lam(F,lam(G,Det)),
%     compose(Det,Surf,[F,G]).

semlex(vf,iv,[symbol:[Surf],sem:SR]) :-
    convertSymbol(Surf,SR).

semlex(vf,tv,[symbol:[Surf],sem:SR]) :-
    convertSymbol(Surf,SR).

semlex(vf,adj,[symbol:[Surf],sem:Surf]).

semlex(vf,adv,[symbol:[Surf],sem:Surf]).

semlex(vf,conn,[symbol:[and],sem:SR]) :-
    SR = lam(P,lam(Q,and(P,Q))).

semlex(vf,conn,[symbol:[or],sem:SR]) :-
    SR = lam(P,lam(Q,or(P,Q))).


/* ==========================================================
  Predicates to change variable-free forms to Polish notation
============================================================= */

convertVF(pred(F,G),R) :- !,
    convertVF(F,F1),
    convertVF(G,G1),
    string_concat('PRED ',F1,R1),
    string_concat(R1,' ',R2),
    string_concat(R2,G1,R).

convertVF(and(F,G),R) :- !,
    convertVF(F,F1),
    convertVF(G,G1),
    string_concat('AND ',F1,R1),
    string_concat(R1,' ',R2),
    string_concat(R2,G1,R).

convertVF(or(F,G),R) :- !,
    convertVF(F,F1),
    convertVF(G,G1),
    string_concat('OR ',F1,R1),
    string_concat(R1,' ',R2),
    string_concat(R2,G1,R).

convertVF(exist(F,G),R) :- !,
    convertVF(F,F1),
    convertVF(G,G1),
    string_concat('EXIST ',F1,R1),
    string_concat(R1,' ',R2),
    string_concat(R2,G1,R).

convertVF(forall(F,G),R) :- !,
    convertVF(F,F1),
    convertVF(G,G1),
    string_concat('ALL ',F1,R1),
    string_concat(R1,' ',R2),
    string_concat(R2,G1,R).

convertVF(two(F,G),R) :- !,
    convertVF(F,F1),
    convertVF(G,G1),
    string_concat('TWO ',F1,R1),
    string_concat(R1,' ',R2),
    string_concat(R2,G1,R).

convertVF(three(F,G),R) :- !,
    convertVF(F,F1),
    convertVF(G,G1),
    string_concat('THREE ',F1,R1),
    string_concat(R1,' ',R2),
    string_concat(R2,G1,R).

convertVF(not(F),R) :- !,
    convertVF(F,F1),
    string_concat('NOT ',F1,R).

convertVF(inv(F),R) :- !,
    convertVF(F,F1),
    string_concat('INV ',F1,R).

convertVF(F,R) :-
    string_upper(F,R).


/* ==========================================================
  Assign polarities to each predicate in variable-free form
============================================================= */

reversePol(pos,neg).
reversePol(neg,pos).

assignPolVF(pred(F,G),Pol,Res) :- !,
    assignPolVF(F,Pol,R1),
    assignPolVF(G,Pol,R2),
    append(R1,R2,Res).

assignPolVF(and(F,G),Pol,Res) :- !,
    assignPolVF(F,Pol,R1),
    assignPolVF(G,Pol,R2),
    append(R1,R2,Res).

assignPolVF(or(F,G),Pol,Res) :- !,
    assignPolVF(F,Pol,R1),
    assignPolVF(G,Pol,R2),
    append(R1,R2,Res).

assignPolVF(exist(F,G),Pol,Res) :- !,
    assignPolVF(F,Pol,R1),
    assignPolVF(G,Pol,R2),
    append(R1,R2,Res).

assignPolVF(forall(F,G),Pol,Res) :- !,
    reversePol(Pol,Rev),
    assignPolVF(F,Rev,R1),
    assignPolVF(G,Pol,R2),
    append(R1,R2,Res).

assignPolVF(two(F,G),Pol,Res) :- !,
    assignPolVF(F,Pol,R1),
    assignPolVF(G,Pol,R2),
    append(R1,R2,Res).

assignPolVF(three(F,G),Pol,Res) :- !,
    assignPolVF(F,Pol,R1),
    assignPolVF(G,Pol,R2),
    append(R1,R2,Res).

assignPolVF(not(F),Pol,Res) :- !,
    reversePol(Pol,Rev),
    assignPolVF(F,Rev,Res).

assignPolVF(inv(F),Pol,Res) :- !,
    assignPolVF(F,Pol,Res).

assignPolVF(F,Pol,R) :-
    atom(F),
    R = [[F,Pol]].


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

compose(Term,Symbol,ArgList):-
    Term =.. [Symbol|ArgList].

labelJudge(yes,yes,yes).
labelJudge(yes,no,yes).
labelJudge(no,yes,yes).
labelJudge(no,no,no).

nicePrint(SR) :-
    \+ \+ (numbervars(SR,0,_), print(SR)).

stringsToAtoms([],[]).
stringsToAtoms([H|T],Res) :-
    string_lower(H,X),
    string_to_atom(X,H1),
    stringsToAtoms(T,T1),
    Res = [H1|T1].

strToAtoms(Str,Atoms) :-
   split_string(Str," ","",StrList),
   stringsToAtoms(StrList,Atoms).


/* ==============================
   Main Predicates
============================== */

% Generate a plain sentence with depth N, selector K, system Sys (fol,vf)
plain(N,K,Sys) :-
   s([sem:_,sys:Sys,depth:N,sel:K,phen:_],Sentence,[]),
   % selector(K),
   yield(Sentence),nl,
   fail.

% Parse a sentence to FOL
sent2fol(D,S) :-
   s([sem:SR,sys:fol,depth:D,sel:1,_],S,[]),
   betaConvert(SR,FOL),
   fol2nltk(FOL,user).

% Parse a sentence string to FOL
sentstr2fol(D,Str) :-
   strToAtoms(Str,S),
   s([sem:SR,sys:fol,depth:D,sel:1,_],S,[]),
   betaConvert(SR,FOL),
   fol2nltk(FOL,user).

% Extract phenomena tags from a sentence
sent2phen(D,S) :-
   s([sem:_,sys:fol,depth:D,sel:1,phen:[sbj_quant:SBJQ,obj_quant:OBJQ,neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]],S,[]),
   format('sbj_quant:~w;obj_quant:~w;negation:~w;adjective:~w;peripheral:~w;center:~w;adverb:~w;conjunction:~w;disjunction:~w',[SBJQ,OBJQ,Neg,Adj,Per,Cen,Adv,Conj,Disj]).

% Parse a sentence to FOL with phenomena tags
sent2folphen(D,S) :-
   s([sem:SR,sys:fol,depth:D,sel:1,phen:[sbj_quant:SBJQ,obj_quant:OBJQ,neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]],S,[]),
   betaConvert(SR,FOL),
   fol2nltk(FOL,user),
   write('@'),
   format('sbj_quant:~w;obj_quant:~w;negation:~w;adjective:~w;peripheral:~w;center:~w;adverb:~w;conjunction:~w;disjunction:~w',[SBJQ,OBJQ,Neg,Adj,Per,Cen,Adv,Conj,Disj]).

% Parse a sentence to VF with brackets
sent2vfb(D,S) :-
   s([sem:SR,sys:vf,depth:D,sel:1,_],S,[]),
   betaConvert(SR,VF),
   write(VF).

% Parse a sentence to VF in Polish notation
sent2vf(D,S) :-
   s([sem:SR,sys:vf,depth:D,sel:1,_],S,[]),
   betaConvert(SR,SR1),
   convertVF(SR1,VF),
   write(VF).

% Parse a sentence to VF and output polarities
sent2vfpol(D,S) :-
   s([sem:SR,sys:vf,depth:D,sel:1,_],S,[]),
   betaConvert(SR,SR1),
   assignPolVF(SR1,pos,Pol),
   write(Pol).

% Calculate polarities for VF format
calcPolVF(VF) :-
   assignPolVF(VF,pos,Pol),
   write(Pol).

% Show the parse result with three forms
semparse(D,S) :-
   s([sem:SR1,sys:fol,depth:D,sel:1,phen:[sbj_quant:SBJQ,obj_quant:OBJQ,neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]],S,[]),
   betaConvert(SR1,FOL),
   fol2nltk(FOL,user),
   s([sem:SR2,sys:vf,depth:D,sel:1,_],S,[]),
   betaConvert(SR2,VFb),
   convertVF(VFb,VF),
   write(VF),
   format('sbj_quant:~w;obj_quant:~w;negation:~w;adjective:~w;peripheral:~w;center:~w;adverb:~w;conjunction:~w;disjunction:~w',[SBJQ,OBJQ,Neg,Adj,Per,Cen,Adv,Conj,Disj]).

% Show the parse result with three forms
display(D,Str) :-
   strToAtoms(Str,S),
   s([sem:FOL,sys:fol,depth:D,sel:1,phen:[sbj_quant:SBJQ,obj_quant:OBJQ,neg:Neg,adj:Adj,per_emb:Per,cen_emb:Cen,adv:Adv,conj:Conj,disj:Disj]],S,[]),
   betaConvert(FOL,NF1),
   write(Str),nl,
   write('FOL:'),
   fol2nltk(NF1,user),
   s([sem:VF,sys:vf,depth:D,sel:1,_],S,[]),
   betaConvert(VF,NF2),
   convertVF(NF2,Res),
   assignPolVF(NF2,pos,VFpol),
   format('~nsbj_quant:~w;~nobj_quant:~w;~nnegation:~w;~nadjective:~w;~nperipheral:~w;~ncenter:~w;~nadverb:~w;~nconjunction:~w;~ndisjunction:~w;~nVFb:~w~nVF:~w~nVFpol:~w',[SBJQ,OBJQ,Neg,Adj,Per,Cen,Adv,Conj,Disj,NF2,Res,VFpol]).

test :-
    write('0'),nl,
    display(0,"ann moved"),nl,
    write('1'),nl,
    display(0,"three tigers moved"),nl,
    write('2'),nl,
    display(0,"all tigers cried or swam"),nl,
    write('3'),nl,
    display(0,"three tigers did not cry slowly"),nl,
    write('4'),nl,
    display(0,"ann did not chase a small dog"),nl,
    write('5'),nl,
    display(0,"ann chased a small dog"),nl,
    write('6'),nl,
    display(1,"every dog that kissed a small lion walked"),nl,
    write('7'),nl,
    display(1,"every dog that all lions kissed walked"),nl,
    write('8'),nl,
    display(1,"one dog that all lions did not kiss did not walk"),nl,
    write('9'),nl,
    display(1,"bob did not touch a cat that every dog did not kiss").
