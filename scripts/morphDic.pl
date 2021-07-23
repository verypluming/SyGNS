/*****************************************************
 A Dictionary for Morphological Conversion
******************************************************/

:- module(morphDic,[convertSymbol/2]).

%%% Singular and Plural Nouns %%%

convertSymbol(dog,dog).
convertSymbol(dogs,dog).
convertSymbol(rabbit,rabbit).
convertSymbol(rabbits,rabbit).
convertSymbol(lion,lion).
convertSymbol(lions,lion).
convertSymbol(cat,cat).
convertSymbol(cats,cat).
convertSymbol(bear,bear).
convertSymbol(bears,bear).
convertSymbol(tiger,tiger).
convertSymbol(tigers,tiger).
convertSymbol(fox,fox).
convertSymbol(foxes,fox).
convertSymbol(monkey,monkey).
convertSymbol(monkeys,monkey).
convertSymbol(bird,bird).
convertSymbol(birds,bird).
convertSymbol(horse,horse).
convertSymbol(horses,horse).

%%% Finite and Infinitee Verbs %%%

convertSymbol(run,run).
convertSymbol(ran,run).
convertSymbol(walk,walk).
convertSymbol(walked,walk).
convertSymbol(come,come).
convertSymbol(came,come).
convertSymbol(swim,swim).
convertSymbol(swam,swim).
convertSymbol(dance,dance).
convertSymbol(danced,dance).
convertSymbol(rush,rush).
convertSymbol(rushed,rush).
convertSymbol(escape,escape).
convertSymbol(escaped,escape).
convertSymbol(cry,cry).
convertSymbol(cried,cry).
convertSymbol(sleep,sleep).
convertSymbol(slept,sleep).
convertSymbol(move,move).
convertSymbol(moved,move).

convertSymbol(kiss,kiss).
convertSymbol(kissed,kiss).
convertSymbol(clean,clean).
convertSymbol(cleaned,clean).
convertSymbol(touch,touch).
convertSymbol(touched,touch).
convertSymbol(love,love).
convertSymbol(loved,love).
convertSymbol(accept,accept).
convertSymbol(accepted,accept).
convertSymbol(lick,lick).
convertSymbol(licked,lick).
convertSymbol(follow,follow).
convertSymbol(followed,follow).
convertSymbol(kick,kick).
convertSymbol(kicked,kick).
convertSymbol(chase,chase).
convertSymbol(chased,chase).
convertSymbol(like,like).
convertSymbol(liked,like).
