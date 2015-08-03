# machine-learning

### `playing_around1.py`

Used neurolab to play around with my [chin-up](http://github.com/ckcollab/chin-up) stats. Felt kind of wonky, 
hard to figure out some things.

### `playing_around2.py`

Used neurolab again to try and make a XOR neural net. Again felt wonky, not sure I was doing things right

### `playing_around3.py`

I was pretty daunted to try sklearn because the docs seem very complicated and the code was really short. What is
this `fit()` method?! How do I set it up and define layers/etc/etc, that all seemed missing. Making this XOR neural
net was very very easy.

Sklearn figures out most of that for you! It's way more beautiful than I initially understood. Very fun to work with!

### `playing_around4.py`

Used sklearn again to mess around with my stats, was a blast. I didn't have to mess with much to immediately start
seeing results, which is the kind of tool I love using.

Example output:
```

==========================================================================
Given the previous day's stats, what can we predict tomorrows stats to be?
==========================================================================

Predict stats for the day after [10, 10, 10, 10, 10, 10]
   Happiness  Motivation  Flexibility  Strength  Endurance  Relationships
0   8.522264    8.576007     7.535281  8.187146   8.151479       8.962717

Predict stats for the day after [1, 1, 1, 1, 1, 1]
   Happiness  Motivation  Flexibility  Strength  Endurance  Relationships
0   4.073277    4.718392     2.304909  3.197984   3.005468       3.526293
```
