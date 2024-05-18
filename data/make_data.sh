#!/bin/bash

if [ -z "$1" ]; then
  context=2
else
  context=$1
fi

head -n 57 probe_known_attacks.arff | tail -n 51 | sed -E "s/@ATTRIBUTE '([a-zA-Z_]*)'.*/\1,/" | tr -d '\n'  | sed -E "s/class,/class\n/" > header.csv

cat header.csv > probe_known_attacks.csv
cat header.csv > probe_known_attacks_small.csv
cat header.csv > probe_similar_attacks.csv
cat header.csv > probe_similar_attacks_small.csv
cat header.csv > probe_new_attacks.csv
cat header.csv > probe_new_attacks_small.csv

tail -n +59 probe_known_attacks.arff >> probe_known_attacks.csv
tail -n +59 probe_similar_attacks.arff >> probe_similar_attacks.csv
tail -n +59 probe_new_attacks.arff >> probe_new_attacks.csv

cat probe_known_attacks.csv | grep -C $context "attack" --no-group-separator >> probe_known_attacks_small.csv
cat probe_similar_attacks.csv | grep -C $context "attack" --no-group-separator >> probe_similar_attacks_small.csv
cat probe_new_attacks.csv | grep -C $context "attack" --no-group-separator >> probe_new_attacks_small.csv


rm probe_known_attacks.csv
rm probe_similar_attacks.csv
rm probe_new_attacks.csv
rm header.csv
