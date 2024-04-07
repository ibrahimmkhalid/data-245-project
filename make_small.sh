#!/bin/bash
head -n 57 probe_known_attacks.arff | tail -n 51 | sed -E "s/@ATTRIBUTE '([a-zA-Z_]*)'.*/\1,/" | tr -d '\n'  | sed -E "s/class,/class\n/" > probe_known_attacks_small.csv
cat probe_known_attacks.csv | grep -C 5 "attack" --no-group-separator >> probe_known_attacks_small.csv
