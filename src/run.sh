#!/bin/bash
cat bg_list.txt | while read line
do
  nohup python3 bigram_gen_bsb.py TASA $line > bsb_$line.out &
done
