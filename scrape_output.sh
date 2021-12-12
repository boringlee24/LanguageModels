#!/bin/bash

BASEDIR=/home/gridsan/$(whoami)/languagemodels/bert-mlm
OUTPUT_FILE=/state/partition1/user/jpmcd/energy_consumed.csv
# OUTPUT_FILE=/state/partition1/user/jpmcd/energy_consumed_inference.csv
rm $OUTPUT_FILE
# for job in 11147303 11148944 11148955 11148966 ; do
job="1"
for d in $BASEDIR/${job}*W ; do
    echo $d
    for f in $d/dcgm* ; do
        awk -v f=$f '/Energy Consumed/ {print f","$6} ' $f >> $OUTPUT_FILE
        # awk -v f=$f '/Energy Consumed/ {print f","$6} ' $f
    done
done
# done
mv $OUTPUT_FILE $BASEDIR/../
