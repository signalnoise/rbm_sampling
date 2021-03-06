#!/bin/sh

source use-conda
source activate rbm

for i in {10..5000..10}
do
    jq  '.checkpoint = "/scratch/data/rbm/rbm_train/T_2.27_epoch_5000_batch_100_hidden_8_ising_8_k_1/text/trained_rbm.pytorch.'${i}'"' jsonin > tmp.$$.json && mv tmp.$$.json jsonin

    python ~/rbmgit/sample_many_rbm/rbm_sample.py --json jsonin --epochs ${i} >& out.txt
    echo Running state ${i}

done
