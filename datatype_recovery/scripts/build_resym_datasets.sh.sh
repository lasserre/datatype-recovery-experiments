# train funcs only (formerly resym_train_5hops)
time dragon build --from-exp resym2_train_5hops 5 ~/exp_imports/resym_train.exp/ --batchsize 1000000 \
    --func-list ~/dev/dtype-recovery-notebooks/misc_notebooks/resym_train_funcs.csv

# all funcs (formerly resym_train_5hops_full)
# time dragon build --from-exp resym_train_5hops_full 5 ~/exp_imports/resym_train.exp/ --batchsize 1000000
