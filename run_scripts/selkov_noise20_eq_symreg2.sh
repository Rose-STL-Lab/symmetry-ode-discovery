# run experiments for seed 0-49
for i in {0..49}; do
    python main.py --seed "$i" --config selkov/noise20_eq_symreg2.cfg
done