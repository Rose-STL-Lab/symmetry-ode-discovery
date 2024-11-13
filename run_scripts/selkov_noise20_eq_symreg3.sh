# run experiments for seed 0-49
for i in {0..49}; do
    echo "Running seed $i"
    python main.py --seed "$i" --config selkov/noise20_eq_symreg3.cfg
done