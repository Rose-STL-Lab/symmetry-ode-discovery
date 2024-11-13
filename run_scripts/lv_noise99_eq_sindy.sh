# run experiments for seed 0-49
for i in {0..49}; do
    python main.py --seed "$i" --config lv/noise99_eq_sindy_2.cfg
done