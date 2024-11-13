# run experiments for seed 0-49
for i in {0..49}; do
    python main_wsindy.py --seed "$i" --config selkov/noise20_eq_wsindy.cfg
done