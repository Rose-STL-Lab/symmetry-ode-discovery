# run experiments for seed 0-49
for i in {0..49}; do
    echo "Running seed $i"
    python main_wsindy.py --seed "$i" --config growth/noise05_wsindy.cfg
done