for i in {0..49}; do
    echo "Running seed $i"
    python main.py --seed "$i" --config lv/noise99_eq_rsymreg.cfg
done