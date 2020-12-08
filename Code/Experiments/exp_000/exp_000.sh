cd /home/ubuntu/atmacup/08/Code/Experiments/exp_000/

rm -R weight
mkdir weight

rm log.txt
touch log.txt
python main.py --config=config.yml > log.txt
