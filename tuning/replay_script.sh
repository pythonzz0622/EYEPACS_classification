#!bin/sh
: <<'END'
python train_v2.py --optimizer "Adam" --lr 0.001 --addtag "set:optim"
python train_v2.py --optimizer "Adam" --lr 0.001 --addtag "set:optim"
python train_v2.py --optimizer "Adam" --lr 0.001 --addtag "set:optim"

python train_v2.py --optimizer "SGD" --lr 0.1 --addtag "set:optim"
python train_v2.py --optimizer "SGD" --lr 0.1 --addtag "set:optim"
python train_v2.py --optimizer "SGD" --lr 0.1 --addtag "set:optim"

python train_v2.py --optimizer "SGD" --lr 0.001 --addtag "set:optim"
python train_v2.py --optimizer "SGD" --lr 0.001 --addtag "set:optim"
python train_v2.py --optimizer "SGD" --lr 0.001 --addtag "set:optim"

python train_v2.py --optimizer "Adamax" --lr 0.01 --addtag "set:optim"
python train_v2.py --optimizer "Adamax" --lr 0.01 --addtag "set:optim"
python train_v2.py --optimizer "Adamax" --lr 0.01 --addtag "set:optim"

python train_v2.py --optimizer "Adamax" --lr 0.01 --addtag "set:optim"
python train_v2.py --optimizer "Adamax" --lr 0.01 --addtag "set:optim"
python train_v2.py --optimizer "Adamax" --lr 0.01 --addtag "set:optim"
END
python train_v2.py --optimizer "SGD" --lr 0.1 --addtag "set:optim"
python train_v2.py --optimizer "SGD" --lr 0.1 --addtag "set:optim"
python train_v2.py --optimizer "SGD" --lr 0.1 --addtag "set:optim"



