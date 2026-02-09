!/usr/bin/env bash

nohup python src/main.py --data airport --method AdaGCN --aug base --cuda 0 > output/airport/adagcn-base.out &
nohup python src/main.py --data citation --method AdaGCN --aug base --cuda 0 > output/citation/adagcn-base.out &

nohup python src/main.py --data airport --method AdaGCN --aug gadget --cuda 0 > output/airport/adagcn-gadget.out &
nohup python src/main.py --data citation --method AdaGCN --aug gadget --cuda 0 > output/citation/adagcn-gadget.out &

