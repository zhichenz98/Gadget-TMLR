#!/usr/bin/env bash

nohup python src/main.py --data airport --method CORAL --aug base --cuda 0 > output/airport/coral-base.out &
nohup python src/main.py --data citation --method CORAL --aug base --cuda 0 > output/citation/coral-base.out &

nohup python src/main.py --data airport --method CORAL --aug gadget --cuda 0 > output/airport/coral-gadget.out &
nohup python src/main.py --data citation --method CORAL --aug gadget --cuda 0 > output/citation/coral-gadget.out &
