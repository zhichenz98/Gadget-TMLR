#!/usr/bin/env bash

nohup python src/main.py --data airport --method VANILLA --aug base --cuda 0 > output/airport/vanilla-base.out &
nohup python src/main.py --data citation --method VANILLA --aug base --cuda 0 > output/citation/vanilla-base.out &

nohup python src/main.py --data airport --method VANILLA --aug gadget --cuda 0 > output/airport/vanilla-gadget.out &
nohup python src/main.py --data citation --method VANILLA --aug gadget --cuda 0 > output/citation/vanilla-gadget.out &
