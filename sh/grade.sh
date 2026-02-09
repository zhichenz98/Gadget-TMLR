#!/usr/bin/env bash

nohup python src/main.py --data airport --method GRADE --aug base --cuda 0 > output/airport/grade-base.out &
nohup python src/main.py --data citation --method GRADE --aug base --cuda 0 > output/citation/grade-base.out &

nohup python src/main.py --data airport --method GRADE --aug gadget --cuda 0 > output/airport/grade-gadget.out &
nohup python src/main.py --data citation --method GRADE --aug gadget --cuda 0 > output/citation/grade-gadget.out &
