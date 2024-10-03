#!/bin/bash

echo "First arg to choose which file you want as input, if it isn't kept it will run on test2.in"
echo

if [ $# -eq 0 ]; then
    python3 main.py < test/test2.in
else
    python3 main.py < $1
fi
