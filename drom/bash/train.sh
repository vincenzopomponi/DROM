#!/bin/bash

python drom/scripts/train_drom.py --config drom/configs/low_dim/StackReal.json

sleep 10

python drom/scripts/train_drom.py --config drom/configs/low_dim/StackNewReal.json

sleep 10

python drom/scripts/train_drom.py --config drom/configs/low_dim/StackCleanupNewReal.json

sleep 10

python drom/scripts/train_drom.py --config drom/configs/low_dim/StackCleanupNewReal1.json

sleep 10

python drom/scripts/train_drom.py --config drom/configs/low_dim/StackCleanupNewReal2.json

sleep 10

python drom/scripts/train_drom.py --config drom/configs/low_dim/StackCleanupNewReal3.json