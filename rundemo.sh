#!/usr/bin/env bash

python demo/test_matting.py configs/mattors/fba/fba_comp1k.py work_dirs/dim/dim.pth tests/data/merged/GT05.jpg tests/data/trimap/GT05.png tests/data/pred/GT05.png 