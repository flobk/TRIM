#!/bin/bash

python calculate_metric.py  --model facebook/opt-6.7b \
                            --cache_dir YOUR/MODEL/CACHE \
                            --ww_metric alpha_peak \
                            --ww_metric_cache ./metrics \