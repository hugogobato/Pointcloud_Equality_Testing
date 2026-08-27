#!/bin/bash
# Run after downloading zips: consolidates 20 zips into results/phase6_highdim_shards deduplicated
set -e
python3 /tmp/consolidate_phase6_shards.py
echo "Then aggregate (partial, will show missing warning until remaining done):"
echo "python experiments/phase6_highdim_pointlaw_tournament.py --mode aggregate --input-dir results/phase6_highdim_shards"
