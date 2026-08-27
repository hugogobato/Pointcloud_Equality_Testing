#!/bin/bash
# After downloading all /tmp/phase6_highdim_extra_dense_pilot_shardXX.zip files, unzip into results/phase6_highdim_extra_dense_pilot/
set -e
mkdir -p results/phase6_highdim_extra_dense_pilot
echo "Unzip all downloaded zips into results/phase6_highdim_extra_dense_pilot/"
# Example: if zips are in ~/Downloads/
# for z in ~/Downloads/phase6_highdim_extra_dense_pilot_shard*.zip; do unzip -o "$z" -d results/phase6_highdim_extra_dense_pilot/; done
# or if already in notebooks/phase6_highdim_extra_dense_pilot/
for z in notebooks/phase6_highdim_extra_dense_pilot/*.zip; do [ -f "$z" ] && unzip -o "$z" -d results/phase6_highdim_extra_dense_pilot/ || true; done
ls -lh results/phase6_highdim_extra_dense_pilot/ | head
echo "Then aggregate: python experiments/phase6_highdim_pointlaw_tournament.py --mode aggregate --input-dir results/phase6_highdim_extra_dense_pilot"
