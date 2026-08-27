#!/bin/bash
# After downloading all /tmp/phase6_highdim_extra_dense_thorough_shardXX.zip files, unzip into results/phase6_highdim_extra_dense_thorough/
set -e
mkdir -p results/phase6_highdim_extra_dense_thorough
echo "Unzip all downloaded zips into results/phase6_highdim_extra_dense_thorough/"
# Example: if zips are in ~/Downloads/
# for z in ~/Downloads/phase6_highdim_extra_dense_thorough_shard*.zip; do unzip -o "$z" -d results/phase6_highdim_extra_dense_thorough/; done
# or if already in notebooks/phase6_highdim_extra_dense_thorough/
for z in notebooks/phase6_highdim_extra_dense_thorough/*.zip; do [ -f "$z" ] && unzip -o "$z" -d results/phase6_highdim_extra_dense_thorough/ || true; done
ls -lh results/phase6_highdim_extra_dense_thorough/ | head
echo "Then aggregate: python experiments/phase6_highdim_pointlaw_tournament.py --mode aggregate --input-dir results/phase6_highdim_extra_dense_thorough"
