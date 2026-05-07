rm *.jsonl
rm */*.pdf
rm *.txt
rm *.json

cd ../..

python -m ae.map_rcd.map_transformer --simgpu
python -m ae.map_rcd.map_transformer --simgpu --init

cd ae/map_rcd

python plot_best_mapping.py \
  best_mapping_transformer_A100_sim_decode.jsonl \
  -o GPU_decode

python plot_best_mapping.py \
  best_mapping_transformer_A100_prefill.jsonl \
  -o GPU_prefill