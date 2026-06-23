#!/bin/bash
rm -f *.csv
rm -f *.pdf

cd ../../..

export PATH=/home/zyzhao/.conda/envs/llmcompass_ae/bin:$PATH
export PYTHONUNBUFFERED=1

echo "=== Using Python: $(which python) ==="

# python -u -m ae.figure5.ijkl.test_transformer --simgpu --roofline
#python -m ae.figure5.ijkl.test_transformer --simtpu --roofline
#python -m ae.figure5.ijkl.test_transformer --simgpu --init --roofline
#python -m ae.figure5.ijkl.test_transformer --simtpu --init --roofline

#python -m ae.figure5.ijkl.test_transformer --simgpu
# python -m ae.figure5.ijkl.test_llama --simgpu --llama
# python -m ae.figure5.ijkl.test_transformer --simgpu --opt
#python -m ae.figure5.ijkl.test_transformer --simtpu
# echo "=== Step 1: CIM Prefill (Init) ==="
# python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 8 --Nbank 24 --core_count 16
# echo "=== Step 1 done ==="


# echo "=== Step 1: CIM Prefill (Init) ==="
# python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 64 --Nbank 24 --core_count 16
# echo "=== Step 1 done ==="


echo "=== Step 1: CIM Prefill (Init) ==="
python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 16 --Nbank 24 --core_count 16
python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 24 --Nbank 24 --core_count 16
python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 32 --Nbank 24 --core_count 16
python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 40 --Nbank 24 --core_count 16
python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 48 --Nbank 24 --core_count 16
python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 56 --Nbank 24 --core_count 16
python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 64 --Nbank 24 --core_count 16
python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 72 --Nbank 24 --core_count 16
python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 80 --Nbank 24 --core_count 16
python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 88 --Nbank 24 --core_count 16
python -u -m ae.figure5.ijkl.test_transformer --simcim --init --opt --array_height 64 --array_width 96 --Nbank 24 --core_count 16
echo "=== Step 1 done ==="

# echo "=== Step 2: CIM Decode (Auto-regression) ==="
# python -u -m ae.figure5.ijkl.test_transformer --simcim --opt --array_height 64 --array_width 96 --Nbank 24 --core_count 16
# echo "=== Step 2 done ==="




#python -m ae.figure5.ijkl.test_transformer --simgpu --init
# python -m ae.figure5.ijkl.test_llama --simgpu --init --llama
# python -m ae.figure5.ijkl.test_transformer --simgpu --init --opt
#python -m ae.figure5.ijkl.test_transformer --simtpu --init

cd ae/figure5/ijkl
#python plot_transformer.py