#!/bin/bash
# Measured runtime of the face-analysis stream (RTX 4090 + i9-13900K).
#   sparse: every 10th frame (25 fps reference) on all 20 videos, end-to-end incl. embeddings and fusion
#   dense:  every native frame on all 20 videos (decode + detection; embedding cost from measured per-face time)
cd /home/mrt/Projects/PVS
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-1}
for v in highjump javelin longjump polevault; do for i in 1 2 3 4 5; do
    V=${v}_video$i
    ./run_with_cuda.sh python revision/runtime/bench_detect.py --video $V --stride 10 2>/dev/null | tail -1
    /home/mrt/Projects/TransFace/venv/bin/python revision/runtime/bench_transface.py --tag ${V}_s10 2>/dev/null | tail -1
    ./run_with_cuda.sh python revision/runtime/bench_arcface.py --tag ${V}_s10 2>/dev/null | tail -1
done; done
python revision/runtime/bench_fusion.py --stride 10
for v in highjump javelin longjump polevault; do for i in 1 2 3 4 5; do
    ./run_with_cuda.sh python revision/runtime/bench_detect.py --video ${v}_video$i --stride 0 --no-save 2>/dev/null | tail -1  # every native frame
done; done
