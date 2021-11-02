#!/bin/bash
conda_env_name=$1
eval "$(conda shell.bash hook)"
conda activate $conda_env_name

# For faster inference you need to build nms, this is needed during evaluating.
(cd ext && python build.py build_ext develop)

# For ThunderNet, extra compilation is needed before running.
(cd od/modeling/head/ThunderNet && python setup.py build develop)

# For CenterNet, extra compilation is needed before running.
[ -d od/modeling/head/centernet/DCNv2 ] && (cd od/modeling/head/centernet/DCNv2 && ./make.sh)

# For FCOS , extra compilation is needed before running.
(cd od/modeling/head/fcos && python setup.py build develop --no-deps)
