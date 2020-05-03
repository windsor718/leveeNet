docker run --gpus all -p 0.0.0.0:9088:9088 \
       -it --name leveenet_train \
       -v ~/work/leveeNet:/opt/analysis/leveeNet latest /bin/bash
