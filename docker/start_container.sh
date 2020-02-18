docker run --runtime=nvidia --rm -d -it --ipc=host \
--name ilr \
-v ~/repo/code:/root/code \
-v ~/data/sen12ms:/root/data \
-v ~/data/dfc2020:/root/val \
-v ~/logs:/root/logs \
lukasliebel/dfc_baseline
