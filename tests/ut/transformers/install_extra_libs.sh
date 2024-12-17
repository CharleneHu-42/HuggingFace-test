pip install autoawq auto-gptq aqlm flash-attn lomo-optim torchao
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install natten==0.17.1+torch230cu121 -f https://shi-labs.com/natten/wheels/
git clone https://github.com/NetEase-FuXi/EETQ.git && cd EETQ/
git submodule update --init --recursive
pip install .
cd ..
git clone https://github.com/NVIDIA/apex && cd apex
if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./