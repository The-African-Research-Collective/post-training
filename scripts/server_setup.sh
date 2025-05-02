#download and install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh

# create a UV environment
uv && uv venv

# activate the environment
source .venv/bin/activate

# install the required packages
uv pip install wandb
uv pip install deepspeed==0.15.4
uv pip install flash-attn --no-build-isolation
uv pip install trl[vllm]@git+https://github.com/huggingface/trl.git@1bca49515ecd5b85d16e68c42c76670e252e19f1


