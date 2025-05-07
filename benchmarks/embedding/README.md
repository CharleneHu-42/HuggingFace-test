# Huggingface Embeddings Benchmark Scripts

This directory contains benchmarks for Huggingface embedding frameworks with synchronous and asynchronous tests.

## Setup

1. Clone the repository

```bash
git clone https://github.com/intel-sandbox/HuggingFace.git
cd HuggingFace/benchmarks/embedding
```

2. Install Python requirements

```bash
# With conda:
conda env create -f environment.yml
# With pip:
pip install -r requirements.txt
```

3. Build a docker environment for the project

4. Launch the docker environment

- If supported by `docker.py` (see [Support Table](#support-table))

```bash
python docker.py --model_name <model-name> --docker_image <docker-tag> [--other flags]
```

- Otherwise, run the docker environment manually according the project specifications. Please ensure that batch size of at least 512 is supported by the backend. The benchmark will sweep batch sizes up to 512.

```bash
# For TEI on HPU
docker run \
    --privileged \
    --rm \
    -p 8081:80 \ # PORT
    -v ~/.cache/huggingface/hub/:/data \
    --ipc=host \
    -e HTTP_PROXY=$HTTP_PROXY \
    -e HTTPS_PROXY=$HTTPS_PROXY \
    -e MAX_WARMUP_SEQUENCE_LENGTH=512 \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --runtime=habana \
    --cap-add=sys_nice \
    tei-gaudi \
    --model-id sentence-transformers/all-mpnet-base-v2 \
    --pooling cls \
    --max-client-batch-size 512 \
```

> [!NOTE]
> Keep track of the exposed `PORT` and `MODEL NAME`. They will be used during the benchmark.

## Benchmarking

Two benchmark scripts are provided:

1. `test_sync.py` runs tests that sends requests to the specified backend synchronously, waiting for a response before sending the next request. Each test runs with a different batch size. By default, the following batch sizes are tested: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]. An example of invoking `test_sync.py` for the TEI project can be seen below:

```bash
MODEL=sentence-transformers/all-mpnet-base-v2
PORT=8081
ENDPOINT=/embed
RESULTS_DIR=../../../hf-benchmarks
mkdir -p $RESULTS_DIR/results
python test_sync.py --model $MODEL \
    --port $PORT --endpoint $ENDPOINT --max_length=512 \
    --save-result \
    --result-dir $RESULTS_DIR \
    --num-prompts 5120
```

2. `test_async.py` runs tests that sends parallel requests to the specified backend asynchronously via multiple clients. Each test runs with a different client pool size. By default, the following number of clients are used for each test: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]. An example of invoking `test_async.py` for the TEI project can be seen below:

```bash
MODEL=sentence-transformers/all-mpnet-base-v2
PORT=8081
ENDPOINT=/embed
RESULTS_DIR=../../../hf-benchmarks
mkdir -p $RESULTS_DIR
python test_async.py --model $MODEL \
    --port $PORT --endpoint $ENDPOINT --max_length=512 \
    --save-result \
    --result-dir $RESULTS_DIR \
    --num-prompts 5120 \
    --batch_size 8 \
```

A benchmark suite is also provided in `run_test.sh`. It runs a `test_sync.py` and multiple instances of `test_async.py` sweeping through batch size. All results are saved in `../../../hf-benchmarks/` (outside of this repository).

```bash
./run_test.sh <MODEL ID>
```

## Support Table

| Project | `test_sync.py` supported | `test_async.py` supported | `docker.py` supported |
| ---- | ---- | ---- | ---- |
| TEI | ✔️ | ✔️ | ✔️ |
| TGI | - | ✔️ | - |
| vLLM | - | ✔️ | - |
| OpenAI completions | - | ✔️ | - |
| OpenAI chat | - | ✔️ | - |
| TensorRT LLM | - | ✔️ | - |

## Utility Scripts

### `docker.py`

The docker script is a wrapper around the `docker run` command and provides several helper flags in creating a docker environment. Currently, it is only validated for TEI docker environments.

The `docker.py` file can also be used as a library for automated docker environment creation and destruction using Python's context manager syntax. A simple usage of it can be shown below:

```python
docker_args = DockerArgs(
    docker_image="tei-gaudi",
    data_volume="~/data",
    truncate=False,
    platform="gaudi2",
    debug=False,
    docker_port=8081,
    docker_env_vars=None,
    trust_remote_code=False,
    max_client_batch_size=32,
)
log_file = Path("~/tei-docker.log")
model = Model("sentence-transformers/all-distilroberta-v1")

with DockerProcess(model, open(log_file,"w"), docker_args, name_prefix="testing") as dp:
    dp.wait_until_ready(log_file, timeout=60)
    # Perform docker arguments here

# Docker environment closed here.
```
