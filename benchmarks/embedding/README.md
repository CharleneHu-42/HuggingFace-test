# Huggingface Embeddings Benchmark Scripts

This directory contains benchmarks for Huggingface embedding frameworks with synchronous and asynchronous tests.

## Prerequisites

Install the Python requirements:

With `conda`:

```sh
conda env create -f environment.yml
```

With `pip`:

```sh
pip install -r requirements.txt
```

## Benchmarking Scripts

Benchmarking single-model scripts using synchronous and asynchronous requests are located in scripts `test_sync.py` and `test_async.py`, respectively. The help messages for each script are listed below:

```
usage: test_async.py [-h] [--backend {tgi,tei,tei-async,vllm,lmdeploy,deepspeed-mii,openai,openai-chat,tensorrt-llm}] [--base-url BASE_URL] [--host HOST] [--port PORT] [--endpoint ENDPOINT] --model MODEL [--tokenizer TOKENIZER] [--num-prompts NUM_PROMPTS] [--min_length MIN_LENGTH] [--max_length MAX_LENGTH]
                     [--batch_size BATCH_SIZE] [--client_num CLIENT_NUM] [--request-rate REQUEST_RATE] [--seed SEED] [--disable-tqdm] [--save-result] [--metadata [KEY=VALUE ...]] [--result-dir RESULT_DIR]

Benchmark the online serving throughput.

options:
  -h, --help            show this help message and exit
  --backend {tgi,tei,tei-async,vllm,lmdeploy,deepspeed-mii,openai,openai-chat,tensorrt-llm}
  --base-url BASE_URL   Server or API base url if not using http host and port.
  --host HOST
  --port PORT
  --endpoint ENDPOINT   API endpoint.
  --model MODEL         Name of the model.
  --tokenizer TOKENIZER
                        Name or path of the tokenizer, if not using the default tokenizer.
  --num-prompts NUM_PROMPTS
                        Number of prompts to process.
  --min_length MIN_LENGTH
                        min length of input prompt's token id.
  --max_length MAX_LENGTH
                        max length of input prompt's token id.
  --batch_size BATCH_SIZE
                        batch size of an input request prompt.
  --client_num CLIENT_NUM
                        num of clients to send requests concurrently
  --request-rate REQUEST_RATE
                        Number of requests per second. If this is inf, then all the requests are sent at time 0. Otherwise, we use Poisson process to synthesize the request arrival times.
  --seed SEED
  --disable-tqdm        Specify to disable tqdm progress bar.
  --save-result         Specify to save benchmark results to a json file
  --metadata [KEY=VALUE ...]
                        Key-value pairs (e.g, --metadata version=0.3.3 tp=1) for metadata of this run to be saved in the result JSON file for record keeping purposes.
  --result-dir RESULT_DIR
                        Specify directory to save benchmark json results.If not specified, results are saved in the current directory.
```

```
usage: test_sync.py [-h] [--backend {tgi,tei,tei-async,vllm,lmdeploy,deepspeed-mii,openai,openai-chat,tensorrt-llm}] [--base-url BASE_URL] [--host HOST] [--port PORT] [--endpoint ENDPOINT] --model MODEL [--tokenizer TOKENIZER] [--num-prompts NUM_PROMPTS] [--min_length MIN_LENGTH] [--max_length MAX_LENGTH]
                    [--request-rate REQUEST_RATE] [--seed SEED] [--disable-tqdm] [--save-result] [--metadata [KEY=VALUE ...]] [--result-dir RESULT_DIR]

Benchmark the online serving throughput.

options:
  -h, --help            show this help message and exit
  --backend {tgi,tei,tei-async,vllm,lmdeploy,deepspeed-mii,openai,openai-chat,tensorrt-llm}
  --base-url BASE_URL   Server or API base url if not using http host and port.
  --host HOST
  --port PORT
  --endpoint ENDPOINT   API endpoint.
  --model MODEL         Name of the model.
  --tokenizer TOKENIZER
                        Name or path of the tokenizer, if not using the default tokenizer.
  --num-prompts NUM_PROMPTS
                        Number of prompts to process.
  --min_length MIN_LENGTH
                        min length of input prompt's token id.
  --max_length MAX_LENGTH
                        max length of input prompt's token id.
  --request-rate REQUEST_RATE
                        Number of requests per second. If this is inf, then all the requests are sent at time 0. Otherwise, we use Poisson process to synthesize the request arrival times.
  --seed SEED
  --disable-tqdm        Specify to disable tqdm progress bar.
  --save-result         Specify to save benchmark results to a json file
  --metadata [KEY=VALUE ...]
                        Key-value pairs (e.g, --metadata version=0.3.3 tp=1) for metadata of this run to be saved in the result JSON file for record keeping purposes.
  --result-dir RESULT_DIR
                        Specify directory to save benchmark json results.If not specified, results are saved in the current directory.
```

## Utility Scripts

### `docker.py`

The docker script is a wrapper around the `docker run` command and provides several helper flags in creating a docker environment. Currently, it is only validated for TEI docker environments.

```
usage: docker.py [-h] [--model_name MODEL_NAME] [--revision REVISION] [--docker_container DOCKER_CONTAINER] [--data_volume DATA_VOLUME] [--truncate] [--platform {gaudi2,a100,cpu}] [--debug] [--docker_port DOCKER_PORT] [--docker_env_vars [DOCKER_ENV_VARS ...]] [--trust_remote_code]
                 [--max_client_batch_size MAX_CLIENT_BATCH_SIZE]

Launch a docker environment

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Model name to launch the docker environment with. (default: sentence-transformers/all-distilroberta-v1)
  --revision REVISION   Revision of the model to use (default: None)
  --docker_container DOCKER_CONTAINER
                        Name or hash of the docker container to launch. (default: tei-gaudi)
  --data_volume DATA_VOLUME
                        Data cache for Huggingface Models. (default: /home/daniel/data)
  --truncate            Truncate sentences to model token limit. (default: False)
  --platform {gaudi2,a100,cpu}
                        Plotform that the benchmark is running on. (default: gaudi2)
  --debug               Switches log to debug mode. (default: False)
  --docker_port DOCKER_PORT
                        Port for sending requests (default: 8081)
  --docker_env_vars [DOCKER_ENV_VARS ...], -e [DOCKER_ENV_VARS ...]
                        Docker environment variables in the format `--docker_env_vars <var1>=<val1> <var2>=<val2> ...` (default: None)
  --trust_remote_code   Enables remote code from model. Required for some models (default: False)
  --max_client_batch_size MAX_CLIENT_BATCH_SIZE
                        Batch size allowed by client (default: 32)
```

The `docker.py` file can also be used as a library for automated docker environment creation and destruction using Python's context manager syntax. A simple usage of it can be shown below:

```python
docker_args = DockerArgs(
    docker_container="tei-gaudi",
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

### `run_test.sh`

The run-test script runs a pre-compiled suite of tests for an embedding model.

Usage:
```sh
./run_test.sh <MODEL ID>
```

This will run a synchronous test sweeping through batch size and multiple asynchronous tests to sweep through client size and batch size. The model results will be saved in an external directory `hf-benchmarks`.
