## Tensor Parallelism

Please make sure `transformers >= 4.48.0`

## use case
Go to `tests/workloads` directory.
### Inference
The following commands defaultly use CPU, please add flag: `--device xpu` if you use XPU.
(TODO: XPU is not ready for now.)

Pass `--tp_size 2` to enable the TP model, please notice you should pass a number greater than 1.
```bash
$ bash ./run.sh -t tp -m meta-llama/Llama-3.1-8B-Instruct --tp_size 2
```