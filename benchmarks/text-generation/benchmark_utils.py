import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from collections import OrderedDict
import os
import numpy as np
import requests

DTYPE_STR_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class BenchmarkPipeline:
    def __init__(self, args):
        self.backend = args.backend
        self.model_name = args.model_name
        self.dtype = DTYPE_STR_MAPPING[args.data_type]
        self.device = args.device
        self.model = self._load_model(
            args.backend,
            args.model_name,
            args.engine_dir,
            self.dtype,
            self.device,
            args.gpu_memory_utilization,
        )
        self.tgi_endpoint = args.tgi_endpoint
        self.tokenizer = self._load_tokenizer(args.model_name)
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id
        self.engine_dir = args.engine_dir
        self.save_dir = args.save_dir
        self.warm_up_steps = args.warm_up_steps
        self.run_steps = args.run_steps
        self.batch_size = args.batch_size
        self.max_input_length = args.max_input_length
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.temperature = args.temperature
        self.max_new_tokens = args.max_new_tokens
        self.generation_config = dict(
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        if self.backend == "vllm":
            from vllm import SamplingParams

            self.sampling_params = SamplingParams(
                n=self.num_beams,
                use_beam_search=True if self.num_beams > 1 else False,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
        else:
            self.sampling_params = None
        self.latencies = []
        self.input_lens = []

    def _load_model(
        self, backend, model_name, engine_dir, dtype, device, gpu_memory_utilization
    ):
        if backend == "tgi":
            model = None
        elif backend == "trt-llm":
            from tensorrt_llm.runtime import ModelRunner

            model = ModelRunner.from_dir(engine_dir)
        elif backend == "optimum-intel":
            from optimum.intel import IPEXModelForCausalLM

            model = IPEXModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=dtype, export=True
            )
        elif backend == "vllm":
            from vllm import LLM

            model = LLM(
                model=model_name,
                trust_remote_code=True,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            if backend == "ipex":
                import intel_extension_for_pytorch as ipex

                model.to(device)
                model = ipex.optimize_transformers(
                    model.eval(), dtype=dtype, device=device, inplace=True
                )
            else:
                model.to(device)
        return model

    def _load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            legacy=False,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _prepare_inputs(self, batch_input_ids, input_lengths):
        if len(set(input_lengths)) > 1:
            max_length = max(input_lengths)
            paddings = [
                torch.ones(max_length - l, dtype=torch.int32) * self.pad_id
                for l in input_lengths
            ]
            batch_input_ids = [
                torch.cat([pad, x]) for x, pad in zip(batch_input_ids, paddings)
            ]
        return torch.stack(batch_input_ids)

    def _process_trt_outputs(self, outputs, input_lengths):
        output_ids = outputs["output_ids"]
        batch_size, num_beams, _ = output_ids.size()
        output_seq_lengths = outputs["sequence_lengths"]
        output_texts = []
        for batch_idx in range(batch_size):
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = output_seq_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][output_begin:output_end].tolist()
                output_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
                output_texts.append(output_text)
        return output_texts

    def decode_prompt(self, batch_prompt):
        batch_input_ids = self.tokenizer(
            batch_prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_input_length,
        )["input_ids"]
        batch_input_ids = [torch.tensor(x, dtype=torch.int32) for x in batch_input_ids]
        return batch_input_ids

    def __call__(self, batch_prompt):
        start = time.time()
        if self.backend == "tgi":
            # since tgi doesn't take text as input, we need to calcuate the input token length
            batch_input_ids = self.decode_prompt(batch_prompt)
            input_lengths = [x.size()[0] for x in batch_input_ids]
            self.input_lens.append(input_lengths)
            start = time.time()
            response = requests.post(
                f"{self.tgi_endpoint}/v1/completions",
                json={
                    "model": "tgi",
                    "prompt": batch_prompt,
                    "max_tokens": self.max_new_tokens,
                    "temperature": 0 if not self.do_sample else self.temperature,
                },
                stream=False,
            )
            response = response.json()
            output_texts = [out["text"] for out in response["choices"]]
        else:
            batch_input_ids = self.decode_prompt(batch_prompt)
            input_lengths = [x.size()[0] for x in batch_input_ids]
            self.input_lens.append(input_lengths)

            if self.backend == "trt-llm":
                outputs = self.model.generate(
                    batch_input_ids,
                    end_id=self.end_id,
                    pad_id=self.pad_id,
                    output_sequence_lengths=True,
                    return_dict=True,
                    **self.generation_config,
                )
                output_texts = self._process_trt_outputs(outputs, input_lengths)
            else:
                batch_input_ids = self._prepare_inputs(batch_input_ids, input_lengths)

                if self.backend == "vllm":
                    batch_input_ids = [
                        batch_input_ids[i].tolist()
                        for i in range(batch_input_ids.size()[0])
                    ]
                    outputs = self.model.generate(
                        prompt_token_ids=batch_input_ids,
                        sampling_params=self.sampling_params,
                    )
                    output_ids = [output.outputs[0].token_ids for output in outputs]
                else:
                    batch_input_ids = batch_input_ids.to(self.device)
                    outputs = self.model.generate(
                        batch_input_ids,
                        pad_token_id=self.pad_id,
                        use_cache=True,
                        **self.generation_config,
                    )
                    output_ids = [
                        outputs[i, max(input_lengths) :]
                        for i in range(outputs.size()[0])
                    ]
                output_texts = [
                    self.tokenizer.decode(output_id, skip_special_tokens=True)
                    for output_id in output_ids
                ]
        self.synchronize_device()
        end = time.time()
        self.latencies.append((end - start) * 1000)
        return output_texts

    def synchronize_device(self):
        if self.device == "xpu":
            torch.xpu.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def check_valid_length(self, prompt):
        return len(self.tokenizer.encode(prompt)) <= self.max_input_length

    def get_report_dict(self):
        report_fields = [
            "model_name",
            "precision",
            "batch_size",
            "do_sample",
            "temperature",
            "num_beams",
            "max_new_tokens",
            "num_samples",
            "total_input_length",
            "total_output_length",
            "avg_latency(ms)",
            "accuracy",
        ]
        report_dict = OrderedDict.fromkeys(report_fields)
        report_dict["model_name"] = self.model_name
        report_dict["precision"] = self.dtype
        report_dict["batch_size"] = self.batch_size
        report_dict["do_sample"] = self.do_sample
        report_dict["temperature"] = self.temperature
        report_dict["num_beams"] = self.num_beams
        report_dict["max_new_tokens"] = self.max_new_tokens
        return report_dict

    def get_csv_filename(self):
        return f"{self.backend}_{self.model_name[-10:]}_{self.dtype}_{self.device}.csv"

    def report(self, accuracy: float = float("nan")):
        report_dict = self.get_report_dict()
        input_lens = self.input_lens[self.warm_up_steps :].copy()
        latencies = self.latencies[self.warm_up_steps :].copy()

        if len(set([len(i) for i in input_lens])) > 1:
            ignore_idx = [
                idx
                for idx, item in enumerate(input_lens)
                if len(item) != self.batch_size
            ]
            input_lens = [
                item for idx, item in enumerate(input_lens) if idx not in ignore_idx
            ]
            latencies = [
                item for idx, item in enumerate(latencies) if idx not in ignore_idx
            ]

        input_lens_flattend = [i for lens in input_lens for i in lens]

        avg_latency = np.mean(latencies)
        report_dict["num_samples"] = len(input_lens_flattend)
        report_dict["total_input_length"] = np.sum(input_lens_flattend)
        report_dict["total_output_length"] = (
            len(input_lens_flattend) * self.max_new_tokens
        )
        report_dict["avg_latency(ms)"] = avg_latency
        report_dict["accuracy"] = accuracy
        report_dict["datetime"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        if os.path.isdir(self.save_dir):
            header = ",".join(report_dict.keys())
            line = ",".join([str(v) for v in report_dict.values()])
            print(line)
            file_name = self.get_csv_filename()
            file_full_path = os.path.join(self.save_dir, file_name)
            if not os.path.isfile(file_full_path):
                with open(file_full_path, "a") as file:
                    file.write(header + "\n")
            with open(file_full_path, "a") as file:
                file.write(line + "\n")

        return report_dict
