import torch
import torch.nn as nn
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict
import os
import numpy as np

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
        self.model_name = args.model_name
        self.model = self._load_model(
            args.eval_mode,
            args.model_name,
            args.engine_dir,
            args.data_type,
            args.device,
        )
        self.tokenizer = self._load_tokenizer(args.model_name)
        self.data_type = args.data_type
        self.batch_size = args.batch_size
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id
        self.eval_mode = args.eval_mode
        self.engine_dir = args.engine_dir
        self.device = args.device
        self.max_input_length = args.max_input_length
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_new_tokens = args.max_new_tokens
        self.generation_config = dict(
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
        )
        self.latencies = []
        self.input_lens = []

    def _load_model(self, eval_mode, model_name, engine_dir, data_type, device):
        dtype = DTYPE_STR_MAPPING[data_type]
        if eval_mode == "trt-llm":
            from tensorrt_llm.runtime import ModelRunner

            model = ModelRunner.from_dir(engine_dir)
        elif eval_mode == "optimum-intel":
            from optimum.intel import IPEXModelForCausalLM

            model = IPEXModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, export=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=device,
                torch_dtype=dtype,
            )
            if eval_mode == "ipex":
                import intel_extension_for_pytorch as ipex

                model = ipex.optimize_transformers(
                    model.eval(), dtype=dtype, device=device, inplace=True
                )
        return model

    def _load_tokenizer(self, model_name):
        # TODO: why left padding?
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
        max_length = max(input_lengths)
        paddings = [
            torch.ones(max_length - l, dtype=torch.int32) * self.pad_id
            for l in input_lengths
        ]
        batch_input_ids = [
            torch.cat([pad, x]) for x, pad in zip(batch_input_ids, paddings)
        ]
        batch_input_ids = torch.stack(batch_input_ids)
        return batch_input_ids

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
                output_text = self.tokenizer.decode(outputs)
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
        batch_input_ids = self.decode_prompt(batch_prompt)
        input_lengths = [x.size()[0] for x in batch_input_ids]
        self.input_lens.append(input_lengths)

        with torch.no_grad():
            if self.eval_mode == "trt-llm":
                start = time.time()
                outputs = self.model.generate(
                    batch_input_ids,
                    end_id=self.end_id,
                    pad_id=self.pad_id,
                    output_sequence_lengths=True,
                    return_dict=True,
                    **self.generation_config,
                )
                end = time.time()
                self.latencies.append((end - start) * 1000)
                self.synchronize_device()
                output_texts = self._process_trt_outputs(outputs, input_lengths)
            else:
                batch_input_ids = self._prepare_inputs(batch_input_ids, input_lengths)
                batch_input_ids = batch_input_ids.to(self.device)

                start = time.time()
                if self.eval_mode == "optimum-intel":
                    outputs = self.model.generate(
                        batch_input_ids, use_cache=True, **self.generation_config
                    )
                else:
                    outputs = self.model.generate(
                        batch_input_ids, **self.generation_config
                    )
                end = time.time()
                self.latencies.append((end - start) * 1000)
                self.synchronize_device()
                output_ids = [
                    outputs[i, max(input_lengths) :] for i in range(outputs.size()[0])
                ]
                output_texts = [
                    self.tokenizer.decode(output_id, skip_special_tokens=True)
                    for output_id in output_ids
                ]
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
        report_dict["precision"] = self.data_type
        report_dict["batch_size"] = self.batch_size
        report_dict["do_sample"] = self.do_sample
        report_dict["num_beams"] = self.num_beams
        report_dict["max_new_tokens"] = self.max_new_tokens
        return report_dict

    def get_csv_filename(self):
        return f"{self.eval_mode}_{self.model_name[-10:]}_{self.data_type}_{self.device}.csv"

    def report(self, output_dir, accuracy):
        report_dict = self.get_report_dict()
        input_lens = self.input_lens.copy()
        latencies = self.latencies.copy()

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

        if os.path.isdir(output_dir):
            header = ",".join(report_dict.keys())
            line = ",".join([str(v) for v in report_dict.values()])
            print(line)
            file_name = self.get_csv_filename()
            file_full_path = os.path.join(output_dir, file_name)
            if not os.path.isfile(file_full_path):          
                with open(file_full_path, "a") as file:
                    file.write(header + "\n")
            with open(file_full_path, "a") as file:
                file.write(line + "\n")
        else:
            kv_pairs = [f"{k} {v}" for k, v in report_dict.items()]
            line = '[BENCHMARK] ' + " ".join(kv_pairs)
            print(line)
