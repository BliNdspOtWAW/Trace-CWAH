"""In-process chat generation; weights are shared, conversation state is not."""
import json
import time
from pathlib import Path

_BACKENDS = {}


def get_backend(path, options):
    key = (str(Path(path).expanduser().resolve()), json.dumps(options, sort_keys=True))
    if key not in _BACKENDS:
        _BACKENDS[key] = LocalBackend(key[0], options)
    return _BACKENDS[key]


class LocalBackend:
    def __init__(self, path, options):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.options = options
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, options.get('dtype', 'bfloat16')),
            device_map=options.get('device_map', 'cuda:0'),
            attn_implementation='sdpa',
        ).eval()

    def generate(self, messages, params):
        import torch
        started = time.monotonic()
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.options.get('thinking', False),
        )
        inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        prompt_tokens = inputs['input_ids'].shape[-1]
        budget = int(params.get('max_tokens', params.get('max_new_tokens', 1024)))
        limit = min(self.options.get('context_limit', 32768),
                    getattr(self.model.config, 'max_position_embeddings', 32768))
        if prompt_tokens + budget > limit:
            raise ValueError(f'Local context exceeds limit: {prompt_tokens}+{budget}>{limit}; no history was truncated')
        inputs = inputs.to(self.model.device)
        temperature = float(params.get('temperature', 0))
        generation = dict(max_new_tokens=budget, do_sample=temperature > 0,
                          pad_token_id=self.tokenizer.eos_token_id,
                          use_cache=True, return_dict_in_generate=False)
        if temperature > 0:
            generation.update(temperature=temperature, top_p=params.get('top_p', 1.0))
        samples, details, completion_tokens = [], [], 0
        # Sequential sampling also supports n>1 with greedy decoding.
        for _ in range(int(params.get('n', params.get('num_return_sequences', 1)))):
            with torch.inference_mode():
                ids = self.model.generate(**inputs, **generation)[0, prompt_tokens:]
            raw = self.tokenizer.decode(ids, skip_special_tokens=True)
            content = raw.split('</think>', 1)[-1] if '</think>' in raw else raw
            if '<think>' in content:
                raise ValueError('Local output exhausted budget inside a thinking block')
            samples.append(content.strip())
            details.append(raw)
            completion_tokens += len(ids)
        n = len(samples)
        return samples, dict(prompt_tokens=prompt_tokens * n,
                             completion_tokens=completion_tokens,
                             total_tokens=prompt_tokens * n + completion_tokens,
                             latency_seconds=time.monotonic() - started,
                             raw_outputs=details)
