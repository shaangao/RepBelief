from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
import numpy as np
import torch
from transformers.generation.utils import GenerationConfig

class LM_hf():
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.to(self.device)
    def generate_response(self, prompt):
        messages = [{"role": "user", "content": prompt},]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded[0]
        
    def parse_chat_response(self, response):
        answer_idx = response.find('[/INST]')
        return response[answer_idx+8:].strip().strip('</s>')
        
    def __call__(self, prompt):
        ans = self.generate_response(prompt)
        return self.parse_chat_response(ans)


class LM_nnsight():
    def __init__(self, model_path, device="cuda", temperature=0.):
        self.device = device
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            base_model.resize_token_embeddings(len(tokenizer))
        base_model.generation_config = GenerationConfig.from_pretrained(model_path)
        if temperature == 0:
            base_model.generation_config.do_sample = False
            base_model.generation_config.temperature = None
            base_model.generation_config.top_p = None
            base_model.generation_config.top_k = None
            #print(base_model.generation_config.temperature)
        else:
            base_model.generation_config.temperature = temperature
        base_model.to(self.device)
        base_model.eval()
        self.tokenizer = tokenizer
        # nnsight>=0.7 does not expose `config` on a model wrapped from an
        # existing module, so keep a reference to the underlying HF config.
        self.config = base_model.config
        self.model = LanguageModel(base_model, tokenizer=tokenizer)

    def generate_response(self, prompt, max_new_tokens=2000):
        # nnsight>=0.7: tracing-context generation; the final sequence is
        # exposed via `model.generator.output`.
        with self.model.generate(prompt, max_new_tokens=max_new_tokens):
            output = self.model.generator.output.save()
        return self.tokenizer.decode(output[0])

    def __call__(self, prompt, max_new_tokens=2000):
        ans = self.generate_response(prompt, max_new_tokens)
        return ans

    @staticmethod
    def _to_seq_hidden(proxy):
        """Resolve a saved nnsight proxy to a (Tokens, hidden) numpy array.

        In nnsight>=0.7 saved proxies resolve directly to ``torch.Tensor``
        (no ``.value``). Decoder-layer outputs may or may not carry a leading
        batch dim depending on the transformers version, so squeeze it here.
        """
        arr = proxy.detach().cpu().float().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    def get_all_states(self, prompt):
        n_heads = self.config.num_attention_heads

        all_hidden_states = []
        all_attention_states = []
        with self.model.trace(prompt):
            for layer in self.model.model.layers:
                all_attention_states.append(layer.self_attn.output[0].save())
                all_hidden_states.append(layer.output.save())

        all_hidden_states_numpy = []
        all_attention_states_numpy = []
        for HS, AS in zip(all_hidden_states, all_attention_states):
            all_hidden_states_numpy.append(self._to_seq_hidden(HS))
            atts = self._to_seq_hidden(AS)
            all_attention_states_numpy.append(atts.reshape(atts.shape[0], n_heads, -1))
        all_hidden_states_numpy = np.array(all_hidden_states_numpy)
        all_attention_states_numpy = np.array(all_attention_states_numpy)

        return all_hidden_states_numpy, all_attention_states_numpy
        # all_hidden_states: (Layers, Tokens, hidden_size)
        # all_attention_states: (Layers, Tokens, Heads, head_dim)

    def intervention(self, prompt, interventions_dict, alpha=10, max_new_tokens=3):
        n_heads = self.config.num_attention_heads
        head_dim = int(self.config.hidden_size / n_heads)
        # nnsight>=0.7: inside `generate`, module-output edits apply to every
        # generated token automatically (interleaver default_all), so no manual
        # per-token `.next()` loop is required.
        with self.model.generate(prompt, max_new_tokens=max_new_tokens):
            for layer_id, layer in enumerate(self.model.model.layers):
                if layer_id in interventions_dict:
                    attn_out = layer.self_attn.output[0]
                    for (head, dir, std, _) in interventions_dict[layer_id]:
                        direction = torch.as_tensor(
                            np.asarray(dir), device=attn_out.device, dtype=attn_out.dtype
                        )
                        attn_out[0, -1, head * head_dim: (head + 1) * head_dim] += alpha * std * direction
            output = self.model.generator.output.save()
        return self.tokenizer.decode(output[0])








        