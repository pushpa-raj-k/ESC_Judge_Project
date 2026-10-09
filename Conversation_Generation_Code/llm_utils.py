"""
llm_utils.py  --  one place that loads and runs every open-weight model.

Every other script in the project (persona diversification, conversation
generation, judging) imports LocalLLM from here, so no script needs OpenAI,
Groq or Gemini.

Typical use (Kaggle / Colab / local GPU):

    from llm_utils import LocalLLM, chat_messages
    llm = LocalLLM("llama")                       # alias or full HF repo id
    outs = llm.generate([chat_messages("You are ...", "Hello")], max_new_tokens=200)
    llm.unload()                                  # free VRAM before the next model

Requires: torch, transformers>=4.50, accelerate, bitsandbytes (for 4-bit).
"""
import gc
import os
from typing import Dict, List, Optional

# Short aliases -> Hugging Face repo ids (the models listed in the project SOP).
MODEL_REGISTRY = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",     # help-seeker
    "qwen": "Qwen/Qwen2.5-3B-Instruct",              # consultant A / judge 1
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",  # consultant B (4-bit)
    "phi": "microsoft/Phi-4-mini-instruct",          # judge 2
    "gemma": "google/gemma-3-4b-it",                 # judge 3
}


def resolve_model_name(name: str) -> str:
    return MODEL_REGISTRY.get(name.lower(), name)


def get_hf_token() -> Optional[str]:
    """Token for gated models (Llama, Gemma). Looks in env, then Kaggle secrets."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:  # Kaggle: Add-ons -> Secrets -> label "HF_TOKEN"
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    try:  # Colab: key icon -> "HF_TOKEN"
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def chat_messages(system: Optional[str], user: str) -> List[Dict[str, str]]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs


class LocalLLM:
    def __init__(self, model_name: str, load_in_4bit: Optional[bool] = None,
                 hf_token: Optional[str] = None, device: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.repo = resolve_model_name(model_name)
        token = hf_token or get_hf_token()

        # Default: 4-bit only for the 7B model (matches the SOP); small models run in fp16.
        if load_in_4bit is None:
            load_in_4bit = "7b" in self.repo.lower()
        self.load_in_4bit = load_in_4bit

        # Mistral (v0.1-v0.3) templates reject a separate system role.
        self.merge_system = "mistral" in self.repo.lower()

        self.tok = AutoTokenizer.from_pretrained(self.repo, token=token)
        self.tok.padding_side = "left"          # required for batched decoder-only generation
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kwargs = dict(token=token, torch_dtype=torch.float16)
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": device}  # keep the whole model on one GPU (Kaggle T4 x2: 0 or 1)
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.repo, **kwargs)
        except Exception:
            # Gemma-3 4B is a multimodal checkpoint; fall back to the generic loader.
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(self.repo, **kwargs)
        self.model.eval()
        try:  # weights only, in GiB: ~4.5 means 4-bit worked, ~13.5 means the 7B model is in fp16
            gib = self.model.get_memory_footprint() / 2**30
        except Exception:
            gib = float("nan")
        print(f"[LocalLLM] loaded {self.repo} (4bit={load_in_4bit}, device={device}, weights={gib:.1f} GiB)")

    # ------------------------------------------------------------------ prompt
    def _to_prompt(self, messages: List[Dict[str, str]]) -> str:
        msgs = [dict(m) for m in messages]
        if self.merge_system and msgs and msgs[0]["role"] == "system":
            sys_text = msgs.pop(0)["content"]
            if msgs and msgs[0]["role"] == "user":
                msgs[0]["content"] = f"{sys_text}\n\n{msgs[0]['content']}"
            else:
                msgs.insert(0, {"role": "user", "content": sys_text})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ---------------------------------------------------------------- generate
    def generate(self, conversations: List[List[Dict[str, str]]], max_new_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 0.9, batch_size: int = 4,
                 verbose: bool = True, repetition_penalty: float = 1.0) -> List[str]:
        """conversations: list of chat-message lists. Returns one string per conversation,
        in the same order. temperature=0 means greedy decoding."""
        torch = self.torch
        prompts = [self._to_prompt(m) for m in conversations]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths together
        outputs: List[Optional[str]] = [None] * len(prompts)
        # last_hit_limit[i] is True when output i stopped because max_new_tokens was reached
        # (i.e. the reply is probably cut off mid-sentence) instead of ending naturally.
        self.last_hit_limit: List[bool] = [False] * len(prompts)
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.eos_token_id, self.tok.pad_token_id}

        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
        if repetition_penalty and repetition_penalty != 1.0:
            gen_kwargs.update(repetition_penalty=repetition_penalty)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)

        def run(idx_list):
            enc = self.tok([prompts[i] for i in idx_list], return_tensors="pt",
                           padding=True, add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            hit = [int(row[-1]) not in eos_ids for row in new_tokens]
            return [t.strip() for t in self.tok.batch_decode(new_tokens, skip_special_tokens=True)], hit

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            try:
                texts, hits = run(idx)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                texts, hits = [], []
                for i in idx:                    # fall back to one-by-one on OOM
                    t1, h1 = run([i])
                    texts.extend(t1)
                    hits.extend(h1)
                    torch.cuda.empty_cache()
            for i, t, h in zip(idx, texts, hits):
                outputs[i] = t
                self.last_hit_limit[i] = h
            if verbose:
                print(f"  generated {min(start + batch_size, len(order))}/{len(order)}", end="\r")
        if verbose:
            print()
        return outputs

    # ------------------------------------------------------------------ unload
    def unload(self):
        """Free GPU memory so the next model can be loaded."""
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class MockLLM:
    """Stand-in used for dry runs (no GPU, no downloads). Returns canned text that
    contains whichever marker ('Final Persona:' etc.) the calling prompt asks for."""

    def generate(self, conversations, **kwargs):
        self.last_hit_limit = [False] * len(conversations)
        outs = []
        for conv in conversations:
            text = conv[-1]["content"]
            if '"System Prompt:"' in text:
                outs.append("System Prompt:\n- Persona: mock persona text for testing the pipeline end to end.\n"
                            "- Key life events: mock events.\n- Behavioral traits: mock traits.\n"
                            "- Ongoing challenge: mock challenge.")
            elif '"Key Events:"' in text:
                outs.append("Key Events:\n1. mock event one for testing purposes only.\n2. mock event two.")
            elif "on each of the following dimensions" in text:
                import random as _r, re as _re
                names = _re.findall(r"^\d+\. (.+?): ", text, flags=_re.M)
                rng = _r.Random(hash(text) % 10_000)
                blocks = [f"### {n}\nReasoning: mock reasoning.\nVerdict: {rng.choice(['Model A', 'Model B', 'Tie'])}"
                          for n in names]
                if rng.random() < 0.2:          # simulate a truncated answer to exercise the retry logic
                    blocks = blocks[:-1]
                outs.append("\n\n".join(blocks))
            elif "Final Persona:" in text:
                outs.append("Final Persona: A mock persona who is 25 years old, used only for dry-run testing.")
            else:  # conversation turn
                outs.append("Mock reply for dry-run testing.")
        return outs

    def unload(self):
        pass
