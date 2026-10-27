from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import requests

try:
    from .model_profiles import ModelProfile, build_hf_model_map, default_model_list, resolve_profile
    from .parsing import infer_task_id, normalize_label
except ImportError:
    from model_profiles import ModelProfile, build_hf_model_map, default_model_list, resolve_profile
    from parsing import infer_task_id, normalize_label


class LLMAdapter:
    def __init__(self, legacy_module):
        self.legacy = legacy_module
        self.log_dir = Path(legacy_module.OUT_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.log_dir / "llm_audit.jsonl"
        self._audit_lock = threading.Lock()  # V16: thread-safe audit writes

    def set_log_dir(self, log_dir: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.log_dir / "llm_audit.jsonl"

    def check_backend(self) -> bool:
        backend = self.legacy.BACKEND
        model_list = getattr(self.legacy, "MODEL_LIST", default_model_list())

        if backend == "openrouter":
            api_key = getattr(self.legacy, "OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                self.legacy._log(
                    "[V18][ERR] BACKEND=openrouter but OPENROUTER_API_KEY is not set. "
                    "Export OPENROUTER_API_KEY before running.",
                    "error",
                )
                return False
            for model_name in model_list:
                profile = resolve_profile(model_name)
                is_fallback = profile.alias != model_name
                flag = " [FALLBACK — no explicit profile]" if is_fallback else ""
                self.legacy._log(
                    f"[V18] planned model={model_name} | openrouter_id={profile.hf_model_id}{flag} | "
                    f"family={profile.family} | max_new_tokens={profile.max_new_tokens} | "
                    f"system_style={profile.system_style}"
                )
                if is_fallback:
                    self.legacy._log(
                        f"[V18][WARN] '{model_name}' resolved via family fallback to '{profile.alias}'. "
                        f"Add an explicit MODEL_PROFILES entry.",
                        "warning",
                    )
            self.legacy._log(f"[V18] OpenRouter backend ready | key_prefix={api_key[:8]}***")
            return True

        if backend == "vllm":
            try:
                import torch
                from vllm import LLM, SamplingParams  # noqa: F401

                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                tp = getattr(self.legacy, "VLLM_TENSOR_PARALLEL_SIZE", 0) or gpu_count or 1
                self.legacy._log(
                    f"[V18] vLLM backend available | torch={torch.__version__} | "
                    f"cuda={'available' if torch.cuda.is_available() else 'NOT available'} | "
                    f"gpu_count={gpu_count} | tensor_parallel_size={tp}"
                )
                for model_name in model_list:
                    profile = resolve_profile(model_name)
                    if profile.family == "fingpt":
                        self.legacy._log(
                            f"[V18][ERR] vLLM backend does not support PEFT/LoRA adapter models "
                            f"(family='fingpt'). Remove '{model_name}' from MODEL_LIST or use "
                            f"BACKEND=transformers.",
                            "error",
                        )
                        return False
                    self.legacy._log(
                        f"[V18] planned model={model_name} | hf={profile.hf_model_id} | "
                        f"family={profile.family} | max_new_tokens={profile.max_new_tokens}"
                    )
                return True
            except ImportError as exc:
                self.legacy._log(
                    f"[V18][ERR] vLLM backend unavailable: {exc}. "
                    f"Install with: pip install vllm",
                    "error",
                )
                return False

        if backend == "transformers":
            try:
                import torch
                import transformers  # noqa: F401

                self.legacy._log(
                    f"[V18] Transformers backend available | torch={torch.__version__} | "
                    f"cuda={'available, device_count=' + str(torch.cuda.device_count()) if torch.cuda.is_available() else 'NOT available'}"
                )

                # V17: check peft availability upfront if any fingpt model is planned.
                needs_peft = any(resolve_profile(m).family == "fingpt" for m in model_list)
                if needs_peft:
                    try:
                        import peft  # noqa: F401
                        self.legacy._log(f"[V18] peft available: {peft.__version__}")
                    except ImportError:
                        self.legacy._log(
                            "[V18][ERR] Model list includes a fingpt (LoRA) model but 'peft' "
                            "package is not installed. Run: pip install peft",
                            "error",
                        )
                        return False

                for model_name in model_list:
                    profile = resolve_profile(model_name)
                    is_fallback = profile.alias != model_name
                    flag = " [FALLBACK — no explicit profile]" if is_fallback else ""
                    peft_note = f" | base_model={profile.base_model_id}" if profile.family == "fingpt" else ""
                    self.legacy._log(
                        f"[V18] planned model={model_name} | profile_alias={profile.alias}{flag} | "
                        f"family={profile.family} | hf={profile.hf_model_id}{peft_note} | "
                        f"max_new_tokens={profile.max_new_tokens} | system_style={profile.system_style}"
                    )
                    if is_fallback:
                        self.legacy._log(
                            f"[V18][WARN] '{model_name}' resolved via family fallback to '{profile.alias}'. "
                            f"This may cause incorrect generation config. Add explicit MODEL_PROFILES entry.",
                            "warning",
                        )
                return True
            except ImportError as exc:
                self.legacy._log(f"[V18][ERR] Transformers backend unavailable: {exc}", "error")
                return False

        # Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
            self.legacy._log("[V18] Ollama backend is available.")
            return True
        except Exception as exc:  # noqa: BLE001
            self.legacy._log(f"[V18][ERR] Ollama backend unavailable: {exc}", "error")
            return False

    def query(self, prompt: str, model: str | None = None) -> str:
        model_name = model or self.legacy.MODEL
        profile = resolve_profile(model_name)
        task_id = infer_task_id(prompt)
        system_text, user_text = self._split_prompt(prompt)
        try:
            raw = self._query_backend(model_name, profile, system_text, user_text)
        except Exception as exc:
            self.legacy._log(
                f"[V18][ERR] _query_backend raised {type(exc).__name__}: {exc} "
                f"(model={model_name}, task={task_id}). Returning sentinel, skipping sample.",
                "error",
            )
            raw = "[NULL_CONTENT]"
        normalized = normalize_label(raw, task_id)
        self._audit(model_name, profile, task_id, system_text, user_text, raw, normalized)
        return normalized

    def _split_prompt(self, prompt: str) -> tuple[str, str]:
        system_text = ""
        user_text = prompt
        if "[CB_SYSTEM]" in prompt and "[/CB_SYSTEM]" in prompt:
            system_text = prompt.split("[CB_SYSTEM]", 1)[1].split("[/CB_SYSTEM]", 1)[0].strip()
        if "[CB_USER]" in prompt and "[/CB_USER]" in prompt:
            user_text = prompt.split("[CB_USER]", 1)[1].split("[/CB_USER]", 1)[0].strip()
        return system_text, user_text

    def _query_backend(self, model_name: str, profile: ModelProfile, system_text: str, user_text: str) -> str:
        if self.legacy.BACKEND == "openrouter":
            return self._query_openrouter(model_name, profile, system_text, user_text)
        if self.legacy.BACKEND == "vllm":
            return self._query_vllm(model_name, profile, system_text, user_text)
        if self.legacy.BACKEND == "transformers":
            # V17: route LoRA/PEFT adapter models to dedicated loader
            if profile.family == "fingpt":
                return self._query_peft_lora(model_name, profile, system_text, user_text)
            return self._query_transformers(model_name, profile, system_text, user_text)
        return self._query_ollama(model_name, profile, system_text, user_text)

    def _query_ollama(self, model_name: str, profile: ModelProfile, system_text: str, user_text: str) -> str:
        prompt = user_text
        if system_text:
            prompt = f"{system_text}\n\n{user_text}"
        if profile.add_no_think:
            prompt = f"{prompt}\n/no_think"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": profile.temperature,
            "stream": False,
        }
        response = requests.post(self.legacy.OLLAMA_URL, json=payload, timeout=self.legacy.TIMEOUT)
        response.raise_for_status()
        return response.json()["response"].strip()

    # ── vLLM multi-GPU backend ────────────────────────────────────────────────

    def _get_vllm_tp(self) -> int:
        """Return tensor_parallel_size: config value or all visible GPUs."""
        tp = getattr(self.legacy, "VLLM_TENSOR_PARALLEL_SIZE", 0)
        if tp and tp > 0:
            return int(tp)
        try:
            import torch
            count = torch.cuda.device_count()
            return count if count > 0 else 1
        except ImportError:
            return 1

    def _init_vllm_engine(self, hf_model_id: str, tp: int):
        """Create or return cached vLLM LLM engine."""
        from vllm import LLM

        cache_key = f"vllm::{hf_model_id}::tp{tp}"
        if getattr(self.legacy, "_vllm_engine_key", None) == cache_key:
            return self.legacy._vllm_engine

        if getattr(self.legacy, "_vllm_engine_key", None) is not None:
            self.legacy._log(
                f"[V18] vLLM engine swap: unloading previous engine, "
                f"loading '{hf_model_id}' tp={tp}"
            )

        gpu_util = getattr(self.legacy, "VLLM_GPU_MEMORY_UTIL", 0.90)
        self.legacy._log(
            f"[V18] Loading vLLM engine: model={hf_model_id} | "
            f"tensor_parallel_size={tp} | gpu_memory_utilization={gpu_util}"
        )
        engine = LLM(
            model=hf_model_id,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_util,
            trust_remote_code=True,
            download_dir=self.legacy.HF_CACHE_DIR,
            dtype="float16",
        )
        self.legacy._vllm_engine = engine
        self.legacy._vllm_engine_key = cache_key
        self.legacy._log(f"[V18] vLLM engine loaded OK: {hf_model_id} | tp={tp}")
        return engine

    def _build_vllm_prompt(self, profile: ModelProfile, system_text: str, user_text: str) -> str:
        """Format a (system, user) pair into a single string using the chat template."""
        from transformers import AutoTokenizer

        hf_model_id = profile.hf_model_id
        if not hf_model_id:
            raise ValueError(f"[V18] vLLM: hf_model_id not set for profile '{profile.alias}'")

        tok_key = f"tok::{hf_model_id}"
        if getattr(self.legacy, "_vllm_tok_key", None) != tok_key:
            self.legacy._log(f"[V18] Loading tokenizer for prompt formatting: {hf_model_id}")
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model_id, trust_remote_code=True, cache_dir=self.legacy.HF_CACHE_DIR
            )
            self.legacy._vllm_tokenizer = tokenizer
            self.legacy._vllm_tok_key = tok_key

        tokenizer = self.legacy._vllm_tokenizer

        messages = []
        if system_text and profile.use_system_prompt:
            messages.append({"role": "system", "content": system_text})
        user_payload = user_text
        if system_text and not profile.use_system_prompt:
            user_payload = f"{system_text}\n\n{user_payload}"
        if profile.add_no_think:
            user_payload = f"{user_payload}\n/no_think"
        messages.append({"role": "user", "content": user_payload})

        if getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

        if profile.prompt_format == "alpaca":
            return self._alpaca_format(system_text, user_payload)
        return self._llama2_format(system_text, user_payload)

    def _batch_generate_vllm(
        self, formatted_prompts: list[str], profile: ModelProfile, model_name: str
    ) -> list[str]:
        """Run vLLM batch generation. Returns raw output strings (stripped)."""
        from vllm import SamplingParams

        hf_model_id = profile.hf_model_id or model_name
        tp = self._get_vllm_tp()
        engine = self._init_vllm_engine(hf_model_id, tp)

        temperature = profile.temperature if profile.do_sample else 0.0
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=profile.max_new_tokens,
        )

        batch_size = getattr(self.legacy, "VLLM_BATCH_SIZE", 512)
        all_raw: list[str] = []

        for i in range(0, len(formatted_prompts), batch_size):
            chunk = formatted_prompts[i: i + batch_size]
            self.legacy._log(
                f"[V18] vLLM generate: model={hf_model_id} | tp={tp} | "
                f"batch={i + 1}-{i + len(chunk)}/{len(formatted_prompts)} | "
                f"max_tokens={profile.max_new_tokens}"
            )
            t0 = time.time()
            outputs = engine.generate(chunk, sampling_params=sampling_params)
            elapsed = time.time() - t0
            raw_chunk = [out.outputs[0].text.strip() for out in outputs]
            all_raw.extend(raw_chunk)
            self.legacy._log(
                f"[V18] vLLM done: n={len(raw_chunk)} | "
                f"elapsed={elapsed:.1f}s | throughput={len(raw_chunk) / elapsed:.1f} req/s"
            )

        return all_raw

    def query_batch(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Batch inference: accepts V18-format prompts, returns normalized labels.

        Dispatches to the appropriate backend batch engine:
          - BACKEND='vllm'         → vLLM PagedAttention engine
          - BACKEND='transformers' → native HF batched generate (left-padded)

        Each element of `prompts` is the same format passed to query().
        Used by _vllm_batch_infer_loop in common.py.
        """
        if not prompts:
            return []

        model_name = model or self.legacy.MODEL
        profile = resolve_profile(model_name)

        parsed = [self._split_prompt(p) for p in prompts]
        task_ids = [infer_task_id(p) for p in prompts]

        backend = getattr(self.legacy, "BACKEND", "transformers")
        if backend == "vllm":
            formatted = [
                self._build_vllm_prompt(profile, sys_t, usr_t)
                for sys_t, usr_t in parsed
            ]
            raw_outputs = self._batch_generate_vllm(formatted, profile, model_name)
        else:
            raw_outputs = self._batch_generate_transformers(parsed, profile, model_name)

        results: list[str] = []
        for prompt, raw, task_id, (sys_t, usr_t) in zip(prompts, raw_outputs, task_ids, parsed):
            normalized = normalize_label(raw, task_id)
            self._audit(model_name, profile, task_id, sys_t, usr_t, raw, normalized)
            results.append(normalized)

        return results

    def _batch_generate_transformers(
        self,
        parsed_prompts: list[tuple[str, str]],
        profile: "ModelProfile",
        model_name: str,
    ) -> list[str]:
        """Dispatch to data-parallel (N GPUs × M replicas) or single-device batch inference.

        Data-parallel path  (n_gpus > 1, multi_gpu_only=False):
            Load REPLICAS_PER_GPU model copies on each GPU (default 1).
            Each replica gets its own thread — n_gpus × replicas threads run
            simultaneously, one per model copy.

        Single-device path  (n_gpus == 1, or multi_gpu_only=True):
            Original left-padded batch inference on one logical device.
            For 70B, device_map="auto" already spans all GPUs (tensor parallel).
        """
        import torch
        hf_model_id = profile.hf_model_id or build_hf_model_map().get(model_name) or model_name
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

        if n_gpus > 1 and not profile.multi_gpu_only:
            replicas_per_gpu = int(os.environ.get("REPLICAS_PER_GPU", "1"))
            return self._batch_generate_data_parallel(
                parsed_prompts, profile, model_name, hf_model_id, n_gpus, replicas_per_gpu
            )
        return self._batch_generate_single_device(
            parsed_prompts, profile, model_name, hf_model_id
        )

    # ── Prompt formatting helper (shared by both paths) ──────────────────────

    def _format_prompt_texts(
        self,
        parsed_prompts: list[tuple[str, str]],
        profile: "ModelProfile",
        tokenizer,
    ) -> list[str]:
        """Convert (system_text, user_text) pairs → tokenizer-ready strings."""
        texts: list[str] = []
        for system_text, user_text in parsed_prompts:
            messages = []
            if system_text and profile.use_system_prompt:
                messages.append({"role": "system", "content": system_text})
            user_payload = user_text
            if system_text and not profile.use_system_prompt:
                user_payload = f"{system_text}\n\n{user_payload}"
            if profile.add_no_think:
                user_payload = f"{user_payload}\n/no_think"
            messages.append({"role": "user", "content": user_payload})

            if getattr(tokenizer, "chat_template", None):
                try:
                    pt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    pt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
            elif profile.prompt_format == "alpaca":
                pt = self._alpaca_format(system_text, user_payload)
            else:
                pt = self._llama2_format(system_text, user_payload)
            texts.append(pt)
        return texts

    # ── Single-device batch inference (original path, also used for 70B TP) ──

    def _batch_generate_single_device(
        self,
        parsed_prompts: list[tuple[str, str]],
        profile: "ModelProfile",
        model_name: str,
        hf_model_id: str,
    ) -> list[str]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        loaded_name = getattr(self.legacy, "_hf_loaded_name", None)
        if loaded_name != hf_model_id:
            if loaded_name is not None:
                self.legacy._log(
                    f"[V18] [BATCH] Model swap: unloading '{loaded_name}', loading '{hf_model_id}'"
                )
            else:
                self.legacy._log(f"[V18] [BATCH] First model load: '{hf_model_id}'")
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model_id, trust_remote_code=True, cache_dir=self.legacy.HF_CACHE_DIR,
            )
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True, use_safetensors=True,
                attn_implementation="sdpa",
                cache_dir=self.legacy.HF_CACHE_DIR,
            )
            model.eval()
            self.legacy._hf_tokenizer = tokenizer
            self.legacy._hf_model = model
            self.legacy._hf_loaded_name = hf_model_id
            n_params = sum(p.numel() for p in model.parameters())
            self.legacy._log(
                f"[V18] [BATCH] Model loaded OK: '{hf_model_id}' | n_params={n_params:,} | "
                f"device={next(model.parameters()).device} | dtype={next(model.parameters()).dtype}"
            )
        else:
            self.legacy._log(f"[V18] [BATCH] Reusing cached model '{hf_model_id}'")

        tokenizer = self.legacy._hf_tokenizer
        model = self.legacy._hf_model
        prompt_texts = self._format_prompt_texts(parsed_prompts, profile, tokenizer)

        batch_size = profile.batch_size or getattr(self.legacy, "TRANSFORMERS_BATCH_SIZE", 16)
        gen_kwargs: dict = {
            "max_new_tokens": profile.max_new_tokens,
            "do_sample": profile.do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if profile.do_sample:
            gen_kwargs["temperature"] = profile.temperature

        all_raw: list[str] = []
        for i in range(0, len(prompt_texts), batch_size):
            chunk = prompt_texts[i: i + batch_size]
            self.legacy._log(
                f"[V18] [BATCH] transformers generate: model={hf_model_id} | "
                f"batch={i + 1}-{i + len(chunk)}/{len(prompt_texts)} | "
                f"max_new_tokens={profile.max_new_tokens}"
            )
            t0 = time.time()
            inputs = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
            ).to(model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            elapsed = time.time() - t0
            raw_chunk = [
                tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
                for seq in outputs
            ]
            all_raw.extend(raw_chunk)
            self.legacy._log(
                f"[V18] [BATCH] done: n={len(raw_chunk)} | "
                f"elapsed={elapsed:.1f}s | throughput={len(raw_chunk) / elapsed:.1f} req/s"
            )
        return all_raw

    # ── Data-parallel multi-GPU inference ────────────────────────────────────

    def _ensure_dp_replicas(
        self, hf_model_id: str, n_gpus: int, replicas_per_gpu: int = 1
    ) -> tuple:
        """Load or return cached model replicas + shared tokenizer.

        Loads replicas_per_gpu copies on each of the n_gpus GPUs, giving
        n_gpus × replicas_per_gpu total workers.

        Returns (tokenizer, [(gpu_id, model), ...]) — flat worker list.
        Clears cached replicas automatically when hf_model_id changes.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # cache key: (gpu_id, rep_id) → model
        cache: dict = getattr(self.legacy, "_dp_models", {})
        cache_id: str | None = getattr(self.legacy, "_dp_model_id", None)

        if cache_id != hf_model_id:
            if cache_id is not None:
                self.legacy._log(
                    f"[V18] [DP] Model swap: releasing {len(cache)} replica(s) "
                    f"for '{cache_id}'"
                )
            cache = {}
            self.legacy._dp_models = cache
            self.legacy._dp_model_id = hf_model_id
            self.legacy._dp_tokenizer = None

        if getattr(self.legacy, "_dp_tokenizer", None) is None:
            self.legacy._log(f"[V18] [DP] Loading shared tokenizer: {hf_model_id}")
            tok = AutoTokenizer.from_pretrained(
                hf_model_id, trust_remote_code=True, cache_dir=self.legacy.HF_CACHE_DIR,
            )
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self.legacy._dp_tokenizer = tok

        tokenizer = self.legacy._dp_tokenizer

        for gpu_id in range(n_gpus):
            for rep_id in range(replicas_per_gpu):
                key = (gpu_id, rep_id)
                if key not in cache:
                    self.legacy._log(
                        f"[V18] [DP] Loading: gpu={gpu_id} rep={rep_id + 1}/{replicas_per_gpu} "
                        f"'{hf_model_id}' → cuda:{gpu_id}"
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        hf_model_id,
                        torch_dtype=torch.float16,
                        device_map={"": f"cuda:{gpu_id}"},
                        trust_remote_code=True,
                        use_safetensors=True,
                        attn_implementation="sdpa",
                        cache_dir=self.legacy.HF_CACHE_DIR,
                    )
                    model.eval()
                    cache[key] = model
                    n_params = sum(p.numel() for p in model.parameters())
                    self.legacy._log(
                        f"[V18] [DP] Ready: cuda:{gpu_id} rep={rep_id} | "
                        f"n_params={n_params:,} | dtype=float16"
                    )

        # Flat worker list preserves GPU assignment for _run_generation_on_gpu
        workers = [
            (gpu_id, cache[(gpu_id, rep_id)])
            for gpu_id in range(n_gpus)
            for rep_id in range(replicas_per_gpu)
        ]

        # Keep single-model legacy cache in sync (used by _query_transformers fallback)
        self.legacy._hf_model = cache[(0, 0)]
        self.legacy._hf_tokenizer = tokenizer
        self.legacy._hf_loaded_name = hf_model_id

        return tokenizer, workers

    def _run_generation_on_gpu(
        self,
        gpu_id: int,
        model,
        tokenizer,
        prompt_texts: list[str],
        profile: "ModelProfile",
    ) -> list[str]:
        """Batched generation for one GPU replica. Runs inside a worker thread."""
        import torch

        device = f"cuda:{gpu_id}"
        replicas_per_gpu = int(os.environ.get("REPLICAS_PER_GPU", "1"))
        if replicas_per_gpu > 1:
            # profile.batch_size is calibrated for single-replica (fills one GPU alone).
            # With N replicas sharing a GPU, model weights alone take N×model_GB of VRAM,
            # leaving little room for KV cache.  Use TRANSFORMERS_BATCH_SIZE explicitly
            # (run_hpc.sh defaults it to 8 when REPLICAS_PER_GPU>1).
            batch_size = getattr(self.legacy, "TRANSFORMERS_BATCH_SIZE", 8)
        else:
            batch_size = profile.batch_size or getattr(self.legacy, "TRANSFORMERS_BATCH_SIZE", 16)
        gen_kwargs: dict = {
            "max_new_tokens": profile.max_new_tokens,
            "do_sample": profile.do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if profile.do_sample:
            gen_kwargs["temperature"] = profile.temperature

        all_raw: list[str] = []
        for i in range(0, len(prompt_texts), batch_size):
            chunk = prompt_texts[i: i + batch_size]
            self.legacy._log(
                f"[V18] [DP] cuda:{gpu_id} generate: "
                f"{i + 1}-{i + len(chunk)}/{len(prompt_texts)} | "
                f"max_new_tokens={profile.max_new_tokens}"
            )
            t0 = time.time()
            inputs = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
            ).to(device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            elapsed = time.time() - t0
            raw_chunk = [
                tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
                for seq in outputs
            ]
            all_raw.extend(raw_chunk)
            self.legacy._log(
                f"[V18] [DP] cuda:{gpu_id} done: n={len(raw_chunk)} | "
                f"elapsed={elapsed:.1f}s | {len(raw_chunk) / elapsed:.2f} req/s"
            )
        return all_raw

    def _batch_generate_data_parallel(
        self,
        parsed_prompts: list[tuple[str, str]],
        profile: "ModelProfile",
        model_name: str,
        hf_model_id: str,
        n_gpus: int,
        replicas_per_gpu: int = 1,
    ) -> list[str]:
        """Data-parallel inference: n_gpus × replicas_per_gpu threads, one per model copy.

        Each replica gets an independent subset of prompts and runs model.generate()
        in its own thread simultaneously.
        """
        from concurrent.futures import ThreadPoolExecutor

        tokenizer, workers = self._ensure_dp_replicas(hf_model_id, n_gpus, replicas_per_gpu)

        # Format ALL prompts in the main thread (apply_chat_template is not thread-safe).
        prompt_texts = self._format_prompt_texts(parsed_prompts, profile, tokenizer)

        # Split prompts across all available workers
        n_workers = min(len(workers), len(prompt_texts))
        chunk_size = (len(prompt_texts) + n_workers - 1) // n_workers
        chunks = [
            prompt_texts[i * chunk_size: (i + 1) * chunk_size]
            for i in range(n_workers)
        ]
        chunks = [c for c in chunks if c]
        n_active = len(chunks)
        active_workers = workers[:n_active]

        self.legacy._log(
            f"[V18] [DP] Starting: {len(prompt_texts)} prompts → "
            f"{n_gpus} GPU(s) × {replicas_per_gpu} replica(s) = {n_active} threads × "
            f"~{len(chunks[0])} prompts | model={hf_model_id}"
        )

        results: list[list[str] | None] = [None] * n_active

        def _worker(idx: int) -> None:
            gpu_id, model = active_workers[idx]
            results[idx] = self._run_generation_on_gpu(
                gpu_id, model, tokenizer, chunks[idx], profile
            )

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n_active) as pool:
            futures = [pool.submit(_worker, i) for i in range(n_active)]
            for fut in futures:
                fut.result()  # re-raises any worker exception in the main thread
        elapsed = time.time() - t0

        all_raw = [r for chunk_result in results for r in chunk_result]
        self.legacy._log(
            f"[V18] [DP] All {n_active} workers done | total={len(all_raw)} | "
            f"wall={elapsed:.1f}s | throughput={len(all_raw) / elapsed:.2f} req/s"
        )
        return all_raw

    def _query_vllm(
        self, model_name: str, profile: ModelProfile, system_text: str, user_text: str
    ) -> str:
        """Single-prompt vLLM inference (wraps batch API with n=1)."""
        formatted = self._build_vllm_prompt(profile, system_text, user_text)
        raw_list = self._batch_generate_vllm([formatted], profile, model_name)
        raw = raw_list[0]
        self.legacy._log(
            f"[V18] vLLM single query done: model={model_name} | raw={repr(raw[:80])}"
        )
        return raw

    def _query_openrouter(
        self, model_name: str, profile: ModelProfile, system_text: str, user_text: str
    ) -> str:
        """Call OpenRouter chat-completions endpoint with retry and token auto-scaling."""
        api_key = (
            getattr(self.legacy, "OPENROUTER_API_KEY", "")
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        if not api_key:
            raise RuntimeError("[V18] OPENROUTER_API_KEY not set.")

        base_url  = getattr(self.legacy, "OPENROUTER_BASE_URL",       "https://openrouter.ai/api/v1")
        site_url  = getattr(self.legacy, "OPENROUTER_SITE_URL",       "https://anonymous.institution")
        site_name = getattr(self.legacy, "OPENROUTER_SITE_NAME",      "CommitBench")
        timeout   = getattr(self.legacy, "OPENROUTER_TIMEOUT",        120)
        max_retries  = getattr(self.legacy, "OPENROUTER_MAX_RETRIES",       5)
        retry_base   = getattr(self.legacy, "OPENROUTER_RETRY_BASE_DELAY",  2.0)
        MAX_TOKENS_CEILING = 4096

        openrouter_model_id = profile.hf_model_id or model_name
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": site_url,
            "X-Title": site_name,
            "Content-Type": "application/json",
        }

        messages = []
        if system_text and profile.use_system_prompt:
            messages.append({"role": "system", "content": system_text})
        user_payload = user_text
        if profile.add_no_think:
            user_payload = f"{user_payload}\n/no_think"
        messages.append({"role": "user", "content": user_payload})

        current_max_tokens = profile.max_new_tokens
        payload = {
            "model": openrouter_model_id,
            "messages": messages,
            "temperature": profile.temperature,
            "max_tokens": current_max_tokens,
        }

        last_exc: Exception | None = None
        last_status: int | None = None
        for attempt in range(max_retries):
            payload["max_tokens"] = current_max_tokens
            try:
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_status = resp.status_code
                    delay = retry_base * (2 ** attempt)
                    self.legacy._log(
                        f"[V18] OpenRouter HTTP {resp.status_code} — retrying in {delay:.1f}s "
                        f"(attempt {attempt+1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                resp.raise_for_status()

                # Guard against non-JSON or malformed response bodies (V16 fix).
                try:
                    body = resp.json()
                except ValueError as exc:
                    last_exc = exc
                    delay = retry_base * (2 ** attempt)
                    self.legacy._log(
                        f"[V18][WARN] OpenRouter non-JSON response (attempt {attempt+1}/{max_retries}): "
                        f"{exc} — retrying in {delay:.1f}s",
                        "warning",
                    )
                    time.sleep(delay)
                    continue

                choices = body.get("choices")
                if not choices:
                    self.legacy._log(
                        f"[V18][ERR] OpenRouter response has no choices: {str(body)[:200]}. "
                        f"Returning sentinel, skipping sample.",
                        "error",
                    )
                    return "[NULL_CONTENT]"

                choice  = choices[0]
                content = choice.get("message", {}).get("content")
                if content is None:
                    finish  = choice.get("finish_reason", "")
                    refusal = choice.get("message", {}).get("refusal") or ""
                    if finish == "length":
                        if current_max_tokens < MAX_TOKENS_CEILING:
                            new_tokens = min(current_max_tokens * 4, MAX_TOKENS_CEILING)
                            self.legacy._log(
                                f"[V18][WARN] finish_reason='length' + null content: "
                                f"auto-scaling max_tokens {current_max_tokens}->{new_tokens} "
                                f"(attempt {attempt+1}/{max_retries})"
                            )
                            current_max_tokens = new_tokens
                            continue
                        self.legacy._log(
                            f"[V18][ERR] finish_reason='length' at ceiling max_tokens={MAX_TOKENS_CEILING} "
                            f"for model={openrouter_model_id}. Returning sentinel.",
                            "error",
                        )
                        return "[NULL_CONTENT]"
                    self.legacy._log(
                        f"[V18][ERR] null content with finish_reason={finish!r}, "
                        f"refusal={refusal[:60]!r}. Returning sentinel.",
                        "error",
                    )
                    return "[NULL_CONTENT]"
                self.legacy._log(
                    f"[V18] openrouter done: model={openrouter_model_id} | "
                    f"max_tokens_used={current_max_tokens} | raw={repr(content[:80])}"
                )
                return content.strip()
            except requests.RequestException as exc:
                last_exc = exc
                delay = retry_base * (2 ** attempt)
                self.legacy._log(
                    f"[V18][WARN] OpenRouter request error: {exc} — retrying in {delay:.1f}s "
                    f"(attempt {attempt+1}/{max_retries})"
                )
                time.sleep(delay)

        failure_reason = str(last_exc) if last_exc else f"HTTP {last_status} after all retries"
        raise RuntimeError(f"[V18] OpenRouter failed after {max_retries} attempts: {failure_reason}")

    def _query_transformers(self, model_name: str, profile: ModelProfile, system_text: str, user_text: str) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model_id_from_map = build_hf_model_map().get(model_name)
        hf_model_id = profile.hf_model_id or hf_model_id_from_map or model_name

        is_fallback = profile.alias != model_name
        self.legacy._log(
            f"[V18] _query_transformers: model_alias={model_name} | "
            f"profile_alias={profile.alias}{'  ← FALLBACK' if is_fallback else ''} | "
            f"hf_model_id={hf_model_id} | max_new_tokens={profile.max_new_tokens} | "
            f"system_style={profile.system_style} | add_no_think={profile.add_no_think}"
        )
        if is_fallback:
            self.legacy._log(
                f"[V18][WARN] Profile mismatch: requested '{model_name}' but profile is for "
                f"'{profile.alias}' (family-fallback). Add an explicit MODEL_PROFILES entry.",
                "warning",
            )
        if profile.hf_model_id is None and hf_model_id_from_map:
            self.legacy._log(
                f"[V18][INFO] profile.hf_model_id was None — resolved via HF_MODEL_MAP to '{hf_model_id}'."
            )

        loaded_name = getattr(self.legacy, "_hf_loaded_name", None)
        if loaded_name != hf_model_id:
            if loaded_name is not None:
                self.legacy._log(f"[V18] Model swap: unloading '{loaded_name}', loading '{hf_model_id}'")
            else:
                self.legacy._log(f"[V18] First model load: '{hf_model_id}'")
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model_id, trust_remote_code=True, cache_dir=self.legacy.HF_CACHE_DIR,
            )
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True, use_safetensors=True,
                attn_implementation="sdpa",
                cache_dir=self.legacy.HF_CACHE_DIR,
            )
            model.eval()
            self.legacy._hf_tokenizer = tokenizer
            self.legacy._hf_model = model
            self.legacy._hf_loaded_name = hf_model_id

            try:
                n_params = sum(p.numel() for p in model.parameters())
                first_param = next(model.parameters())
                self.legacy._log(
                    f"[V18] Model loaded OK: '{hf_model_id}' | n_params={n_params:,} | "
                    f"device={first_param.device} | dtype={first_param.dtype} | "
                    f"vocab_size={tokenizer.vocab_size}"
                )
                if "70b" in model_name.lower() and n_params < 50_000_000_000:
                    raise RuntimeError(
                        f"[V18] SAFETY ABORT: Loading '{model_name}' expected ≥50B parameters "
                        f"but got {n_params:,}. Check HF_HOME / cache isolation."
                    )
            except RuntimeError:
                raise
            except Exception:  # noqa: BLE001
                self.legacy._log(f"[V18] Model loaded (param count unavailable): '{hf_model_id}'")
        else:
            self.legacy._log(f"[V18] Reusing cached model '{hf_model_id}'")

        tokenizer = self.legacy._hf_tokenizer
        model = self.legacy._hf_model

        messages = []
        if system_text and profile.use_system_prompt:
            messages.append({"role": "system", "content": system_text})
        user_payload = user_text
        # Prepend system text for models that don't support a system role (e.g. Gemma-2).
        if system_text and not profile.use_system_prompt:
            user_payload = f"{system_text}\n\n{user_payload}"
        if profile.add_no_think:
            user_payload = f"{user_payload}\n/no_think"
        messages.append({"role": "user", "content": user_payload})

        if getattr(tokenizer, "chat_template", None):
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        else:
            if profile.prompt_format == "alpaca":
                prompt_text = self._alpaca_format(system_text, user_payload)
            else:
                prompt_text = self._llama2_format(system_text, user_payload)

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": profile.max_new_tokens,
                "do_sample": profile.do_sample,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if profile.do_sample:
                gen_kwargs["temperature"] = profile.temperature
            outputs = model.generate(**inputs, **gen_kwargs)
        output_tokens = outputs[0][input_len:]
        raw = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        self.legacy._log(
            f"[V18] generation done: input_tokens={input_len} | "
            f"output_tokens={len(output_tokens)} / max={profile.max_new_tokens} | "
            f"raw={repr(raw[:80])}"
        )
        return raw

    def _query_peft_lora(
        self, model_name: str, profile: ModelProfile, system_text: str, user_text: str
    ) -> str:
        """V17: Load a PEFT/LoRA adapter on top of a base model and run inference.

        Used for family="fingpt" (e.g. FinGPT-Forecaster).
          profile.hf_model_id  → LoRA adapter repo
          profile.base_model_id → base pretrained model
        Cached in legacy._hf_loaded_name under key "peft::{adapter_id}".
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            from peft import PeftModel
        except ImportError:
            raise RuntimeError(
                "[V18] 'peft' package not found. Install it first: pip install peft\n"
                "Required for LoRA adapter models (family='fingpt')."
            )

        adapter_id   = profile.hf_model_id or model_name
        base_model_id = profile.base_model_id or "meta-llama/Llama-2-7b-chat-hf"
        cache_key    = f"peft::{adapter_id}"

        self.legacy._log(
            f"[V18] _query_peft_lora: model_alias={model_name} | "
            f"adapter={adapter_id} | base={base_model_id} | "
            f"max_new_tokens={profile.max_new_tokens} | system_style={profile.system_style}"
        )

        loaded_name = getattr(self.legacy, "_hf_loaded_name", None)
        if loaded_name != cache_key:
            if loaded_name is not None:
                self.legacy._log(
                    f"[V18] Model swap: unloading '{loaded_name}', "
                    f"loading PEFT adapter '{adapter_id}' on '{base_model_id}'"
                )
            else:
                self.legacy._log(
                    f"[V18] Loading base model '{base_model_id}' + LoRA adapter '{adapter_id}'"
                )
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_id, trust_remote_code=True, cache_dir=self.legacy.HF_CACHE_DIR,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id, torch_dtype=torch.float16, device_map="auto",
                    trust_remote_code=True, use_safetensors=True,
                    attn_implementation="sdpa",
                    cache_dir=self.legacy.HF_CACHE_DIR,
                )
                model = PeftModel.from_pretrained(
                    base_model, adapter_id, cache_dir=self.legacy.HF_CACHE_DIR,
                )
                model.eval()
            except Exception as load_exc:
                raise RuntimeError(
                    f"[V18] PEFT model load failed: base='{base_model_id}', "
                    f"adapter='{adapter_id}'. Error: {load_exc}"
                ) from load_exc

            self.legacy._hf_tokenizer = tokenizer
            self.legacy._hf_model     = model
            self.legacy._hf_loaded_name = cache_key
            try:
                n_params = sum(p.numel() for p in model.parameters())
                first_param = next(model.parameters())
                self.legacy._log(
                    f"[V18] PEFT model loaded OK: adapter='{adapter_id}' | n_params={n_params:,} | "
                    f"device={first_param.device} | vocab_size={tokenizer.vocab_size}"
                )
            except Exception:
                self.legacy._log(f"[V18] PEFT model loaded (param count unavailable): '{adapter_id}'")
        else:
            self.legacy._log(f"[V18] Reusing cached PEFT model '{adapter_id}'")

        tokenizer = self.legacy._hf_tokenizer
        model     = self.legacy._hf_model

        messages = []
        if system_text and profile.use_system_prompt:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

        if getattr(tokenizer, "chat_template", None):
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                prompt_text = self._llama2_format(system_text, user_text)
        else:
            prompt_text = self._llama2_format(system_text, user_text)

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": profile.max_new_tokens,
                "do_sample": profile.do_sample,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if profile.do_sample:
                gen_kwargs["temperature"] = profile.temperature
            outputs = model.generate(**inputs, **gen_kwargs)
        output_tokens = outputs[0][input_len:]
        raw = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        self.legacy._log(
            f"[V18] PEFT generation done: input_tokens={input_len} | "
            f"output_tokens={len(output_tokens)} / max={profile.max_new_tokens} | "
            f"raw={repr(raw[:80])}"
        )
        return raw

    @staticmethod
    def _llama2_format(system_text: str, user_text: str) -> str:
        """Manual LLaMA-2 chat format fallback when chat_template is absent."""
        if system_text:
            return f"[INST] <<SYS>>\n{system_text}\n<</SYS>>\n\n{user_text} [/INST]"
        return f"[INST] {user_text} [/INST]"

    @staticmethod
    def _alpaca_format(system_text: str, user_text: str) -> str:
        """Alpaca instruction format for FinMA (trained on Alpaca-style data)."""
        instruction = f"{system_text}\n{user_text}" if system_text else user_text
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

    def _audit(
        self,
        model_name: str,
        profile: ModelProfile,
        task_id: str,
        system_text: str,
        user_text: str,
        raw: str,
        normalized: str,
    ) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "family": profile.family,
            "task_id": task_id,
            "system": system_text,
            "user": user_text,
            "raw_output": raw,
            "normalized_output": normalized,
        }
        with self._audit_lock:  # V16: thread-safe for parallel inference
            with self.audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
