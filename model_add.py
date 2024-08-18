"""
model_add.py

This module defines a custom invocation layer for the Llama model using the llama-cpp library. It extends the PromptModelInvocationLayer class to provide additional functionality for handling prompts and generating text.

Classes:
    - LlamaCPPInvocationLayer: Custom invocation layer for the Llama model.

Modules:
    - haystack.nodes: Various nodes for processing and retrieving documents.
    - llama_cpp: Llama model library.
    - os: Operating system library.
    - typing: Type hinting library.
    - logging: Logging library.
"""

from haystack.nodes import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer import DefaultTokenStreamingHandler
from llama_cpp import Llama
import os
from typing import Dict, List, Union, Type, Optional

import logging 

logger = logging.getLogger(__name__)

class LlamaCPPInvocationLayer(PromptModelInvocationLayer):
    """
    Custom invocation layer for the Llama model using the llama-cpp library.

    Attributes:
        model_name_or_path (Union[str, os.PathLike]): The name or path of the underlying model.
        max_length (Optional[int]): Maximum length of the generated text.
        max_context (Optional[int]): Maximum context length for the model.
        n_parts (Optional[int]): Number of parts for the model.
        seed (Optional[int]): Random seed for reproducibility.
        f16_kv (Optional[bool]): Use 16-bit floating point for key-value pairs.
        logits_all (Optional[bool]): Return logits for all tokens.
        vocab_only (Optional[bool]): Only load the vocabulary.
        use_mmap (Optional[bool]): Use memory-mapped files.
        use_mlock (Optional[bool]): Use memory locking.
        embedding (Optional[bool]): Enable embedding mode.
        n_threads (Optional[int]): Number of threads to use.
        n_batch (Optional[int]): Batch size for processing.
        last_n_tokens_size (Optional[int]): Size of the last n tokens.
        lora_base (Optional[str]): Base path for LoRA.
        lora_path (Optional[str]): Path to the LoRA model.
        verbose (Optional[bool]): Enable verbose logging.
        model (Model): The Llama model instance.
    """

    def __init__(self, model_name_or_path: Union[str, os.PathLike],
                 max_length: Optional[int] = 128,
                 max_context: Optional[int] = 2048,
                 n_parts: Optional[int] = -1,
                 seed: Optional[int] = 1337,
                 f16_kv: Optional[bool] = True,
                 logits_all: Optional[bool] = False,
                 vocab_only: Optional[bool] = False,
                 use_mmap: Optional[bool] = True,
                 use_mlock: Optional[bool] = False,
                 embedding: Optional[bool] = False,
                 n_threads: Optional[int] = None,
                 n_batch: Optional[int] = 512,
                 last_n_tokens_size: Optional[int] = 64,
                 lora_base: Optional[str] = None,
                 lora_path: Optional[str] = None,
                 verbose: Optional[bool] = True,
                 **kwargs):
        """
        Initializes a new instance of the LlamaCPPInvocationLayer class.

        Args:
            model_name_or_path (Union[str, os.PathLike]): The name or path of the underlying model.
            max_length (Optional[int]): Maximum length of the generated text.
            max_context (Optional[int]): Maximum context length for the model.
            n_parts (Optional[int]): Number of parts for the model.
            seed (Optional[int]): Random seed for reproducibility.
            f16_kv (Optional[bool]): Use 16-bit floating point for key-value pairs.
            logits_all (Optional[bool]): Return logits for all tokens.
            vocab_only (Optional[bool]): Only load the vocabulary.
            use_mmap (Optional[bool]): Use memory-mapped files.
            use_mlock (Optional[bool]): Use memory locking.
            embedding (Optional[bool]): Enable embedding mode.
            n_threads (Optional[int]): Number of threads to use.
            n_batch (Optional[int]): Batch size for processing.
            last_n_tokens_size (Optional[int]): Size of the last n tokens.
            lora_base (Optional[str]): Base path for LoRA.
            lora_path (Optional[str]): Path to the LoRA model.
            verbose (Optional[bool]): Enable verbose logging.
            kwargs: Additional keyword arguments.
        """
        if model_name_or_path is None or len(model_name_or_path) == 0:
            raise ValueError("model_name_or_path cannot be None or empty string")

        self.model_name_or_path = model_name_or_path
        self.max_context = max_context
        self.max_length = max_length
        self.n_parts = n_parts
        self.seed = seed
        self.f16_kv = f16_kv
        self.logits_all = logits_all
        self.vocab_only = vocab_only
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.embedding = embedding
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.last_n_tokens_size = last_n_tokens_size
        self.lora_base = lora_base
        self.lora_path = lora_path
        self.verbose = verbose
        self.model: Model = Llama(model_path=model_name_or_path,
                                  n_ctx=max_context,
                                  n_parts=n_parts,
                                  seed=seed,
                                  f16_kv=f16_kv,
                                  logits_all=logits_all,
                                  vocab_only=vocab_only,
                                  use_mmap=use_mmap,
                                  use_mlock=use_mlock,
                                  embedding=embedding,
                                  n_threads=n_threads,
                                  n_batch=n_batch,
                                  last_n_tokens_size=last_n_tokens_size,
                                  lora_base=lora_base,
                                  lora_path=lora_path,
                                  verbose=verbose)

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """
        Ensures that the length of the prompt and answer is within the maximum token length of the PromptModel.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): Prompt text to be sent to the generative model.

        Returns:
            Union[str, List[Dict[str, str]]]: The truncated prompt if necessary.
        """
        if not isinstance(prompt, str):
            raise ValueError(f"Prompt must be of type str but got {type(prompt)}")

        context_length = self.model.n_ctx()
        tokenized_prompt = self.model.tokenize(bytes(prompt, 'utf-8'))
        if len(tokenized_prompt) + self.max_length > context_length:
            logger.warning(
                "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
                "answer length (%s tokens) fit within the max token limit (%s tokens). "
                "Shorten the prompt to prevent it from being cut off",
                len(tokenized_prompt),
                max(0, context_length - self.max_length),
                self.max_length,
                context_length,
            )
            return bytes.decode(self.model.detokenize(tokenized_prompt[:max(0, context_length - self.max_length)]), 'utf-8')

        return prompt

    def invoke(self, *args, **kwargs):
        """
        Takes a prompt and returns a list of generated text using the underlying model.

        Args:
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            List[str]: A list of generated text.
        """
        output: List[Dict[str, str]] = []
        stream = kwargs.pop("stream", False)

        generated_texts = []

        if kwargs and "prompt" in kwargs:
            prompt = kwargs.pop("prompt")

            # For more details refer to call documentation for Llama CPP https://abetlen.github.io/llama-cpp-python/#llama_cpp.llama.Llama.__call__
            model_input_kwargs = {
                key: kwargs[key]
                for key in [
                    "suffix",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "logprobs",
                    "echo",
                    "repeat_penalty",
                    "top_k",
                    "stop"
                ]
                if key in kwargs
            }

        if stream:
            for token in self.model(prompt, stream=True, **model_input_kwargs):
                generated_texts.append(token['choices'][0]['text'])
        else:
            output = self.model(prompt, **model_input_kwargs)
            generated_texts = [o['text'] for o in output['choices']]
        return generated_texts

    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Checks if the given model is supported by this invocation layer.

        Args:
            model_name_or_path (str): The name or path of the model.
            kwargs: Additional keyword arguments.

        Returns:
            bool: True if this invocation layer supports the model, False otherwise.
        """
        return model_name_or_path is not None and len(model_name_or_path) > 0