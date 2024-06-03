import abc
from typing import List

import torch

from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LanguageModel:
    def __init__(self, max_gen_length, num_gen_seq):
        self.max_gen_length = max_gen_length
        self.num_gen_seq = num_gen_seq
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_num = 0 if torch.cuda.is_available() else -1

    @abc.abstractmethod
    def load_model(self, **kwargs):
        """Load the language model."""
        pass

    def generate_text(self, prompts: List[str], seed_text=None) -> List[str]:
        """Generate text given a prompts."""
        if seed_text is not None:
            prompts = [cur_prompt + " " + seed_text for cur_prompt in prompts]
        return self._generate(prompts)

    @abc.abstractmethod
    def _generate(self, prompts: List[str]) -> List[str]:
        pass

    @abc.abstractmethod
    def get_vocab_tokens(self) -> List[str]:
        """
        Returns the vocabulary of the language model
        Necessary so that the tokenizer only searches over valid tokens
        """
        pass


# https://github.com/DebugML/adversarial_prompting/blob/master/utils/text_utils/language_model.py
class HuggingFaceLanguageModel(LanguageModel):
    def load_model(self, model_name: str, path_to_model: str):
        if model_name == "gpt2":
            from transformers import (
                GPT2LMHeadModel as ModelClass,
                GPT2Tokenizer as TokenizerClass,
            )
        else:
            from transformers import (
                AutoModelForCausalLM as ModelClass,
                AutoTokenizer as TokenizerClass,
            )
        self.model_name = path_to_model
        self.generator = pipeline(
            "text-generation",
            model=path_to_model,
            device=self.device_num,
            trust_remote_code=True,
        )
        self.tokenizer = TokenizerClass.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = ModelClass.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(device)

    def _generate(self, prompts: List[str]) -> List[str]:
        gen_texts = self.generator(
            prompts,
            max_length=self.max_gen_length,
            num_return_sequences=self.num_gen_seq,
            top_k=50,
            do_sample=True,
        )
        gen_texts = [
            (
                prompts[i],
                [cur_dict["generated_text"][len(prompts[i]) :] for cur_dict in gen],
            )
            for i, gen in enumerate(gen_texts)
        ]
        return gen_texts

    def get_vocab_tokens(self):

        return list(self.tokenizer.get_vocab().keys())
