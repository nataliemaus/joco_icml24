import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from joco.tasks.llm_tasks.utils.token_embedder import OPTEmbedding
from joco.tasks.llm_tasks.utils.utility_functions import (
    AngryToxicityLoss,
    CountLetterLoss,
    EmotionLoss,
    PerplexityLoss,
    ToxicityLoss,
)
from joco.tasks.objective import Objective

WORD_EMBEDDING_DIM = 768
import copy


class LLMObjective(Objective):
    """The goal of this task is to optimize prompts consisting
    of four words that, when passed into LLaMA-2, generate toxic text outputs"""

    def __init__(
        self,
        **kwargs,
    ):
        input_dim = self.get_default_input_dim()
        output_dim = self.get_default_output_dim()
        assert output_dim == 312
        assert input_dim % WORD_EMBEDDING_DIM == 0
        self.n_tokens = input_dim // WORD_EMBEDDING_DIM
        self.max_gen_length = 100
        self.save_all_gen_text = (
            True  # save all text generated during optimization locally
        )
        self.set_llm_name()  # must be defined in child class
        self.total_len_gen_seq = self.n_tokens + self.max_gen_length
        self.num_gen_seq = 3  # output_dim//self.max_gen_length
        self.set_task_specifc_vars()
        self.prepend_to_text = None
        self.language_model = self.get_language_model(
            max_gen_length=self.max_gen_length,
            n_tokens=self.n_tokens,
            seed_text=self.prepend_to_text,
            num_gen_seq=self.num_gen_seq,
        )
        self.language_model_vocab_tokens = self.language_model.get_vocab_tokens()
        self.token_embedder = OPTEmbedding(self.language_model_vocab_tokens)
        self.vocab = self.token_embedder.get_vocab()
        self.num_prompt_tokens = self.n_tokens
        if self.prepend_to_text is not None:
            self.num_prompt_tokens += len(self.prepend_to_text.split(" "))
        self.utility_fn = self.get_utility_fn(reward_type=self.reward_type)
        self.custom_y_compression_model = Y_Compression_Model(
            all_embeddings=copy.deepcopy(self.token_embedder.output_embeddings),
            y_hat_dim=32,
            num_gen_seq=self.num_gen_seq,
            total_len_gen_seq=self.total_len_gen_seq,
        ).cuda()
        self.default_budget = 2_000

        super().__init__(
            lb=None,
            ub=None,
            use_custom_y_compression_model=True,
            **kwargs,
        )
        self.set_save_data_path()

    def set_save_data_path(self):
        save_data_dir = f"{self.llm_name}_all_prompts_and_generated_text"
        if not os.path.exists(save_data_dir):
            os.mkdir(save_data_dir)
        self.save_data_path = save_data_dir + f"/run_id_{self.unique_run_id}.csv"

    def get_default_input_dim(self):
        return 3072

    def get_default_output_dim(self):
        return 312

    def set_task_specifc_vars(self):
        """Function sets the variable self.reward_type to a string specifying the reward/utility function we want to use (see optioins in self.get_utility_fn)

        Args:
            None

        Returns:
            None

        """
        self.reward_type = "angry-toxic"

    def prompts_to_texts(self, prompts):
        seed_text = self.prepend_to_text
        if seed_text is not None:
            prompts = [cur_prompt + " " + seed_text for cur_prompt in prompts]
        input_ids = self.language_model.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).input_ids.to(self.language_model.torch_device)
        gen_ids = self.language_model.model.generate(
            input_ids,
            max_length=self.language_model.max_gen_length,
            num_return_sequences=self.language_model.num_gen_seq,
            top_k=50,
            do_sample=True,
            trust_remote_code=True,
            pad_token_id=self.language_model.tokenizer.pad_token_id,
        )
        gen_texts = self.language_model.tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True,
        )
        generated_text = [
            (
                prompts[i],
                [
                    gen_texts[i * self.language_model.num_gen_seq + j][
                        len(prompts[i]) :
                    ]
                    for j in range(self.language_model.num_gen_seq)
                ],
            )
            for i in range(len(prompts))
        ]
        return generated_text

    def get_utility_fn(self, reward_type="perplexity"):
        if reward_type.startswith("count_letter"):
            # encourage text outputs with lots of the same letter repeated (easy toy task)
            letter = reward_type.split("_")[-1]  # i.e. count_letter_t
            return CountLetterLoss(letter=letter)
        elif reward_type.startswith("emotion"):
            # encourage text outputs with a particular emotional sentiment
            emotion_class = reward_type.split("_")[-1]  # i.e. emotion_anger
            assert emotion_class in [
                "anger",
                "joy",
                "sadness",
                "fear",
                "surprise",
                "disgust",
                "neutral",
            ]
            return EmotionLoss(emotion_class=emotion_class)
        elif reward_type == "toxicity":
            # encourage text outputs that are toxic
            return ToxicityLoss()
        elif reward_type == "perplexity":
            # encourage text outputs that are non-sensical
            return PerplexityLoss()
        elif reward_type == "angry-toxic":
            # encourage text outputs that are both angry and toxic
            #   (strictly optimizing toxcicity often leads to non-sensical toxic text)
            return AngryToxicityLoss()
        else:
            raise NotImplementedError

    def x_to_prompts(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float16)
        x = x.cuda()
        embeddings = x.reshape(-1, self.n_tokens, self.token_embedder.embed_dim)
        prompts = self.token_embedder.batched_embeddings_to_tokens(embeddings)
        # generated_text = self.prompts_to_texts(prompts)
        seed_text = self.prepend_to_text
        if seed_text is not None:
            prompts = [cur_prompt + " " + seed_text for cur_prompt in prompts]
        return prompts

    def x_to_y(self, x):
        prompts = self.x_to_prompts(x)
        input_ids = self.language_model.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).input_ids.to(self.language_model.torch_device)
        gen_ids = self.language_model.model.generate(
            input_ids,
            max_length=self.language_model.max_gen_length,
            num_return_sequences=self.language_model.num_gen_seq,
            top_k=50,
            do_sample=True,
            pad_token_id=self.language_model.tokenizer.pad_token_id,
        )  # (3, 104) = (n_gen_seq, max_gen_length+4)
        if gen_ids.shape[-1] < (self.max_gen_length + 4):
            more_padding = torch.tensor(
                [
                    [self.language_model.tokenizer.pad_token_id]
                    * ((self.max_gen_length + 4) - gen_ids.shape[-1])
                ]
                * 3
            )
            gen_ids = torch.cat((gen_ids, more_padding.cuda()), dim=-1)

        y = gen_ids.reshape(1, -1).float()  # torch.Size([1, 312])
        assert y.shape[-1] == self.output_dim
        self.num_calls += 1
        return y.cpu()

    def y_to_score(self, y):
        # Y to genertaed text:
        gen_ids = y.reshape(self.language_model.num_gen_seq, -1)
        # convert to nearest ints
        gen_ids = gen_ids.to(torch.int)  # I also think this needs to be bounded...
        gen_ids = torch.clamp(gen_ids, min=0, max=len(self.language_model_vocab_tokens))
        gen_texts = self.language_model.tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True,
        )
        prompts = []
        generated_text = []
        for gen_txt in gen_texts:
            gen_txt = gen_txt.split(" ")
            gen_txt_without_prompt = gen_txt[self.num_prompt_tokens :]
            gen_txt_without_prompt = " ".join(gen_txt_without_prompt)  #
            generated_text.append(gen_txt_without_prompt)
            # also grab prompt to save data
            prompt_only = gen_txt[0 : self.num_prompt_tokens]
            prompt_only = " ".join(prompt_only)  #
            prompts.append(prompt_only)

        rewards = self.utility_fn([[generated_text]])
        mean_reward = torch.mean(rewards, axis=1)

        # Save data
        if self.save_all_gen_text:  # saves generated text w/ associated prompt
            # repalce \n with random greek letter to make it easier to read
            weird_char = "β"
            save_prompts = np.array(
                [prompt.replace("\n", weird_char) for prompt in prompts]
            )
            save_gen_text = np.array(
                [gtext.replace("\n", weird_char) for gtext in generated_text]
            )
            save_rewards = np.array(rewards.squeeze().tolist())
            log_dict = {
                "prompt": save_prompts,
                "gen_text": save_gen_text,
                "reward": save_rewards,
            }
            df = pd.DataFrame.from_dict(log_dict)
            if os.path.exists(self.save_data_path):
                # append data to existing data file
                df.to_csv(self.save_data_path, mode="a", index=False, header=False)
            else:
                # create new data file
                df.to_csv(self.save_data_path, index=None)

        return mean_reward.item()


class Y_Compression_Model(nn.Module):
    def __init__(
        self,
        all_embeddings,
        y_hat_dim=32,
        num_gen_seq=3,
        total_len_gen_seq=104,
    ):
        super().__init__()
        self.y_hat_dim = y_hat_dim
        self.num_gen_seq = num_gen_seq
        self.all_embeddings = all_embeddings.cpu()
        self.hidden_dim = 32
        self.n_layers = 2
        self.fc1 = nn.Linear(WORD_EMBEDDING_DIM * num_gen_seq, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, y_hat_dim)

    def forward(self, x):
        with torch.no_grad():
            batch_seq_embeddings = []
            for i in range(x.shape[0]):
                y = x[i]
                gen_ids = y.reshape(self.num_gen_seq, -1)
                gen_ids = gen_ids.to(torch.int)
                gen_ids = torch.clamp(
                    gen_ids, min=0, max=self.all_embeddings.shape[0] - 1
                )  # torch.Size([3, 104])
                mean_sequence_embeddings = []
                for i in range(self.num_gen_seq):
                    gen_ids_single_seq = gen_ids[i]
                    embeddings = self.all_embeddings[
                        gen_ids_single_seq.tolist()
                    ]  # embeddings.shape = torch.Size([104, 768])
                    mean_sequence_embedding = embeddings.mean(0)  # torch.Size([768])
                    mean_sequence_embeddings.append(
                        mean_sequence_embedding.unsqueeze(0)
                    )
                mean_sequence_embeddings = torch.cat(
                    mean_sequence_embeddings, 0
                )  # torch.Size([3, 768])
                batch_seq_embeddings.append(
                    mean_sequence_embeddings.unsqueeze(0)
                )  # torch.Size([1, 3, 768])
            x = torch.cat(
                batch_seq_embeddings, 0
            )  # (bsz, num_gen_seq, embed_sz) # torch.Size([10, 3, 768])
            x = x.detach().cuda()
            batch_size = x.size(0)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        x = x.contiguous().view(
            batch_size, -1
        )  # torch.Size([10, 3*768]) = (bsz, 3*768)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation on final layer
        return x
