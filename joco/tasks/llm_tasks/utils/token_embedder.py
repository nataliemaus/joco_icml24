import abc

import torch
from joco.paths import PATH_TO_OPT_EMBEDDING_MODEL


class TokenEmbedding:
    def __init__(self, language_model_vocab_tokens):
        self.embed_dim = None
        self.language_model_vocab_tokens = language_model_vocab_tokens
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_embeddings = None
        self.sorted_vocab_keys = None  # a tensor of vocab tokens sorted by index
        self.all_vocab = None

    @abc.abstractmethod
    def set_embeddings(self):
        pass

    @abc.abstractmethod
    def batched_tokens_to_embeddings(self, batched_tokens):
        pass

    def batched_embeddings_to_tokens(self, batched_embeddings):
        """
        Given batched_embeddings, convert to tokens
        args:
            embeddings: (batch_size, num_tokens, embed_dim) word embedding
        returns:
            proj_tokens: (batch_size, num_tokens) projected tokens
        """
        batch_size = batched_embeddings.shape[0]
        proj_tokens = []
        for i in range(batch_size):
            subset_embedding = batched_embeddings[i, :, :]
            proj_tokens.append(self.proj_embeddings_to_tokens(subset_embedding))
        return proj_tokens

    def proj_embeddings_to_tokens(self, embeddings):
        distances = torch.norm(self.all_embeddings - embeddings.unsqueeze(1), dim=2)
        n_prompts = embeddings.shape[0]
        vocab_size = self.all_embeddings.shape[1]
        assert distances.shape == (
            n_prompts,
            vocab_size,
        ), f"Distances was shape {distances.shape} but should be  ({embeddings.shape[0]}, {self.embed_dim})"

        min_indices = torch.argmin(distances, axis=1).tolist()
        # SPACES BETWEEN EACH TOKEN
        proj_tokens = " ".join(self.sorted_vocab_keys[index] for index in min_indices)
        return proj_tokens

    def get_vocab(self):
        return self.sorted_vocab_keys

    def get_embeddings_mean_and_std(self):
        return (
            torch.mean(self.all_embeddings).item(),
            torch.std(self.all_embeddings).item(),
        )

    def get_embeddings_avg_dist(self):
        """
        Computes average Euclidean distance between embeddings
        """
        embeds = self.all_embeddings
        distances = torch.cdist(embeds, embeds)

        # Get the upper triangular part of the matrix, excluding the diagonal
        upper_triangular = torch.triu(distances, diagonal=1)

        # Compute the average pairwise distance
        avg_distance = (
            torch.sum(upper_triangular)
            / (upper_triangular.numel() - embeds.shape[0])
            * 2
        )
        return avg_distance.item()


class OPTEmbedding(TokenEmbedding):
    def __init__(self, language_model_vocab_tokens):
        super().__init__(language_model_vocab_tokens)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Can be changed to other models
        tokenizer = AutoTokenizer.from_pretrained(PATH_TO_OPT_EMBEDDING_MODEL)
        self.all_vocab = tokenizer.get_vocab()
        lm = AutoModelForCausalLM.from_pretrained(PATH_TO_OPT_EMBEDDING_MODEL).to(
            self.torch_device
        )
        self.all_embeddings = (
            lm.get_input_embeddings().weight
        )  # torch tensor 50272, 768 torch.Size([50272, 768])
        self.set_embeddings()  # self.all_embeddings has dim 1, 8687, 768 after unsqueezing torch.Size([1, 8687, 768])
        self.embed_dim = self.all_embeddings.shape[2]  # 768
        self.output_embeddings = (
            lm.get_output_embeddings().weight.detach()
        )  ## torch.Size([50272, 768])

    def set_embeddings(self):
        # Valid vocab is the intersection between all vocab and the language model vocab
        self.valid_vocab_keys = set(self.all_vocab.keys()).intersection(
            set(self.language_model_vocab_tokens)
        )
        # Filter the valid embeddings from the original embeddings tensor
        self.valid_indices = sorted(
            [self.all_vocab[key] for key in self.valid_vocab_keys]
        )
        self.all_embeddings = self.all_embeddings[self.valid_indices].unsqueeze(
            0
        )  # torch.Size([1, 8687, 768])
        # Create a new list with valid keys sorted by their indices
        self.sorted_vocab_keys = [
            key
            for key, value in sorted(self.all_vocab.items(), key=lambda item: item[1])
            if key in self.valid_vocab_keys
        ]

    def proj_embeddings_to_tokens(self, embeddings):
        proj_tokens = super().proj_embeddings_to_tokens(embeddings)
        return proj_tokens.replace("Ġ", " ")
