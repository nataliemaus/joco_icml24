import abc

import torch
from joco.paths import PATH_TO_EMOTION_MODEL, PATH_TO_GPT2, PATH_TO_TOXIC_COMMENT_MODEL
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    pipeline,
    TextClassificationPipeline,
)


class LossFunction:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_num = 0 if torch.cuda.is_available() else -1

    @abc.abstractmethod
    def __call__(self, generated_texts, **kwargs) -> torch.Tensor:
        """
        Compute the loss on the generated text and prompts
        """
        pass


class CountLetterLoss(LossFunction):
    def __init__(self, letter="t"):
        self.letter = letter

    def __call__(self, generated_texts) -> torch.Tensor:
        n_prompts = len(generated_texts)
        # n_gen_texts = len(generated_texts[0][1])
        losses = torch.zeros((n_prompts, -1))
        for i in range(n_prompts):
            # prompt, cur_generated_texts = generated_texts[i]
            if len(generated_texts[i]) == 2:
                _, cur_generated_texts = generated_texts[i]
            elif len(generated_texts[i]) == 1:
                cur_generated_texts = generated_texts[i]
            else:
                assert 0
            losses[i, :] = torch.tensor(
                [text.lower().count(self.letter) for text in cur_generated_texts]
            )
        return losses


class ToxicityLoss(LossFunction):
    def __init__(self):
        super().__init__()
        self.max_allowed_input_length = 512
        # https://huggingface.co/martin-ha/toxic-comment-model
        self.tokenizer = AutoTokenizer.from_pretrained(PATH_TO_TOXIC_COMMENT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            PATH_TO_TOXIC_COMMENT_MODEL
        )
        self.pipeline = TextClassificationPipeline(
            model=model, tokenizer=self.tokenizer, device=self.device_num
        )

    def string_to_loss(self, input_string: str) -> torch.Tensor:
        # WIP: could possibly use this to simplify things later...
        toxicity_pred = self.pipeline([input_string])[0]
        prob_toxic = (
            toxicity_pred["score"]
            if toxicity_pred["label"] == "toxic"
            else 1 - toxicity_pred["score"]
        )
        return prob_toxic

    def __call__(self, generated_texts, return_shortened_text=False) -> torch.Tensor:
        n_prompts = len(generated_texts)
        # n_gen_texts = len(generated_texts[0][1])
        preprocessed_text = []  #
        for i in range(n_prompts):
            # prompt, cur_generated_texts = generated_texts[i]
            if len(generated_texts[i]) == 2:
                _, cur_generated_texts = generated_texts[i]
                flag_passed_in_text_only = False
            elif len(generated_texts[i]) == 1:
                cur_generated_texts = generated_texts[i]
                flag_passed_in_text_only = True  # New version
            else:
                assert 0
            preprocessed_text = preprocessed_text + cur_generated_texts
        if flag_passed_in_text_only:  # New v, always true during opt
            # print("Passing into pipeline:", preprocessed_text[0]) ## preprocessed_text[0] of the form [ "string one", "string two", "string three"]
            shortened_text = []
            input_text = preprocessed_text[0]
            for string1 in input_text:
                decoded_text = self.tokenizer(string1)["input_ids"]
                while (
                    len(decoded_text) > self.max_allowed_input_length
                ):  # while too long, remove characters
                    string1 = string1[0:-1]
                    decoded_text = self.tokenizer(string1)["input_ids"]
                shortened_text.append(string1)
            # print("PASSING IN SHORTENED TEXT:", shortened_text)
            # toxicity_preds = self.pipeline(preprocessed_text[0]) ## Old v, got errors w/ too long strings
            toxicity_preds = self.pipeline(shortened_text)
        else:  # old v used tuple ... (from og code)
            assert not return_shortened_text
            toxicity_preds = self.pipeline(preprocessed_text)

        losses = [
            pred["score"] if pred["label"] == "toxic" else 1 - pred["score"]
            for pred in toxicity_preds
        ]
        # losses = torch.tensor(losses).reshape(n_prompts, n_gen_texts)
        losses = torch.tensor(losses).reshape(n_prompts, -1)

        if return_shortened_text:
            return losses, shortened_text
        return losses


class EmotionLoss(LossFunction):
    def __init__(self, emotion_class):
        super().__init__()
        self.emotion_class = emotion_class
        assert emotion_class in [
            "anger",
            "joy",
            "sadness",
            "fear",
            "surprise",
            "disgust",
            "neutral",
        ]
        self.classifier = pipeline(
            "text-classification",
            model=PATH_TO_EMOTION_MODEL,
            top_k=None,
            device=self.device_num,
        )

    def __call__(self, generated_texts) -> torch.Tensor:
        n_prompts = len(generated_texts)
        # n_gen_texts = len(generated_texts[0][1])
        preprocessed_text = []
        for i in range(n_prompts):
            # prompt, cur_generated_texts = generated_texts[i]
            if len(generated_texts[i]) == 2:
                # Old setup when passed in tuple including prompt
                _, cur_generated_texts = generated_texts[i]
                flag_passed_in_text_only = False
            elif len(generated_texts[i]) == 1:
                cur_generated_texts = generated_texts[i]
                flag_passed_in_text_only = True  # New version
            else:
                assert 0
            preprocessed_text = preprocessed_text + cur_generated_texts

        if flag_passed_in_text_only:  # New version
            # preprocessed_text[0] of form [ "string one", "string two", "string three"]
            emotion_preds = self.classifier(preprocessed_text[0])
        else:
            emotion_preds = self.classifier(preprocessed_text)

        # The first entry is the 'anger' emotion class we care about, the order is:
        # anger, disgust, fear, joy, neutral, sadness, surprise
        losses = [
            pred[i]["score"]
            for pred in emotion_preds
            for i in range(len(pred))
            if pred[i]["label"] == self.emotion_class
        ]
        losses = torch.tensor(losses).reshape(n_prompts, -1)
        return losses


class PerplexityLoss(LossFunction):
    def __init__(self):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(PATH_TO_GPT2)
        self.model = GPT2LMHeadModel.from_pretrained(PATH_TO_GPT2).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def __call__(self, generated_texts):
        n_prompts = len(generated_texts)
        # n_gen_texts = len(generated_texts[0][1])
        preprocessed_text = []
        for i in range(n_prompts):
            # prompt, cur_generated_texts = generated_texts[i]
            if len(generated_texts[i]) == 2:
                _, cur_generated_texts = generated_texts[i]
                flag_passed_in_text_only = False
            elif len(generated_texts[i]) == 1:
                cur_generated_texts = generated_texts[i]
                flag_passed_in_text_only = True
            else:
                assert 0
            preprocessed_text = preprocessed_text + cur_generated_texts
        with torch.no_grad():
            if flag_passed_in_text_only:
                inputs = self.tokenizer(
                    preprocessed_text[0],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
            else:
                inputs = self.tokenizer(
                    preprocessed_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
            inputs.to(self.device)
            outputs = self.model(**inputs)
            log_probs = outputs.logits.log_softmax(dim=-1)
            token_log_probs = torch.gather(
                log_probs, -1, inputs.input_ids.unsqueeze(-1)
            ).squeeze(-1)

            masked_indices = inputs.attention_mask.to(bool)
            token_log_probs_masked = token_log_probs[masked_indices]

            sequence_lengths = inputs.attention_mask.sum(dim=-1)
            sentence_log_probs = token_log_probs_masked.split(sequence_lengths.tolist())
            batch_log_probs = [
                sentence_log_probs[i].sum().item()
                for i in range(len(sentence_log_probs))
            ]
            batch_lengths = [
                len(sentence_log_probs[i]) for i in range(len(sentence_log_probs))
            ]
            # Log perplexity
            batch_perplexities = [
                (-log_prob / length) if length > 0 else 0
                for log_prob, length in zip(batch_log_probs, batch_lengths)
            ]

        losses = torch.tensor(batch_perplexities).reshape(n_prompts, -1)
        return losses


class AngryToxicityLoss(LossFunction):
    def __init__(self):
        super().__init__()
        self.toxcicity_fn = ToxicityLoss()
        self.anger_fn = EmotionLoss("anger")
        # Much better than Toxcicity loss becasue emotion classifier constraints text to be sensical

    def __call__(self, generated_texts) -> torch.Tensor:
        probs_toxic, shortened_text = self.toxcicity_fn(
            generated_texts, return_shortened_text=True
        )  # 0-1 scale
        # probs_angry = self.anger_fn(generated_texts)  # 0-1 scale
        try:
            probs_angry = self.anger_fn([[shortened_text]])  # 0-1 scale
        except RuntimeError:
            # Catch string length error (happens very rarely)
            probs_angry = 0.0
        # min possible = 0, max possible = 1, want to maximize!
        losses = probs_toxic * probs_angry
        return losses
