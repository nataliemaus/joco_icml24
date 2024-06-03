import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from joco.paths import (
    PATH_TO_CLIP_TOKENIZER,
    PATH_TO_IMAGENET_DATA,
    PATH_TO_STABLE_DIFFUSION_MODEL,
)
from joco.tasks.objective import Objective
from joco.tasks.stable_diffusion_tasks.imagenet_utils import (
    get_imagenet_sub_classes,
    load_imagenet,
)
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


class StableDiffusionObjective(Objective):
    def __init__(
        self,
        **kwargs,
    ):
        self.default_budget = 1_000
        self.set_task_specific_vars()
        self.exclude_high_similarity_tokens = True
        self.word_embedding_dim = 768
        input_dim = self.get_default_input_dim()
        output_dim = self.get_default_output_dim()
        assert input_dim % self.word_embedding_dim == 0
        self.n_tokens = input_dim // self.word_embedding_dim
        total_img_dim = 3 * 224 * 224
        assert output_dim % total_img_dim == 0
        self.avg_over_N_latents = output_dim // total_img_dim
        self.custom_y_compression_model = Y_Compression_Model(
            y_hat_dim=32,
            avg_over_N_latents=self.avg_over_N_latents,
        )
        project_back = True
        minimize = False  # reward now not loss so maximize reward :)
        batch_size = 10
        num_inference_steps = 25
        seed = 0
        use_fixed_latents = False
        similar_token_threshold = -3.0

        self.optimal_sub_classes = get_imagenet_sub_classes(self.optimal_class)

        if self.avoid_class is not None:
            self.avoid_sub_classes = get_imagenet_sub_classes(self.avoid_class)
        else:
            self.avoid_sub_classes = []
        assert len(self.optimal_sub_classes) > 0

        self.N_extra_prepend_tokens = len(self.prepend_to_text.split())
        if self.prepend_to_text:
            assert project_back
        self.project_back = project_back
        self.minimize = minimize
        self.batch_size = batch_size
        self.height = 512  # default height of Stable Diffusion
        self.width = 512  # default width of Stable Diffusion
        self.num_inference_steps = num_inference_steps  # Number of denoising steps, this value is decreased to speed up generation
        self.guidance_scale = 7.5  # Scale for classifier-free guidance
        self.generator = torch.manual_seed(
            seed
        )  # Seed generator to create the inital latent noise
        self.max_num_tokens = (
            self.n_tokens + self.N_extra_prepend_tokens
        ) * 2  # maximum number of tokens in input, at most 75 and at least 1
        self.dtype = torch.float16
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. The resnet18 model for classifying images into imagenet classes
        # self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet18 = tmodels.resnet18(pretrained=True)
        # self.resnet18 = torch.load()
        self.resnet18.eval()
        self.resnet18.to(self.torch_device)

        # 2. The UNet model for generating the latents.
        # self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=self.dtype, revision="fp16", use_auth_token=HUGGING_FACE_TOKEN)
        self.unet = UNet2DConditionModel.from_pretrained(
            PATH_TO_STABLE_DIFFUSION_MODEL,
            subfolder="unet",
            torch_dtype=self.dtype,
            revision="fp16",
        )

        # 3. Load the autoencoder model which will be used to decode the latents into image space.
        # self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=self.dtype, revision="fp16", use_auth_token=HUGGING_FACE_TOKEN)
        self.vae = AutoencoderKL.from_pretrained(
            PATH_TO_STABLE_DIFFUSION_MODEL,
            subfolder="vae",
            torch_dtype=self.dtype,
            revision="fp16",
        )

        # 4. Load the tokenizer and text encoder to tokenize and encode the text.
        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=self.dtype)
        # self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=self.dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            PATH_TO_CLIP_TOKENIZER,
            torch_dtype=self.dtype,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            PATH_TO_CLIP_TOKENIZER,
            torch_dtype=self.dtype,
        )
        self.text_model = self.text_encoder.text_model
        self.vae = self.vae.to(self.torch_device)
        self.text_encoder = self.text_encoder.to(self.torch_device)
        self.text_model = self.text_model.to(self.torch_device)
        self.unet = self.unet.to(self.torch_device)

        # Scheduler for noise in image
        self.scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            skip_prk_steps=True,
            steps_offset=1,
        )
        self.scheduler.set_timesteps(self.num_inference_steps)

        # For reading the imagenet classes:
        # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
        self.preprocess_img = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.word_embedder = self.text_encoder.get_input_embeddings()
        self.uncond_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.max_num_tokens + 2,
            return_tensors="pt",
        )
        with torch.no_grad():
            self.uncond_embed = self.word_embedder(
                self.uncond_input.input_ids.to(self.torch_device)
            )

        if use_fixed_latents:
            self.fixed_latents = torch.randn(
                (1, 4, self.height // 8, self.width // 8),
                generator=self.generator,
                dtype=self.dtype,
            ).to(self.torch_device)
            self.fixed_latents = self.fixed_latents.repeat(self.batch_size, 1, 1, 1)
        else:
            self.fixed_latents = None

        self.vocab = self.tokenizer.get_vocab()
        self.reverse_vocab = {self.vocab[k]: k for k in self.vocab.keys()}

        if self.exclude_high_similarity_tokens:
            path = (
                PATH_TO_IMAGENET_DATA
                + f"/high_similarity_tokens/{self.optimal_class}_high_similarity_tokens.csv"
            )
            df = pd.read_csv(path)
            tokens = df["token"].values
            losses = df["loss"].values
            self.related_vocab = tokens[losses >= similar_token_threshold].tolist()
            self.all_token_idxs = self.get_non_related_values()
        else:
            self.all_token_idxs = list(self.vocab.values())
            self.related_vocab = []
        self.all_token_embeddings = self.word_embedder(
            torch.tensor(self.all_token_idxs).to(self.torch_device)
        )
        self.imagenet_class_to_ix, self.ix_to_imagenet_class = load_imagenet()
        self.optimal_class_idxs = []
        for clss in self.optimal_sub_classes:
            class_ix = self.imagenet_class_to_ix[clss]
            self.optimal_class_idxs.append(class_ix)
        self.avoid_class_idxs = []
        for clss in self.avoid_sub_classes:
            class_ix = self.imagenet_class_to_ix[clss]
            self.avoid_class_idxs.append(class_ix)

        # Define valid input and output types for pipeline
        self.valid_input_types = ["prompt", "word_embedding", "CLIP_embedding", "image"]
        self.valid_output_types = [
            "tokens",
            "word_embedding",
            "processed_word_embedding",
            "CLIP_embedding",
            "image",
            "reward",
        ]

        super().__init__(
            lb=None,
            ub=None,
            use_custom_y_compression_model=True,
            **kwargs,
        )

    def get_default_input_dim(self):
        return 3072

    def get_default_output_dim(self):
        return 451584

    def set_task_specific_vars(self):
        """Method sets the following three variables for the specific
        prompt optimization task
        1. self.optimal_class: string specifying the optimal/target imagenet class
            (we seek to find prompts that generate images of this class)
        2. self.avoid_class: string specifying the imagenet class that we do not want to generate images of, use None if irrelevant
        3.self.prepend_to_text: string specifying text that we prepend all prompts to, if not a prepending task use empty string
        """
        raise NotImplementedError(
            "Must implement set_task_specific_vars() specific to desired stable diffusion prompt optimization task"
        )

    def get_non_related_values(self):
        tmp = []
        for word in self.related_vocab:
            if type(word) == str:
                tmp.append(word)
            else:
                tmp.append(str(word))
            # tmp.append(word+'</w>')
        self.related_vocab = tmp
        non_related_values = []
        for key in self.vocab.keys():
            if not ((key in self.related_vocab) or (self.optimal_class in key)):
                non_related_values.append(self.vocab[key])
        return non_related_values

    def prompt_to_token(self, prompt):
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_num_tokens + 2,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.torch_device)
        return tokens

    def tokens_to_word_embed(self, tokens):
        with torch.no_grad():
            word_embed = self.word_embedder(tokens)
        return word_embed

    """
        Preprocesses word embeddings
        word_embeddings can have max_num_tokens or max_num_tokens + 2, either with or without the
        start-of-sentence and end-of-sentence tokens
        Will manually concatenate the correct SOS and EOS word embeddings if missing

        In the setting where word_embed is fed from tokens_to_word_embed, the middle dimension will be
        max_num_tokens + 2

        For manual optimization, suffices to only use dimension max_num_tokens to avoid redundancy

        args:
            word_embed: dtype pytorch tensor shape (batch_size, max_num_tokens, 768)
                        or (batch_size, max_num_tokens+2, 768)
        returns:
            proc_word_embed: dtype pytorch tensor shape (batch_size, max_num_tokens+2, 768)
    """

    def preprocess_word_embed(self, word_embed):

        # The first token dim is the start of text token and
        # the last token dim is the end of text token
        # if word_embed is manually generated and missing these, we manually add it
        if word_embed.shape[1:] == (self.max_num_tokens, 768):
            batch_size = word_embed.shape[0]
            rep_uncond_embed = self.uncond_embed.repeat(batch_size, 1, 1)
            word_embed = torch.cat(
                [rep_uncond_embed[:, 0:1, :], word_embed, rep_uncond_embed[:, -1:, :]],
                dim=1,
            )

        return word_embed

    def build_causal_attention_mask(self, bsz, seq_len, dtype):
        # Helper function to build the causal attention mask when using the CLIP Model
        # Fix from: https://github.com/fastai/diffusion-nbs/issues/37?fbclid=IwAR3W4PaUd9S18reAWP8UfwegZf5E8wgZComHGUVcMOuxeJxBycBrpR7c0X4
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

    """
        Modified from https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/clip/modeling_clip.py#L611
        args:
            proc_word_embed: dtype pytorch tensor shape (2, batch_size, max_num_tokens+2, 768)
        returns:
            CLIP_embed: dtype pytorch tensor shape (2, batch_size, max_num_tokens+2, 768)
    """

    def preprocessed_to_CLIP(self, proc_word_embed):

        # Hidden state from word embedding
        hidden_states = self.text_model.embeddings(inputs_embeds=proc_word_embed)

        attention_mask = None
        output_attentions = None
        output_hidden_states = None
        return_dict = None
        bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324

        causal_attention_mask = self.build_causal_attention_mask(
            bsz, seq_len, hidden_states.dtype
        )
        # Old code only works with older version of transformers (pip install -q --upgrade transformers==4.25.1 diffusers ftfy)
        # causal_attention_mask = self.text_model._build_causal_attention_mask(
        #     bsz, seq_len, hidden_states.dtype
        # )
        causal_attention_mask = causal_attention_mask.to(hidden_states.device)
        with torch.no_grad():
            encoder_outputs = self.text_model.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        CLIP_embed = last_hidden_state
        return CLIP_embed

    """
        Generates images from CLIP embeddings using stable diffusion
        args:
            clip_embed: dtype pytorch tensor shape (batch_size, max_num_tokens + 2, 768)
        returns:
            images: array of PIL images

    """

    def CLIP_embed_to_image(self, clip_embed, fixed_latents=None):
        batch_size = clip_embed.shape[0]
        rep_uncond_embed = self.uncond_embed.repeat(batch_size, 1, 1)

        # Concat unconditional and text embeddings, used for classifier-free guidance
        clip_embed = torch.cat([rep_uncond_embed, clip_embed])
        if fixed_latents is not None:
            assert fixed_latents.shape == (
                batch_size,
                self.unet.in_channels,
                self.height // 8,
                self.width // 8,
            )
            latents = fixed_latents
        else:
            # Generate initial random noise
            latents = torch.randn(
                (batch_size, self.unet.in_channels, self.height // 8, self.width // 8),
                generator=self.generator,
                dtype=self.dtype,
            ).to(self.torch_device)
        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            skip_prk_steps=True,
            steps_offset=1,
        )

        scheduler.set_timesteps(self.num_inference_steps)
        latents = latents * scheduler.init_noise_sigma
        # Diffusion process
        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=clip_embed
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        # Use vae to decode latents into image
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1)

        # ## TODO: don't convert here, convert later!! ...
        # image = image.numpy()
        # images = (image * 255).round().astype("uint8")
        # pil_images = [Image.fromarray(image) for image in images]

        # return pil_images

        ## ADDED TO CONVERT HERE:
        image = image.numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        self.curr_pil_images = pil_images
        input_tensors = []
        for img in pil_images:  # for img in imgs
            input_tensors.append(self.preprocess_img(img))
        image_tensors = torch.stack(
            input_tensors
        )  # input_tensors = torch.stack(input_tensors)

        return image_tensors

    """
        Uses resnet18 as a cat classifier
        Our loss is the negative log probability of the class of cats
        Loss will be small when stable diffusion generates images of cats
        reward is the positive log probability of the target/ optimal class!
    """
    # def image_to_loss(self, img_tensor):
    # def image_to_loss(self, imgs):
    def image_to_reward(self, image_tensors):
        input_batch = image_tensors.to(self.torch_device)
        with torch.no_grad():
            output = self.resnet18(input_batch)
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output, dim=1)
        # probabilities.shape = (bsz,1000) = (bsz,n_imagenet_classes)
        most_probable_classes = torch.argmax(probabilities, dim=-1)  # (bsz,)
        self.most_probable_classes_batch = [
            self.ix_to_imagenet_class[ix.item()] for ix in most_probable_classes
        ]
        # prob of max prob class of all optimal class options!
        total_probs = torch.max(
            probabilities[:, self.optimal_class_idxs], dim=1
        ).values  # classes 281:282 are HOUSE cat classes
        if len(self.avoid_class_idxs) > 0:
            probs_avoid = torch.max(
                probabilities[:, self.avoid_class_idxs], dim=1
            ).values
            total_probs = total_probs * (1 - probs_avoid)

        rewards = torch.log(total_probs)
        # total_cat_probs = torch.max(probabilities[:,281:286], dim = 1).values # classes 281:286 are cat classes
        # total_dog_probs = torch.sum(probabilities[:,151:268], dim = 1) # classes 151:268 are dog classes
        # p_dog = total_dog_probs / (total_cat_sprobs + total_dog_probs)

        return rewards

    """
        Pipeline order
        prompt -> tokens -> word_embedding -> processed_word_embedding ->
        CLIP_embedding -> image -> loss (reward)

        Function that accepts intermediary values in the pipeline and outputs downstream values

        input_type options: ['prompt', 'word_embedding', 'CLIP_embedding', 'image']
        output_type options: ['tokens', 'word_embedding', 'processed_word_embedding',
                                'CLIP_embedding', 'image', 'reward']
    """

    def pipeline(self, input_type, input_value, output_types, fixed_latents=None):
        if input_type not in self.valid_input_types:
            raise ValueError(
                f"input_type must be one of {self.valid_input_types} but was {input_type}"
            )
        for cur_output_type in output_types:
            if cur_output_type not in self.valid_output_types:
                raise ValueError(
                    f"output_type must be one of {self.valid_output_types} but was {cur_output_type}"
                )
        # Check that output is downstream
        pipeline_order = [
            "prompt",
            "tokens",
            "word_embedding",
            "processed_word_embedding",
            "CLIP_embedding",
            "image",
            "reward",
        ]
        pipeline_maps = {
            "prompt": self.prompt_to_token,
            "tokens": self.tokens_to_word_embed,
            "word_embedding": self.preprocess_word_embed,
            "processed_word_embedding": self.preprocessed_to_CLIP,
            "CLIP_embedding": self.CLIP_embed_to_image,
            "image": self.image_to_reward,
        }

        start_index = pipeline_order.index(input_type)
        max_end_index = start_index
        for cur_output_type in output_types:
            cur_end_index = pipeline_order.index(cur_output_type)
            if start_index >= cur_end_index:
                raise ValueError(f"{output_types} is not downstream of {input_type}.")
            else:
                max_end_index = max(max_end_index, cur_end_index)

        cur_pipe_val = input_value
        output_dict = {}
        for i in range(start_index, max_end_index):
            cur_type = pipeline_order[i]
            mapping = pipeline_maps[cur_type]
            if cur_type == "CLIP_embedding":
                cur_pipe_val = mapping(cur_pipe_val, fixed_latents=fixed_latents)
            else:
                cur_pipe_val = mapping(cur_pipe_val)
            next_type = pipeline_order[i + 1]
            if next_type in output_types:
                output_dict[next_type] = cur_pipe_val
        return output_dict

    def x_to_y(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float16)
        x = x.cuda()
        x = x.reshape(-1, self.n_tokens, self.word_embedding_dim)
        out_types = ["image"]
        input_type = "word_embedding"
        if self.project_back:
            x = self.proj_word_embedding(x)
            input_type = "prompt"
            x = [x1[0] for x1 in x]

        assert self.fixed_latents is None  # not using fixed latents...
        imgs_per_latent = (
            []
        )  ## N latents x bsz : [ [latent 0, bsz imgs ], [latent 1, bsz imgs], ..., [latent N bsz imgs]]
        for _ in range(self.avg_over_N_latents):
            out_dict = self.pipeline(
                input_type=input_type,
                input_value=x,
                output_types=out_types,
                fixed_latents=self.fixed_latents,
            )
            imgs = out_dict["image"]
            imgs_per_latent.append(imgs)
        imgs_from_promt = torch.cat(imgs_per_latent)  # torch.Size([5, 3, 224, 224])
        imgs_from_promt = imgs_from_promt.reshape(1, -1)  # torch.Size([1, 752640])
        self.num_calls += 1
        return imgs_from_promt

    def y_to_score(self, y):
        assert self.fixed_latents is None
        out_types = ["reward"]
        input_type = "image"
        imgs_from_promt = y.reshape(self.avg_over_N_latents, 3, 224, 224)

        out_dict = self.pipeline(
            input_type=input_type,
            input_value=imgs_from_promt,
            output_types=out_types,
            fixed_latents=self.fixed_latents,
        )
        probs_optimal_class = out_dict["reward"]
        reward = probs_optimal_class.mean()
        if self.minimize:
            reward = reward * -1

        return reward.item()

    def prompt_to_save_images(self, prompt, n_imgs_gen=10):
        """
        Given a prompt, generate an image from it
        args:
            prompt: string representing a prompt (can be empty)
        returns:
            generated image (torch.tensor)
        """
        input_prompt = prompt.split(" ")
        input_prompt = [input_prompt]
        input_type = "prompt"
        out_types = ["image", "reward"]
        if not os.path.exists("saved_imgs/"):
            os.mkdir("saved_imgs")
        # imgs_per_latent = [] ## N latents x bsz : [ [latent 0, bsz imgs ], [latent 1, bsz imgs], ..., [latent N bsz imgs]]
        for ix in range(n_imgs_gen):
            out_dict = self.pipeline(
                input_type=input_type,
                input_value=prompt,
                output_types=out_types,
                fixed_latents=self.fixed_latents,
            )
            pil_image = self.curr_pil_images[0]
            reward = out_dict["reward"].item()
            pil_image.save(
                f"saved_imgs/{prompt}_{ix}_task{self.task_version}_reward{reward}.jpg"
            )

    def proj_word_embedding(self, word_embedding):
        """
        Given a word embedding, project it to the closest word embedding of actual tokens using cosine similarity
        Iterates through each token dim and projects it to the closest token
        args:
            word_embedding: (batch_size, max_num_tokens, 768) word embedding
        returns:
            proj_tokens: (batch_size, max_num_tokens) projected tokens
        """
        # Get word embedding of all possible tokens as torch tensor
        proj_tokens = []
        # Iterate through batch_size
        for i in range(word_embedding.shape[0]):
            # Euclidean Norm
            dists = torch.norm(
                self.all_token_embeddings.unsqueeze(1) - word_embedding[i, :, :], dim=2
            )
            closest_tokens = torch.argmin(dists, axis=0)
            closest_tokens = torch.tensor(
                [self.all_token_idxs[token] for token in closest_tokens]
            ).to(self.torch_device)
            closest_vocab = self.tokenizer.decode(closest_tokens)
            if self.prepend_to_text:
                closest_vocab = (
                    closest_vocab + " " + self.prepend_to_text + " <|endoftext|>"
                )
            cur_proj_tokens = [closest_vocab]
            proj_tokens.append(cur_proj_tokens)

        return proj_tokens


class Y_Compression_Model(nn.Module):
    def __init__(
        self,
        y_hat_dim=32,
        avg_over_N_latents=3,
        verbose=False,
    ):
        super().__init__()
        self.avg_over_N_latents = avg_over_N_latents
        self.y_hat_dim = y_hat_dim
        self.verbose = verbose
        self.conv1 = nn.Conv2d(3, 6, 5)
        # we use the maxpool multiple times, but define it once
        self.pool = nn.MaxPool2d(2, 2)
        # in_channels = 6 because self.conv1 output 6 channel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)
        # 5*5 comes from the dimension of the last convnet layer
        self.fc1 = nn.Linear(
            19_200, 256
        )  # You didnt provide the numbers here but i did calculate the in channels based off the prev layer
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, y_hat_dim)

    def forward(self, x):
        if self.verbose:
            print("input shape: {}".format(x.shape))
        x = x.reshape(-1, 3, 224, 224)
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = self.conv3(x)
        x = self.pool(F.relu(x))
        x = self.conv4(x)
        x = self.pool(F.relu(x))
        x = x.reshape(
            -1, self.avg_over_N_latents, x.shape[-3], x.shape[-2], x.shape[-1]
        )
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation on final layer
        if self.verbose:
            print("output shape: {}".format(x.shape))
        return x


if __name__ == "__main__":
    # Example saving random generated images from a some optimal prompts found by optimizer:
    optimal_prompts_list = ["prompt 1 text", "prompt 2 text"]
    objective = StableDiffusionObjective()
    for prompt in optimal_prompts_list:
        objective.prompt_to_save_images(prompt, n_imgs_gen=10)
