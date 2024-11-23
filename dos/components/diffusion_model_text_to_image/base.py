import json
import os
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel, BertForMaskedLM

# from threestudio.utils.base import BaseObject
from dos.utils.misc import barrier, cleanup
# from dos.utils.ops import shifted_cosine_decay, shifted_expotional_decay
from dos.utils.typing import *

from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
from torch import Tensor

from functools import partial

def hash_prompt(model: str, prompt: str) -> str:
    import hashlib

    identifier = f"{model}-{prompt}"
    return hashlib.md5(identifier.encode()).hexdigest()


@dataclass
class DirectionConfig:
    name: str
    prompt: Callable[[str], str]
    negative_prompt: Callable[[str], str]
    condition: Callable[
        [Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]],
        Float[Tensor, "B"],
    ]


@dataclass
class PromptProcessorOutput:
    text_embeddings: Float[Tensor, "N Nf"]
    uncond_text_embeddings: Float[Tensor, "N Nf"]
    text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    uncond_text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    directions: List[DirectionConfig]                   
    direction2idx: Dict[str, int]                       
    # use_perp_neg: bool
    # perp_neg_f_sb: Tuple[float, float, float]
    # perp_neg_f_fsb: Tuple[float, float, float]
    # perp_neg_f_fs: Tuple[float, float, float]
    # perp_neg_f_sf: Tuple[float, float, float]

    def get_text_embeds(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Float[Tensor, "BB N Nf"]:
        batch_size = elevation.shape[0]


        if view_dependent_prompting:
            # Get direction
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances)
                ] = self.direction2idx[d.name]

            # Get text embeddings
            text_embeddings = self.text_embeddings_vd[direction_idx]  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]  # type: ignore
        else:
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings.expand(  # type: ignore
                batch_size, -1, -1
            )

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)

    def get_text_embeddings_perp_neg(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Tuple[Float[Tensor, "BBBB N Nf"], Float[Tensor, "B 2"]]:
        assert (
            view_dependent_prompting
        ), "Perp-Neg only works with view-dependent prompting"

        batch_size = elevation.shape[0]

        direction_idx = torch.zeros_like(elevation, dtype=torch.long)
        for d in self.directions:
            direction_idx[
                d.condition(elevation, azimuth, camera_distances)
            ] = self.direction2idx[d.name]
        # 0 - side view
        # 1 - front view
        # 2 - back view
        # 3 - overhead view

        pos_text_embeddings = []
        neg_text_embeddings = []
        neg_guidance_weights = []
        uncond_text_embeddings = []

        side_emb = self.text_embeddings_vd[0]
        front_emb = self.text_embeddings_vd[1]
        back_emb = self.text_embeddings_vd[2]
        overhead_emb = self.text_embeddings_vd[3]

        for idx, ele, azi, dis in zip(
            direction_idx, elevation, azimuth, camera_distances
        ):
            azi = shift_azimuth_deg(azi)  # to (-180, 180)
            uncond_text_embeddings.append(
                self.uncond_text_embeddings_vd[idx]
            )  # should be ""
            if idx.item() == 3:  # overhead view
                pos_text_embeddings.append(overhead_emb)  # side view
                # dummy
                neg_text_embeddings += [
                    self.uncond_text_embeddings_vd[idx],
                    self.uncond_text_embeddings_vd[idx],
                ]
                neg_guidance_weights += [0.0, 0.0]
            else:  # interpolating views
                if torch.abs(azi) < 90:
                    # front-side interpolation
                    # 0 - complete side, 1 - complete front
                    r_inter = 1 - torch.abs(azi) / 90
                    pos_text_embeddings.append(
                        r_inter * front_emb + (1 - r_inter) * side_emb
                    )
                    neg_text_embeddings += [front_emb, side_emb]
                    neg_guidance_weights += [
                        -shifted_expotional_decay(*self.perp_neg_f_fs, r_inter),
                        -shifted_expotional_decay(*self.perp_neg_f_sf, 1 - r_inter),
                    ]
                else:
                    # side-back interpolation
                    # 0 - complete back, 1 - complete side
                    r_inter = 2.0 - torch.abs(azi) / 90
                    pos_text_embeddings.append(
                        r_inter * side_emb + (1 - r_inter) * back_emb
                    )
                    neg_text_embeddings += [side_emb, front_emb]
                    neg_guidance_weights += [
                        -shifted_expotional_decay(*self.perp_neg_f_sb, r_inter),
                        -shifted_expotional_decay(*self.perp_neg_f_fsb, r_inter),
                    ]

        text_embeddings = torch.cat(
            [
                torch.stack(pos_text_embeddings, dim=0),
                torch.stack(uncond_text_embeddings, dim=0),
                torch.stack(neg_text_embeddings, dim=0),
            ],
            dim=0,
        )

        return text_embeddings, torch.as_tensor(
            neg_guidance_weights, device=elevation.device
        ).reshape(batch_size, 2)

# Define condition functions
def front_view_condition(ele, azi, dis):
    return view_condition(ele, azi, dis, front_threshold)

def back_view_condition(ele, azi, dis):
    return view_condition(ele, azi, dis, back_threshold)

def side_view_condition(ele, azi, dis):
    # Assuming side view has some specific condition; for example purposes, this returns True for all
    return torch.ones_like(ele, dtype=torch.bool)

def overhead_view_condition(ele, azi, dis):
    # Assuming the overhead view is determined by elevation being above a certain threshold
    return ele > overhead_threshold

def generic_view_condition(ele, azi, dis):
    # Example condition for a generic view, adjust as needed
    return torch.ones_like(ele, dtype=torch.bool)

def alternative_front_view_condition(ele, azi, dis):
    # Condition for an alternative front view
    return (shift_azimuth_deg(azi) > -front_threshold) & (shift_azimuth_deg(azi) < front_threshold)

def alternative_back_view_condition(ele, azi, dis):
    # Condition for an alternative back view
    return (shift_azimuth_deg(azi) > 180 - back_threshold) | (shift_azimuth_deg(azi) < -180 + back_threshold)

def alternative_overhead_view_condition(ele, azi, dis):
    # Condition for an alternative overhead view
    return ele > overhead_threshold
    
    
# Prompt functions
def side_view_prompt(base_prompt):
    return f"side view of {base_prompt}"

def front_view_prompt(base_prompt):
    return f"front view of {base_prompt}"

def back_view_prompt(base_prompt):
    return f"backside view of {base_prompt}"

def overhead_view_prompt(base_prompt):
    return f"overhead view of {base_prompt}"

def generic_view_prompt(base_prompt):
    return f"{base_prompt}, generic view"

def alternative_front_view_prompt(base_prompt):
    return f"alternative front view of {base_prompt}"

def alternative_back_view_prompt(base_prompt):
    return f"alternative back view of {base_prompt}"

def alternative_overhead_view_prompt(base_prompt):
    return f"alternative overhead view of {base_prompt}"


class PromptProcessor():
    @dataclass
    class Config():
        prompt: str = "a cow running"

        # manually assigned view-dependent prompts
        prompt_side: Optional[str] = None
        prompt_back: Optional[str] = None
        prompt_overhead: Optional[str] = None

        negative_prompt: str = ""
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        overhead_threshold: float = 60.0
        front_threshold: float = 45.0
        back_threshold: float = 45.0
        view_dependent_prompt_front: bool = True
        use_cache: bool = False
        spawn: bool = True

        # perp neg
        use_perp_neg: bool = False
        # a*e(-b*r) + c
        # a * e(-b) + c = 0
        perp_neg_f_sb: Tuple[float, float, float] = (1, 0.5, -0.606)
        perp_neg_f_fsb: Tuple[float, float, float] = (1, 0.5, +0.967)
        perp_neg_f_fs: Tuple[float, float, float] = (
            4,
            0.5,
            -2.426,
        )  # f_fs(1) = 0, a, b > 0
        perp_neg_f_sf: Tuple[float, float, float] = (4, 0.5, -2.426)

        # prompt debiasing
        use_prompt_debiasing: bool = True # Previously False
        pretrained_model_name_or_path_prompt_debiasing: str = "bert-base-uncased"
        # index of words that can potentially be removed
        prompt_debiasing_mask_ids: Optional[List[int]] = None
        
    # cfg: Config
    
    # Added
    def __init__(self):
        # Initialize cfg with an instance of the Config class
        self.cfg = self.Config()
    
    
    def configure_text_encoder(self) -> None:
        raise NotImplementedError

    def destroy_text_encoder(self) -> None:
        raise NotImplementedError

    def configure(self) -> None:
        self._cache_dir = ".threestudio_cache/text_embeddings"  # FIXME: hard-coded path
        
        # Configuring directions with callable prompts
        if self.cfg.view_dependent_prompt_front:
            self.directions = [
                DirectionConfig("side", side_view_prompt, "", partial(side_view_condition, cfg=self.cfg)),
                DirectionConfig("front", front_view_prompt, "", partial(front_view_condition, cfg=self.cfg)),
                DirectionConfig("back", back_view_prompt, "", partial(back_view_condition, cfg=self.cfg)),
                DirectionConfig("overhead", overhead_view_prompt, "", partial(overhead_view_condition, cfg=self.cfg, threshold_elevation=self.cfg.overhead_threshold)),
            ]
        else:
            self.directions = [
                DirectionConfig("generic", generic_view_prompt, "", partial(generic_view_condition, cfg=self.cfg)),
                DirectionConfig("alternative_front", alternative_front_view_prompt, "", partial(alternative_front_view_condition, cfg=self.cfg)),
                DirectionConfig("alternative_back", alternative_back_view_prompt, "", partial(alternative_back_view_condition, cfg=self.cfg)),
                DirectionConfig("alternative_overhead", alternative_overhead_view_prompt, "", partial(alternative_overhead_view_condition, cfg=self.cfg, threshold_elevation=self.cfg.overhead_threshold)),
            ]

        # view-dependent text embeddings
        self.directions: List[DirectionConfig]
        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}

        # with open(os.path.join("load/prompt_library.json"), "r") as f:
        #     self.prompt_library = json.load(f)
        # # use provided prompt or find prompt in library
        # self.prompt = self.preprocess_prompt(self.cfg.prompt)
        
        self.prompt = self.cfg.prompt # added
        
        # use provided negative prompt
        self.negative_prompt = self.cfg.negative_prompt

        print(f"Using prompt [{self.prompt}] and negative prompt [{self.negative_prompt}]")

        # view-dependent prompting
        if self.cfg.use_prompt_debiasing:
            assert (
                self.cfg.prompt_side is None
                and self.cfg.prompt_back is None
                and self.cfg.prompt_overhead is None
            ), "Do not manually assign prompt_side, prompt_back or prompt_overhead when using prompt debiasing"
            prompts = self.get_debiased_prompt(self.prompt)
            self.prompts_vd = [
                d.prompt(prompt) for d, prompt in zip(self.directions, prompts)
            ]
        else:
            self.prompts_vd = [
                d.prompt(self.cfg.get(f"prompt_{d.name}", None) or self.prompt)  # type: ignore
                for d in self.directions
            ]

        prompts_vd_display = " ".join(
            [
                f"[{d.name}]:[{prompt}]"
                for prompt, d in zip(self.prompts_vd, self.directions)
            ]
        )
        print(f"Using view-dependent prompts {prompts_vd_display}")

        # Original
        # self.negative_prompts_vd = [
        #     d.negative_prompt(self.negative_prompt) for d in self.directions
        # ]

        # Updated to
        self.negative_prompts_vd = [d.negative_prompt for d in self.directions]

        # self.prepare_text_embeddings()
        # self.load_text_embeddings()

    
    def spawn_func(self, pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            device_map="auto",
        )

        with torch.no_grad():
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(tokens.input_ids.to(text_encoder.device))[0]

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del text_encoder
    
    def prepare_text_embeddings(self):
        os.makedirs(self._cache_dir, exist_ok=True)

        all_prompts = (
            [self.prompt]
            + [self.negative_prompt]
            + self.prompts_vd
            + self.negative_prompts_vd
        )
        prompts_to_process = []
        for prompt in all_prompts:
            if self.cfg.use_cache:
                # some text embeddings are already in cache
                # do not process them
                cache_path = os.path.join(
                    self._cache_dir,
                    f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
                )
                if os.path.exists(cache_path):
                    print(f"Text embeddings for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] are already in cache, skip processing.")
                    continue
            prompts_to_process.append(prompt)

        if len(prompts_to_process) > 0:
            if self.cfg.spawn:
                ctx = mp.get_context("spawn")
                subprocess = ctx.Process(
                    target=self.spawn_func,
                    args=(
                        self.cfg.pretrained_model_name_or_path,
                        prompts_to_process,
                        self._cache_dir,
                    ),
                )
                subprocess.start()
                subprocess.join()
            else:
                self.spawn_func(
                    self.cfg.pretrained_model_name_or_path,
                    prompts_to_process,
                    self._cache_dir,
                )
            cleanup()

    def load_text_embeddings(self):
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        barrier()
        
        self.text_embeddings = self.load_from_cache(self.prompt)[None, ...]
        self.uncond_text_embeddings = self.load_from_cache(self.negative_prompt)[
            None, ...
        ]
        self.text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.prompts_vd], dim=0
        )
        self.uncond_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.negative_prompts_vd], dim=0
        )
        print(f"Loaded text embeddings.")

    def load_from_cache(self, prompt):
        cache_path = os.path.join(
            self._cache_dir,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Text embedding file {cache_path} for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        return torch.load(cache_path, map_location=self.device)


    def preprocess_prompt(self, prompt: str) -> str:
        if prompt.startswith("lib:"):
            # find matches in the library
            candidate = None
            keywords = prompt[4:].lower().split("_")
            for prompt in self.prompt_library["dreamfusion"]:
                if all([k in prompt.lower() for k in keywords]):
                    if candidate is not None:
                        raise ValueError(
                            f"Multiple prompts matched with keywords {keywords} in library"
                        )
                    candidate = prompt
            if candidate is None:
                raise ValueError(
                    f"Cannot find prompt with keywords {keywords} in library"
                )
            print("Find matched prompt in library: " + candidate)
            return candidate
        else:
            return prompt

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B ..."], Float[Tensor, "B ..."]]:
        raise NotImplementedError

    def get_debiased_prompt(self, prompt: str) -> List[str]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path_prompt_debiasing
        )
        model = BertForMaskedLM.from_pretrained(
            self.cfg.pretrained_model_name_or_path_prompt_debiasing
        )

        views = [d.name for d in self.directions]
        view_ids = tokenizer(" ".join(views), return_tensors="pt").input_ids[0]
        view_ids = view_ids[1:5]

        def modulate(prompt):
            prompt_vd = f"This image is depicting a [MASK] view of {prompt}"
            tokens = tokenizer(
                prompt_vd,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            mask_idx = torch.where(tokens.input_ids == tokenizer.mask_token_id)[1]

            logits = model(**tokens).logits
            logits = F.softmax(logits[0, mask_idx], dim=-1)
            logits = logits[0, view_ids]
            probes = logits / logits.sum()
            return probes

        prompts = [prompt.split(" ") for _ in range(4)]
        full_probe = modulate(prompt)
        n_words = len(prompt.split(" "))
        prompt_debiasing_mask_ids = (
            self.cfg.prompt_debiasing_mask_ids
            if self.cfg.prompt_debiasing_mask_ids is not None
            else list(range(n_words))
        )
        words_to_debias = [prompt.split(" ")[idx] for idx in prompt_debiasing_mask_ids]
        print(f"Words that can potentially be removed: {words_to_debias}")
        for idx in prompt_debiasing_mask_ids:
            words = prompt.split(" ")
            prompt_ = " ".join(words[:idx] + words[(idx + 1) :])
            part_probe = modulate(prompt_)

            pmi = full_probe / torch.lerp(part_probe, full_probe, 0.5)
            for i in range(pmi.shape[0]):
                if pmi[i].item() < 0.95:
                    prompts[i][idx] = ""

        debiased_prompts = [" ".join([word for word in p if word]) for p in prompts]
        for d, debiased_prompt in zip(views, debiased_prompts):
            print(f"Debiased prompt of the {d} view is [{debiased_prompt}]")

        del tokenizer, model
        cleanup()

        return debiased_prompts

    def __call__(self) -> PromptProcessorOutput:
        self.configure()
        return PromptProcessorOutput(
            text_embeddings=self.text_embeddings,
            uncond_text_embeddings=self.uncond_text_embeddings,
            text_embeddings_vd=self.text_embeddings_vd,
            uncond_text_embeddings_vd=self.uncond_text_embeddings_vd,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=self.cfg.use_perp_neg,
            perp_neg_f_sb=self.cfg.perp_neg_f_sb,
            perp_neg_f_fsb=self.cfg.perp_neg_f_fsb,
            perp_neg_f_fs=self.cfg.perp_neg_f_fs,
            perp_neg_f_sf=self.cfg.perp_neg_f_sf,
        )
