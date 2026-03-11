#!/usr/bin/env python
# Based on the HuggingFace PEFT BOFT DreamBooth example.
# https://github.com/huggingface/peft/tree/main/examples/boft_dreambooth

import hashlib
import itertools
import logging
import math
import os
from contextlib import nullcontext
from pathlib import Path

import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from utils.args_loader import (
    get_full_repo_name,
    import_model_class_from_model_name_or_path,
    parse_args,
)
from utils.dataset import DreamBoothDataset, PromptDataset, collate_fn
from utils.tracemalloc import TorchTracemalloc, b2mb

from peft import BOFTConfig, get_peft_model


logger = get_logger(__name__)

UNET_TARGET_MODULES = ["to_q", "to_v", "to_k", "query", "value", "key", "to_out.0", "add_k_proj", "add_v_proj"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]


def save_adaptor(accelerator, step, unet, text_encoder, args):
    unwarpped_unet = accelerator.unwrap_model(unet)
    unwarpped_unet.save_pretrained(
        os.path.join(args.output_dir, f"unet/{step}"), state_dict=accelerator.get_state_dict(unet)
    )
    if args.train_text_encoder:
        unwarpped_text_encoder = accelerator.unwrap_model(text_encoder)
        unwarpped_text_encoder.save_pretrained(
            os.path.join(args.output_dir, f"text_encoder/{step}"),
            state_dict=accelerator.get_state_dict(text_encoder),
        )


def main(args):
    validation_prompts = list(filter(None, args.validation_prompt[0].split(".")))

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=accelerator_project_config,
    )
    if args.report_to == "wandb":
        import wandb

        wandb_init = {
            "wandb": {
                "name": args.wandb_run_name,
                "mode": "online",
            }
        }

    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    global_seed = hash(args.wandb_run_name) % (2**32)
    set_seed(global_seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.prior_generation_precision == "fp32":
                dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    if args.use_boft:
        config = BOFTConfig(
            boft_block_size=args.boft_block_size,
            boft_block_num=args.boft_block_num,
            boft_n_butterfly_factor=args.boft_n_butterfly_factor,
            target_modules=UNET_TARGET_MODULES,
            boft_dropout=args.boft_dropout,
            bias=args.boft_bias,
        )
        unet = get_peft_model(unet, config, adapter_name=args.wandb_run_name)
        unet.print_trainable_parameters()

    vae.requires_grad_(False)
    unet.train()

    if args.train_text_encoder and args.use_boft:
        config = BOFTConfig(
            boft_block_size=args.boft_block_size,
            boft_block_num=args.boft_block_num,
            boft_n_butterfly_factor=args.boft_n_butterfly_factor,
            target_modules=TEXT_ENCODER_TARGET_MODULES,
            boft_dropout=args.boft_dropout,
            bias=args.boft_bias,
        )
        text_encoder = get_peft_model(text_encoder, config, adapter_name=args.wandb_run_name)
        text_encoder.print_trainable_parameters()
        text_encoder.train()
    else:
        text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder and not args.use_boft:
            text_encoder.gradient_checkpointing_enable()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = [param for param in unet.parameters() if param.requires_grad]

    if args.train_text_encoder:
        params_to_optimize += [param for param in text_encoder.parameters() if param.requires_grad]

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    data_path = os.path.join(os.getcwd(), "data", "dreambooth")
    if not os.path.exists(data_path):
        os.makedirs(os.path.join(os.getcwd(), "data"), exist_ok=True)
        os.system(f"git clone https://github.com/google/dreambooth.git '{data_path}'")

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.num_dataloader_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_project_name, config=vars(args), init_kwargs=wandb_init)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    if args.train_text_encoder:
        text_encoder.train()

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()

        with TorchTracemalloc() if not args.no_tracemalloc else nullcontext() as tracemalloc:
            for step, batch in enumerate(train_dataloader):
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with accelerator.accumulate(unet):
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    )
                    timesteps = timesteps.long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if args.with_prior_preservation:
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % args.checkpointing_steps == 0 and global_step != 0:
                        if accelerator.is_main_process:
                            save_adaptor(accelerator, global_step, unet, text_encoder, args)

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if (
                    args.validation_prompt is not None
                    and (step + num_update_steps_per_epoch * epoch) % args.validation_steps == 0
                    and global_step > 10
                ):
                    unet.eval()

                    logger.info(
                        f"Running validation... \n Generating {len(validation_prompts)} images with prompt:"
                        f" {validation_prompts[0]}, ......"
                    )
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        safety_checker=None,
                        revision=args.revision,
                    )
                    pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
                    pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    if args.seed is not None:
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    else:
                        generator = None

                    images = []
                    val_img_dir = os.path.join(
                        args.output_dir,
                        f"validation/{global_step}",
                        args.wandb_run_name,
                    )
                    os.makedirs(val_img_dir, exist_ok=True)

                    for val_promot in validation_prompts:
                        image = pipeline(val_promot, num_inference_steps=50, generator=generator).images[0]
                        image.save(os.path.join(val_img_dir, f"{'_'.join(val_promot.split(' '))}.png"[1:]))
                        images.append(image)

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            import wandb

                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {validation_prompts[i]}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    del pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if global_step >= args.max_train_steps:
                    break

        if not args.no_tracemalloc:
            accelerator.print(
                f"GPU Memory before entering the train : {b2mb(tracemalloc.begin)}"
            )
            accelerator.print(
                f"GPU Memory consumed at the end of the train (end-begin): {tracemalloc.used}"
            )
            accelerator.print(
                f"GPU Peak Memory consumed during the train (max-begin): {tracemalloc.peaked}"
            )
            accelerator.print(
                f"GPU Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
