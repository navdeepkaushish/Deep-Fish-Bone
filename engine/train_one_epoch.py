#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:01:19 2026

@author: navdeepkaushish
"""

from tqdm import tqdm

import torch


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device
):

    model.train()

    running_loss = 0

    pbar = tqdm(loader)

    for batch in pbar:

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            "cuda",
            enabled=device.type=="cuda"
        ):

            logits = model(images)

            loss = criterion(
                logits,
                masks
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return (
        running_loss
        /
        len(loader)
    )