import torch


def per_structure_dice(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-6
):

    probs = torch.sigmoid(
        logits
    )

    preds = (
        probs > threshold
    ).float()

    n_classes = preds.shape[1]

    dice_sum = [0.0] * n_classes

    valid_count = [0] * n_classes

    batch_size = preds.shape[0]

    for c in range(
        n_classes
    ):

        for b in range(
            batch_size
        ):

            pred = preds[
                b, c
            ]

            target = targets[
                b, c
            ]

            # Ignore absent GT masks

            if target.sum() == 0:

                continue

            intersection = (
                pred * target
            ).sum()

            union = (
                pred.sum()
                +
                target.sum()
            )

            dice = (
                2 * intersection
                + smooth
            ) / (
                union + smooth
            )

            dice_sum[c] += (
                dice.item()
            )

            valid_count[c] += 1

    return (
        dice_sum,
        valid_count
    )