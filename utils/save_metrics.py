from pathlib import Path

import pandas as pd


def save_metrics_csv(
    metrics_dict,
    filename
):

    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    new_row = pd.DataFrame(
        [metrics_dict]
    )

    if filename.exists():

        old = pd.read_csv(
            filename
        )

        new_row = pd.concat(
            [old, new_row],
            ignore_index=True
        )

    new_row.to_csv(
        filename,
        index=False
    )