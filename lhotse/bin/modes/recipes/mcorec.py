from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.mcorec import download_mcorec, prepare_mcorec
from lhotse.utils import Pathlike

__all__ = ["mcorec"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["all"],
    multiple=True,
    help="List of dataset parts to prepare. To prepare multiple parts, pass each with `-p` "
    "Example: `-p train -p dev`. By default, all parts are prepared.",
)
@click.option(
    "--num-jobs",
    "-j",
    type=int,
    default=1,
    help="Number of parallel jobs to run for array synchronization.",
)
def mcorec(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Sequence[str],
    num_jobs: int,
):
    """MCoRec data preparation."""
    prepare_mcorec(
        corpus_dir,
        output_dir=output_dir,
        dataset_parts=dataset_parts,
        num_jobs=num_jobs,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--force-download",
    type=bool,
    default=False,
    help="If True, download even if file is present.",
)
@click.option(
    "--hf-token",
    type=str,
    help="Hugging Face token.",
)
def mcorec(target_dir: Pathlike, force_download: bool, hf_token: str):
    """MCoRec download."""
    download_mcorec(target_dir, force_download=force_download, hf_token=hf_token)
