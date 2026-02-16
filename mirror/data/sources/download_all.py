"""
Master downloader script - downloads all source datasets.

Usage: python -m mirror.data.sources.download_all
"""

from . import (
    arc,
    bigbench,
    folio,
    gsm8k,
    logiqa,
    math_dataset,
    reclor,
    social_iqa,
    triviaqa,
    additional,
)


def download_all_datasets():
    """Download all source datasets."""
    print("="*60)
    print("Downloading All Source Datasets")
    print("="*60)

    downloaders = [
        ("GSM8K", gsm8k.download),
        ("MATH", math_dataset.download),
        ("LogiQA", logiqa.download),
        ("SocialIQA", social_iqa.download),
        ("TriviaQA", triviaqa.download),
        ("ARC", arc.download),
        ("BIG-Bench", bigbench.download),
        ("FOLIO", folio.download),
        ("ReClor", reclor.download),
        ("Additional", additional.download),
    ]

    for name, downloader in downloaders:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        try:
            downloader()
        except Exception as e:
            print(f"⚠️  {name} download failed: {e}")

    print(f"\n{'='*60}")
    print("Download Complete")
    print(f"{'='*60}")


def load_all_datasets() -> list[dict]:
    """
    Load and normalize all downloaded datasets.

    Returns:
        List of all normalized questions from all sources
    """
    print("="*60)
    print("Loading and Normalizing All Datasets")
    print("="*60)

    loaders = [
        ("GSM8K", gsm8k.load_and_normalize),
        ("MATH", math_dataset.load_and_normalize),
        ("LogiQA", logiqa.load_and_normalize),
        ("SocialIQA", social_iqa.load_and_normalize),
        ("TriviaQA", triviaqa.load_and_normalize),
        ("ARC", arc.load_and_normalize),
        ("BIG-Bench", bigbench.load_and_normalize),
        ("FOLIO", folio.load_and_normalize),
        ("ReClor", reclor.load_and_normalize),
        ("Additional", additional.load_and_normalize),
    ]

    all_questions = []

    for name, loader in loaders:
        print(f"\nLoading {name}...")
        try:
            questions = loader()
            all_questions.extend(questions)
        except Exception as e:
            print(f"⚠️  {name} loading failed: {e}")

    print(f"\n{'='*60}")
    print(f"Total Questions Loaded: {len(all_questions)}")
    print(f"{'='*60}")

    return all_questions


if __name__ == "__main__":
    download_all_datasets()
    load_all_datasets()
