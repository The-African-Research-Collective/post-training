import pandas as pd
from datasets import load_dataset
import time
import json

from post_training.constant import TARGET_LANGUAGES


def load_math_dataset(dataset_name):
    """Load the math dataset from Hugging Face."""
    try:
        dataset = load_dataset(dataset_name)
        df = pd.DataFrame(dataset["train"])
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def define_difficulty_levels(df):
    """Define difficulty levels based on solve rates."""
    # Define 3 difficulty levels: easy, medium, hard
    df["difficulty"] = pd.cut(
        df["llama8b_solve_rate"],
        bins=[0, 0.4, 0.67, 1.0],
        labels=["hard", "medium", "easy"],
    )
    return df


def create_language_samples(original_df, target_samples_per_language, languages):
    """
    Create sample datasets for each language from the original English data.
    Each language gets different samples to ensure variety.

    Args:
        original_df: Original dataframe with all English questions
        target_samples_per_language: How many samples per language
        languages: List of target languages

    Returns:
        Dictionary with language as key and dataframe as value
    """
    language_samples = {}

    for language in languages:
        print(f"Creating sample dataset for {language}...")
        # Each language gets a different random sample
        language_df = original_df.sample(n=target_samples_per_language, replace=True)

        # Add language column
        language_df = language_df.copy()
        language_df["language"] = language
        language_df["translated"] = False  # Flag to track translation status

        language_samples[language] = language_df

    return language_samples


def main():
    # Specify your dataset (replace with your actual dataset path)
    dataset_name = "nvidia/OpenMathInstruct-2"

    # Target samples per language
    target_samples_per_language = 10000

    # Define target languages
    languages = TARGET_LANGUAGES

    # Load dataset
    df = load_math_dataset(dataset_name)
    if df is None:
        raise ValueError("Failed to load the dataset.")

    # Filter dataset problem sources
    df = df[df["problem_source"].isin(["augmented_gsm8k", "gsm8k"])]

    # Create datasets for each language
    language_samples = create_language_samples(
        df, target_samples_per_language, languages
    )

    # Save each language dataset separately
    all_datasets = {}
    for language, dataset in language_samples.items():
        # Set a unique ID for each sample to track them later
        dataset["sample_id"] = [f"{language}_{i}" for i in range(len(dataset))]
        # Add to combined dataset
        all_datasets[language] = dataset

    # Save combined dataset
    combined_df = pd.concat(list(language_samples.values()), ignore_index=True)
    combined_df.to_json(
        "files/all_samples_open_instruct.jsonl", orient="records", lines=True
    )

    # Save translation metadata
    metadata = {
        "dataset_name": dataset_name,
        "total_samples": len(combined_df),
        "samples_per_language": target_samples_per_language,
        "languages": list(languages),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open("files/metadata_openinstruct.jsonl", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSampling complete! Ready for translation step.")


if __name__ == "__main__":
    main()
