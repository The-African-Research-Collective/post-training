import pandas as pd
import numpy as np
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


def create_language_samples(
    original_df, target_samples_per_language, languages, difficulty_ratio
):
    """
    Create sample datasets for each language from the original English data.
    Each language gets different samples to ensure variety.

    Args:
        original_df: Original dataframe with all English questions
        target_samples_per_language: How many samples per language
        languages: List of target languages
        difficulty_ratio: Dictionary with difficulty levels as keys and ratios as values

    Returns:
        Dictionary with language as key and dataframe as value
    """
    language_samples = {}

    for language in languages:
        print(f"Creating sample dataset for {language}...")
        # Each language gets a different random sample
        if len(original_df) >= target_samples_per_language:
            language_easy_df = original_df[original_df["difficulty"] == "easy"].sample(
                n=int(target_samples_per_language * difficulty_ratio["easy"]),
                replace=False,
            )
            language_medium_df = original_df[
                original_df["difficulty"] == "medium"
            ].sample(
                n=int(target_samples_per_language * difficulty_ratio["medium"]),
                replace=False,
            )
            language_hard_df = original_df[original_df["difficulty"] == "hard"].sample(
                n=int(target_samples_per_language * difficulty_ratio["hard"]),
                replace=False,
            )

            language_df = pd.concat(
                [language_easy_df, language_medium_df, language_hard_df],
                ignore_index=True,
            )
        else:
            # If we don't have enough samples, use replacement
            language_df = original_df.sample(
                n=target_samples_per_language, replace=True
            )

        # Add language column
        language_df = language_df.copy()
        language_df["language"] = language
        language_df["translated"] = False  # Flag to track translation status

        language_samples[language] = language_df

    return language_samples


def sample_with_difficulty_distribution(
    df, domains, target_samples, difficulty_ratio=None
):
    """
    Sample questions across domains and difficulties until target sample size is reached.

    Args:
        df: DataFrame containing the math questions
        domains: List of domains to sample from
        target_samples: Total number of samples to generate
        difficulty_ratio: Dictionary with difficulty levels as keys and ratios as values

    Returns:
        DataFrame with sampled questions
    """
    if difficulty_ratio is None:
        # Default ratio favors easy and medium questions (50%, 35%, 15%)
        difficulty_ratio = {"easy": 0.5, "medium": 0.35, "hard": 0.15}

    # Calculate target samples per difficulty level
    targets_per_difficulty = {
        diff: int(target_samples * ratio) for diff, ratio in difficulty_ratio.items()
    }

    # Adjust to make sure total equals target_samples
    total = sum(targets_per_difficulty.values())
    if total < target_samples:
        # Add the remainder to 'easy' category
        targets_per_difficulty["easy"] += target_samples - total

    print(f"Target samples per difficulty: {targets_per_difficulty}")

    # Collect samples
    collected_samples = []

    for difficulty, target_count in targets_per_difficulty.items():
        questions_pool = df[df["difficulty"] == difficulty]

        if len(questions_pool) == 0:
            print(f"Warning: No questions found for difficulty '{difficulty}'")
            continue

        # Calculate how many times we need to cycle through the data
        samples_needed = target_count
        total_available = len(questions_pool)

        if total_available < samples_needed:
            print(
                f"Warning: Only {total_available} {difficulty} questions available, need {samples_needed}"
            )
            # Take all available and sample with replacement for the rest
            collected_samples.append(questions_pool)

            # Calculate remaining samples needed
            remaining = samples_needed - total_available
            if remaining > 0:
                # Sample with replacement
                additional_samples = questions_pool.sample(n=remaining, replace=True)
                collected_samples.append(additional_samples)
        else:
            # We have enough samples, just take what we need
            sampled = questions_pool.sample(n=samples_needed, replace=False)
            collected_samples.append(sampled)

    # Combine all collected samples
    result = pd.concat(collected_samples, ignore_index=True)

    return result


def main():
    # Specify your dataset (replace with your actual dataset path)
    dataset_name = "SynthLabsAI/Big-Math-RL-Verified"

    # Target samples per language
    target_samples_per_language = 6000

    # Define target languages
    languages = TARGET_LANGUAGES

    # Load dataset
    df = load_math_dataset(dataset_name)
    if df is None:
        raise ValueError("Failed to load the dataset.")

    # Get unique domains
    source = df["source"].unique().tolist()
    print(f"Available data source: {source}")

    # Define difficulty levels
    df = define_difficulty_levels(df)

    for difficulty_level in ["easy", "medium", "hard"]:
        difficulty_df = df[df["difficulty"] == difficulty_level]
        print(f"Number of {difficulty_level} questions: {len(difficulty_df)}")

    # Define difficulty distribution (adjust as needed)
    difficulty_ratio = {
        "easy": 0.5,  # 50% easy questions
        "medium": 0.35,  # 35% medium questions
        "hard": 0.15,  # 15% hard questions
    }

    print(df.head(10))

    print(f"Sampling {target_samples_per_language} base questions...")
    sampled_df = sample_with_difficulty_distribution(
        df,
        source,
        target_samples_per_language
        * len(languages),  # Sample more than needed to ensure variety
        difficulty_ratio,
    )
    # Create datasets for each language
    language_samples = create_language_samples(
        sampled_df, target_samples_per_language, languages, difficulty_ratio
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
    combined_df.to_json("files/all_samples.jsonl", orient="records", lines=True)

    # Save translation metadata
    metadata = {
        "dataset_name": dataset_name,
        "total_samples": len(combined_df),
        "samples_per_language": target_samples_per_language,
        "languages": list(languages),
        "difficulty_ratio": difficulty_ratio,
        "domains": source.tolist() if isinstance(source, np.ndarray) else source,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open("files/metadata.jsonl", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSampling complete! Ready for translation step.")


if __name__ == "__main__":
    main()
