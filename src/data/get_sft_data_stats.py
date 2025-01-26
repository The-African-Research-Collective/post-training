import re
import json
import argparse
import statistics

from typing import Dict
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from collections import Counter, defaultdict


class DatasetAnalyzer:
    def __init__(self, data: Dataset, tokenizer_name: str):
        self.data = data
        self.stats = {}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.short_message_length = 10
        self.long_message_length = 1000

    def analyze_basic_stats(self) -> Dict:
        """Calculate basic dataset statistics."""
        self.stats["total_samples"] = len(self.data)
        # Count samples per language
        language_counter = Counter(item["language"] for item in self.data)
        self.stats["language_distribution"] = dict(language_counter)
        model_counter = Counter(item["model"] for item in self.data)

        self.stats["model_distribution"] = dict(model_counter)
        return self.stats

    def analyze_token_statistics(self) -> Dict:
        """
        Analyze token statistics for each language in the dataset.
        Provides detailed token count statistics for both user and assistant messages.
        """
        token_stats = defaultdict(
            lambda: {
                "user_messages": {
                    "token_counts": [],
                    "total_tokens": 0,
                    "unique_tokens": set(),
                },
                "assistant_messages": {
                    "token_counts": [],
                    "total_tokens": 0,
                    "unique_tokens": set(),
                },
            }
        )

        # Process each conversation
        for item in self.data:
            language = item["language"]

            for message in item["messages"]:
                role = message["role"]
                content = message["content"]

                # Tokenize the content
                tokens = self.tokenizer.encode(content)
                token_count = len(tokens)

                # Update statistics based on role
                stats_key = f"{role}_messages"
                token_stats[language][stats_key]["token_counts"].append(token_count)
                token_stats[language][stats_key]["total_tokens"] += token_count

        # Calculate final statistics for each language
        self.stats["token_statistics"] = {}
        for language, stats in token_stats.items():
            language_stats = {}

            for role in ["user_messages", "assistant_messages"]:
                role_stats = stats[role]
                token_counts = role_stats["token_counts"]

                language_stats[role] = {
                    "average_tokens": statistics.mean(token_counts),
                    "median_tokens": statistics.median(token_counts),
                    "min_tokens": min(token_counts),
                    "max_tokens": max(token_counts),
                    "total_tokens": role_stats["total_tokens"],
                    "unique_tokens": len(role_stats["unique_tokens"]),
                    "token_diversity": len(role_stats["unique_tokens"])
                    / role_stats["total_tokens"]
                    if role_stats["total_tokens"] > 0
                    else 0,
                }

            self.stats["token_statistics"][language] = language_stats

        return self.stats

    def analyze_message_structure(self) -> Dict:
        """Analyze the structure and patterns in messages."""
        total_turns = []
        user_msg_lengths = []
        assistant_msg_lengths = []

        for item in self.data:
            messages = item["messages"]
            total_turns.append(
                len(messages) // 2
            )  # Assuming pairs of user-assistant messages

            for msg in messages:
                if msg["role"] == "user":
                    user_msg_lengths.append(len(msg["content"]))
                elif msg["role"] == "assistant":
                    assistant_msg_lengths.append(len(msg["content"]))

        self.stats["conversation_stats"] = {
            "avg_turns": statistics.mean(total_turns),
            "max_turns": max(total_turns),
            "min_turns": min(total_turns),
            "user_message_length": {
                "avg": statistics.mean(user_msg_lengths),
                "max": max(user_msg_lengths),
                "min": min(user_msg_lengths),
                "median": statistics.median(user_msg_lengths),
            },
            "assistant_message_length": {
                "avg": statistics.mean(assistant_msg_lengths),
                "max": max(assistant_msg_lengths),
                "min": min(assistant_msg_lengths),
                "median": statistics.median(assistant_msg_lengths),
            },
        }

        return self.stats

    def analyze_content_patterns(self) -> Dict:
        """Analyze patterns in the content of messages."""
        language_patterns = defaultdict(
            lambda: {
                "question_marks": 0,
                "numbers": 0,
                "special_chars": 0,
                "common_words": Counter(),
            }
        )

        for item in self.data:
            lang = item["language"]
            for msg in item["messages"]:
                content = msg["content"]

                # Count question marks
                language_patterns[lang]["question_marks"] += content.count("?")

                # Count numbers
                language_patterns[lang]["numbers"] += len(re.findall(r"\d+", content))

                # Count special characters
                language_patterns[lang]["special_chars"] += len(
                    re.findall(r"[^\w\s]", content)
                )

                # Count common words (simple tokenization)
                words = content.split()
                language_patterns[lang]["common_words"].update(words)

        # Convert to regular dict and get top words
        self.stats["language_patterns"] = {}
        for lang, patterns in language_patterns.items():
            self.stats["language_patterns"][lang] = {
                "question_marks": patterns["question_marks"],
                "numbers": patterns["numbers"],
                "special_chars": patterns["special_chars"],
                "top_words": dict(patterns["common_words"].most_common(10)),
            }

        return self.stats

    def analyze_quality_indicators(self) -> Dict:
        """Analyze potential quality indicators in the dataset."""
        self.stats["quality_indicators"] = {
            "empty_messages": 0,
            "very_short_responses": 0,
            "very_long_responses": 0,
            "response_length_ratio": [],
        }

        for item in self.data:
            for i in range(0, len(item["messages"]), 2):
                if i + 1 >= len(item["messages"]):
                    break

                user_msg = item["messages"][i]["content"]
                assistant_msg = item["messages"][i + 1]["content"]

                # Check for empty messages
                if not user_msg.strip() or not assistant_msg.strip():
                    self.stats["quality_indicators"]["empty_messages"] += 1
                # Check for very short responses
                if len(assistant_msg) < self.short_message_length:
                    self.stats["quality_indicators"]["very_short_responses"] += 1
                # Check for very long responses
                if len(assistant_msg) > self.long_message_length:
                    self.stats["quality_indicators"]["very_long_responses"] += 1
                # Calculate response length ratio
                if len(user_msg) > 0:
                    ratio = len(assistant_msg) / len(user_msg)
                    self.stats["quality_indicators"]["response_length_ratio"].append(
                        ratio
                    )

        # Calculate average response length ratio
        ratios = self.stats["quality_indicators"]["response_length_ratio"]
        self.stats["quality_indicators"]["avg_response_length_ratio"] = statistics.mean(
            ratios
        )

        self.stats["quality_indicators"].pop("response_length_ratio")

        return self.stats


def analyze_dataset(dataset_name: str, tokenizer_name: str) -> Dict:
    """Main function to analyze the dataset."""

    data = load_dataset(dataset_name)["train"]
    # Initialize analyzer
    analyzer = DatasetAnalyzer(data, tokenizer_name)

    # Run all analyses
    analyzer.analyze_basic_stats()
    analyzer.analyze_token_statistics()  # New token analysis
    analyzer.analyze_message_structure()
    analyzer.analyze_content_patterns()
    analyzer.analyze_quality_indicators()

    return analyzer.stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dataset statistics")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name on the Hugging Face Hub",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        help="Tokenizer name for the dataset",
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file to save the analysis results",
        default="data_stats.json",
    )
    args = parser.parse_args()
    stats = analyze_dataset(args.dataset_name, args.tokenizer_name)

    with open(args.output_file, "w") as f:
        json.dump(stats, f, indent=4)
