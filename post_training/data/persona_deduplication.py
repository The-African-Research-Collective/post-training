import os
import argparse
from typing import List

import jsonlines
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


def create_shingles(input_text: str, shingle_size: int) -> List:
    # convert into lowercase string
    input_text = input_text.strip().lower()

    input_text = input_text.split()

    left = 0
    right = shingle_size

    shingles = []
    while right < len(input_text) + 1:
        shingles.append(" ".join(input_text[left:right]))
        right += 1
        right += 1

    return shingles


def create_hash(shingle_set: List[str]) -> MinHash:
    m = MinHash(num_perm=128)

    for shingle in shingle_set:
        m.update(shingle.encode("utf8"))

    return m


def main(args):
    persona_files = [
        file for file in os.listdir(args.data_directory) if file.endswith(".jsonl")
    ]
    lsh = MinHashLSH(threshold=args.threshold, num_perm=128)

    for file in persona_files:
        num_lines = sum(1 for line in open(os.path.join(args.data_directory, file)))
        print(f"Processing File: {file} with {num_lines} lines")
        with (
            jsonlines.open(os.path.join(args.data_directory, file), "r") as reader,
            jsonlines.open(
                os.path.join(args.data_directory, "deduplicated_personas.jsonl"), "a"
            ) as writer,
        ):
            for i, row in tqdm(enumerate(reader), total=num_lines):
                row["id"] = row["id"] + "#" + str(i)

                if row["persona"]:
                    if isinstance(row["persona"], str):
                        row["shingles"] = create_shingles(
                            row["persona"], args.shingle_size
                        )
                    elif (
                        isinstance(row["persona"], dict) and "persona" in row["persona"]
                    ):
                        if isinstance(row["persona"]["persona"], str):
                            row["shingles"] = create_shingles(
                                row["persona"]["persona"], args.shingle_size
                            )
                        else:
                            continue
                    else:
                        continue

                    minhash = create_hash(row["shingles"])

                    lsh.insert(row["id"], minhash)

                    result = lsh.query(minhash)

                    if len(result) == 1:
                        writer.write(row)


if __name__ == "__main__":
    args = argparse.ArgumentParser("Deduplicate Generate Personas")
    args.add_argument("--data_directory", type=str, help="Data directory")
    args.add_argument("--shingle_size", type=int, default=3, help="Shingle size")
    args.add_argument("--threshold", type=float, default=0.8, help="Threshold")
    args = args.parse_args()
    main(args)
