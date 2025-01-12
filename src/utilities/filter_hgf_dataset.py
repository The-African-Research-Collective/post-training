"""
USAGE: python3 filter.py --language_code en yo am ar ig sw ha sn so te --subjects high_school_mathematics elementary_mathematics college_mathematics abstract_algebra

python3 filter.py --language_code amh eng ewe fra hau ibo kin lin lug orm sna sot swa twi wol xho yor zul  --subjects elementary_mathematics --splits validation dev test  --source_dataset masakhane/afrimmlu --filter_column subject --target_dataset taresco/AFRIMMLU-FILTERED-MATH

python3 filter.py --language_code  YO_NG SW_KE AR_XY --subjects elementary_mathematics abstract_algebra high_school_mathematics --splits test  --source_dataset openai/MMMLU --filter_column Subject --target_dataset taresco/OPENAI-MMLU-FILTERED-MATH
"""
import argparse
from datasets import load_dataset

def _load_dataset(dataset_name: str, language_code: str, split: str):
    dataset = load_dataset(dataset_name, language_code)

    if split not in dataset.keys():
        return None
    
    return dataset[split]

def main(args):

    for language_code in args.language_codes:
        for split in args.splits:
            dataset = _load_dataset(args.source_dataset, language_code, split)
            
            if dataset:
                unique_subject = set(dataset[args.filter_column])
                print(f"Unique labels: {unique_subject}")

                dataset = dataset.filter(lambda x: x[args.filter_column] in args.subjects)

                dataset.push_to_hub(args.target_dataset, split=split, config_name=language_code)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_codes', type=str, required=True, nargs='+')
    parser.add_argument('--splits', type=str, required=True, nargs='+')
    parser.add_argument('--subjects', required=True, nargs='+', type=str)
    parser.add_argument('--source_dataset', type=str, help="Name of huggingface dataset to filter")
    parser.add_argument('--filter_column', type=str, help="Column to filter on")
    parser.add_argument('--target_dataset', type=str, help="Name of huggingface dataset to push filtered data to")
    args = parser.parse_args()
    main(args)
