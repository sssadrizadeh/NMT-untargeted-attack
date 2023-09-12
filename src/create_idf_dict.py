import argparse
import json
from transformers import AutoTokenizer
from datasets import DatasetDict, load_dataset, concatenate_datasets


from src.utils import get_idf_dict
from src.attack_helpers_scale import get_model_name_or_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White-box attack.")


    
    parser.add_argument("--output_file", required=True, type=str,
        help="where to save the json idf dict file")
    parser.add_argument("--model_name", required=True, type=str,
        choices=["marianmt","t5-base","t5-small","mbart50"],
        help="type of model")
    parser.add_argument("--target_lang", default="fr", type=str,
        choices=["fr","de"],
        help="target language")

    args = parser.parse_args()

    model_name_or_path=get_model_name_or_path(args.model_name,args.target_lang)
    print(model_name_or_path)

    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,use_fast=True)


    print("Loading Dataset")
    # data=DatasetDict.load_from_disk(args.data_path)
    data = load_dataset("wmt14", f"{args.target_lang}-en")

    d1 = load_dataset("news_commentary","en-fr", split="train")
    d2 = load_dataset("un_multi","en-fr", split="train")

    d3 = load_dataset("europarl_bilingual", lang1="en", lang2="fr", split="train")
    d4 = load_dataset("opus_books","en-fr", split="train")

    d1 = d1.remove_columns("id")
    d4 = d4.remove_columns("id")

    d1 = d1.cast(data['train'].features)
    d2 = d2.cast(data['train'].features)
    d3 = d3.cast(data['train'].features)
    d4 = d4.cast(data['train'].features)

    data=concatenate_datasets([data['train'],data['test'],data['validation'],d1,d2,d3,d4])

    print("Create idf dict")
    idf_dict=get_idf_dict(data,tokenizer,nthreads=-1)

    print("Saving idf dict")
    with open(args.output_file, "w") as outfile:
        json.dump(idf_dict,outfile,indent=2)
    