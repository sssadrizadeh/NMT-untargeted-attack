from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_dataset, load_metric
import torch
import json
import argparse
from src.attack_helpers import adv_attack
from src.attack_helpers import get_model_name_or_path,get_ids_adv_text




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White-box attack.")



    # Data
    parser.add_argument("--data_path", type=str,
        help="folder where is the dataset (in pyarrow format saved from the wmt dataset)")
    parser.add_argument("--num_sentences", default=100, type=int,
        help="number of sentences to attack")
    parser.add_argument("--shuffle", default=True, type=bool,
        help="to shuffle or not the attacked dataset")
    parser.add_argument("--seed", default=42, type=int,
        help="the random seed to select the sentences of the dataset to attack")
    parser.add_argument("--start_index", default=0, type=int,
        help="starting sample index")
    parser.add_argument("--output_path", required=True, type=str,
        help="where to save the output csv file")
    parser.add_argument("--dataset_name", default="wmt14", type=str,
        help="dataset to attack")

    
    

    # Model
    parser.add_argument("--model_name", required=True, type=str,
        choices=["marianmt","t5-base","t5-small","mbart50"],
        help="type of model")
    # parser.add_argument("--translation_model_path", required=True, type=str,
    #     help="folder where is the translation model (can be on the Hugging Face Hub)")
    parser.add_argument("--language_model_path", required=True, type=str,
        help="folder where is the language model with the same tokenizer as the translation model")
    parser.add_argument("--idf_dict_path", required=True, type=str,
        help="path to the Json idf dict")
    parser.add_argument("--target_lang", default="fr", type=str,
        choices=["fr","de"],
        help="target language")
    

    # Attack setting
    parser.add_argument("--num_iters", default=50, type=int,
        help="number of epochs to train for")
    parser.add_argument("--batch_size", default=5, type=int,
        help="batch size during the optimization of the adversarial distribution")
    parser.add_argument("--margin_loss", default=False, type=bool,
        help="use the margin loss instead of the cross entropy loss")
    parser.add_argument("--adapt_teacher_forcing", default=False, type=bool,
        help="adapt the decoder_inputs_ids during the optimization")
    parser.add_argument("--lr", default=3e-1, type=float,
        help="learning rate")
    parser.add_argument("--kappa", default=5, type=float,
        help="margin in the margin loss")
    parser.add_argument("--lam_sim", default=20, type=float,
        help="embedding similarity regularizer")
    parser.add_argument("--lam_perp", default=1, type=float,
        help="(log) perplexity regularizer")
    parser.add_argument("--num_samples", default=100, type=int,
        help="number of adversarial samples generated")
    parser.add_argument("--batch_size_generation", default=10, type=int,
        help="batch size during the generation of adversarial exemples")
    parser.add_argument("--experiment_name", default="", type=str,
        help="name of the experiment to identify the results files")
    

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric=load_metric('sacrebleu')
    model_name_or_path=get_model_name_or_path(args.model_name,args.target_lang)

    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,use_fast=True)
    print("Loading Translation Model")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)


    sentencepiece_token = "▁"
    

    print("Loading Language Model")
    ref_model = AutoModelForCausalLM.from_pretrained(args.language_model_path,output_hidden_states=True).to(device)

    print("Loading idf dict")
    with open(args.idf_dict_path, "r") as read_file:
        idf_dict = json.load(read_file,)


    task_prefix=""

    if args.model_name=="mbart50":
        tokenizer.src_lang= "en_XX"
        if args.target_lang=='fr':
            tokenizer.tgt_lang= "fr_XX"
        elif args.target_lang=='de':
            tokenizer.tgt_lang= "de_DE"
        model.config.forced_bos_token_id= tokenizer.lang_code_to_id[tokenizer.tgt_lang]

    tokenizer.model_max_length = 512

    print("Loading Dataset")
    if args.dataset_name =="wmt14":
        data = load_dataset("wmt14", f"{args.target_lang}-en",split="test")

    elif args.dataset_name=="europarl_bilingual":
        data = load_dataset("europarl_bilingual", lang1="en", lang2="fr", split="train")
    else:
        data = load_dataset(args.dataset_name, "en-fr",split="train")

    ids_to_attack=get_ids_adv_text(data,args.num_sentences,args.start_index)

    dataset_result=adv_attack(args,model,tokenizer,ref_model,idf_dict,data,ids_to_attack,task_prefix,sentencepiece_token,metric)

