from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_metric
import torch
import argparse
from src.attack_helpers_black import adv_attack

from src.attack_helpers_black import get_model_name_or_path,get_ids_adv_text, tokenizer_mbart




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
    parser.add_argument("--pkl_addr", required=True, type=str,
        help="where the coeff pkl is saved")

    
    

    # Model
    parser.add_argument("--source_model_name", required=True, type=str,
        choices=["marianmt","t5-base","t5-small","mbart50"],
        help="type of model")
    parser.add_argument("--target_model_name", required=True, type=str,
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
    model_name_or_path_source=get_model_name_or_path(args.source_model_name,args.target_lang)
    model_name_or_path_target=get_model_name_or_path(args.target_model_name,args.target_lang)

    print("Loading Tokenizer")
    source_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path_source,use_fast=True)
    target_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path_target,use_fast=True)
    print("Loading Translation Model")
    target_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path_target).to(device)


    sentencepiece_token = "▁"
   


    source_task_prefix=""
    target_task_prefix=""

    source_tokenizer = tokenizer_mbart(args,source_tokenizer,args.source_model_name)
    target_tokenizer = tokenizer_mbart(args,target_tokenizer,args.target_model_name)
    
    if args.target_model_name=="mbart50":
        target_model.config.forced_bos_token_id= target_tokenizer.lang_code_to_id[target_tokenizer.tgt_lang]

    

    print("Loading Dataset")
    data = load_dataset("wmt14", f"{args.target_lang}-en",split="test")

    ids_to_attack=get_ids_adv_text(data,args.num_sentences,args.start_index)

    dataset_result=adv_attack(args,source_tokenizer,target_model,target_tokenizer,data,ids_to_attack,source_task_prefix,target_task_prefix,sentencepiece_token,metric)

    

