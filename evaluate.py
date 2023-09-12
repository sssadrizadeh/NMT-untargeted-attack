import os 
import pandas as pd
from datasets import load_metric
import argparse
import torch
import logging
import jiwer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer
from src.attack_helpers import get_model_name_or_path
import math



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation.")

    parser.add_argument("--path", required=True, type=str,
        help="path to results")

    parser.add_argument("--model_name", required=True, type=str,
        help="model name")

    parser.add_argument("--target_lang", required=True, type=str,
        help="target language")
    

    # Word Error Rate
    def wer(x, y):
        x = " ".join(["%d" % i for i in x])
        y = " ".join(["%d" % i for i in y])

        return jiwer.wer(x, y)

    # Token error rate
    class eval_TER:
        def __init__(self, device,args):
            logging.basicConfig(level=logging.ERROR)
            self.device = device
            model_name_or_path=get_model_name_or_path(args.model_name,args.target_lang)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,use_fast=True)
            if args.model_name=="mbart50":
                tokenizer.src_lang= "en_XX"
                if args.target_lang=='fr':
                    tokenizer.tgt_lang= "fr_XX"
                elif args.target_lang=='de':
                    tokenizer.tgt_lang= "de_DE"

            tokenizer.model_max_length = 512
            self.tokenizer = tokenizer
            
        def compute_TER(self,org_sent,adv_sent):
            input_ids = self.tokenizer(org_sent, truncation=True)["input_ids"]
            adv_ids = self.tokenizer(adv_sent, truncation=True)["input_ids"]

            return wer(adv_ids,input_ids)

    # Perplexity of GPT2
    class gpt_perp:
        def __init__(self, device):
            self.device = device
            model_id = "gpt2-large"
            self.gpt = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
            self.gpt_tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

        def compute_loss(self,sent):
            self.gpt.eval()
            tokens = self.gpt_tokenizer.encode(sent+'\n')
            loss = self.gpt(torch.LongTensor(tokens).unsqueeze(0).to(self.device),labels = torch.LongTensor(tokens).unsqueeze(0).to(self.device)).loss

            return loss.item()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    filename = args.path


    df = pd.read_csv(filename, index_col=None, header=0,converters={'bleu_score_ratio_list': pd.eval,"semantic_similarity_list":pd.eval})

    task_prefix = ""


    chrf = load_metric('chrf')
    bertscore = load_metric("bertscore")
    bleu = load_metric('sacrebleu')

    def adv_chrf_compute(x):
        adv_chrf = chrf.compute(predictions=[x["min_bleu_score_translation"]], references=[[x["original_translation"]]])['score']
        
        return adv_chrf

    def org_chrf_compute(x):
        org_chrf = chrf.compute(predictions=[x["initial_translation"]], references=[[x["original_translation"]]])['score']
        
        return org_chrf

    def src_chrf_compute(x):
        src_chrf = chrf.compute(predictions=[x["min_bleu_score_adv_text"][len(task_prefix):]], references=[[x["original_text"]]])['score']
        
        return src_chrf


    def adv_bertscore_tr(x):
        adv_bertscore = bertscore.compute(predictions=[x["min_bleu_score_translation"]], references=[[x["original_translation"]]], lang="fr")['f1'][0]

        return adv_bertscore

    def org_bertscore_tr(x):
        org_bertscore = bertscore.compute(predictions=[x["initial_translation"]], references=[[x["original_translation"]]], lang="fr")['f1'][0]

        return org_bertscore


    def src_bertscore(x):
        src_bertscore = bertscore.compute(predictions=[x["min_bleu_score_adv_text"][len(task_prefix):]], references=[[x["original_text"]]], lang="en")['f1'][0]

        return src_bertscore




    df["initial_chrf"] = df.apply(org_chrf_compute, axis = 1)
    df["min_bleu_score_chrf"] = df.apply(adv_chrf_compute, axis = 1)
    df["min_bleu_score_chrf_ratio"] = df.apply(lambda x : x["min_bleu_score_chrf"]/x["initial_chrf"] if x["initial_chrf"]!=0 else 1, axis = 1)

    df["initial_bertscore_tr"] = df.apply(org_bertscore_tr, axis = 1)
    df["min_bleu_score_bertscore_tr"] = df.apply(adv_bertscore_tr, axis = 1)
    df["min_bleu_score_bertscore_ratio"] = df.apply(lambda x : x["min_bleu_score_bertscore_tr"]/x["initial_bertscore_tr"] if x["initial_bertscore_tr"]!=0 else 1, axis = 1)



    df["source_chrf"] = df.apply(src_chrf_compute, axis = 1)
    df["source_bertscore"] = df.apply(src_bertscore, axis = 1)

    # corpus BLEU and chRF

    org = []
    org_tr = []
    ref_tr = []
    adv = []
    adv_tr = []

    for i in range(len(df)):
        if df.loc[i]['initial_bleu_score']!=0:
            org.append([df.loc[i]['original_text']])
            ref_tr.append([df.loc[i]['original_translation']])
            org_tr.append(df.loc[i]['initial_translation'])
            adv.append(df.loc[i]['min_bleu_score_adv_text'])
            adv_tr.append(df.loc[i]['min_bleu_score_translation'])


    bleu_corp_adv = bleu.compute(predictions=adv_tr, references=ref_tr)['score']
    bleu_corp_org = bleu.compute(predictions=org_tr, references=ref_tr)['score']

    chrf_corp_adv = chrf.compute(predictions=adv_tr, references=ref_tr)['score']
    chrf_corp_org = chrf.compute(predictions=org_tr, references=ref_tr)['score']

    chrf_src = chrf.compute(predictions=adv, references=org)['score']


    # TER and Perplexity
    tr_evl = eval_TER(device,args)
    chrf = load_metric('chrf')
    bertscore = load_metric("bertscore")
    gpt = gpt_perp(device)


    def adv_lm_compute(x):
        adv_lm_loss = gpt.compute_loss(x["min_bleu_score_adv_text"])
        
        return adv_lm_loss

    def org_lm_compute(x):
        org_lm_loss = gpt.compute_loss(x["original_text"])
        
        return org_lm_loss

    def ter_compute(x):
        ter = tr_evl.compute_TER(x["original_text"],x["min_bleu_score_adv_text"])
        
        return ter


    df["adv_lm_loss"] = df.apply(adv_lm_compute, axis = 1)
    df["org_lm_loss"] = df.apply(org_lm_compute, axis = 1)


    df["ter"] = df.apply(ter_compute, axis = 1)







    print("attack success rate:", (df[df["initial_bleu_score"]!=0]["min_bleu_score_ratio"]<0.5).mean())

    print("adversarial corpus bleu:",bleu_corp_adv)
    print("initial corpus bleu:",  bleu_corp_org)
    print("adversarial corpus schrF:",  chrf_corp_adv)
    print("initial chrF:",  chrf_corp_org)
    

    print("initial bleu:",  df[df["initial_bleu_score"]!=0]["initial_bleu_score"].mean())
    print("adv bleu:",  df[df["initial_bleu_score"]!=0]["min_bleu_score"].mean())

    print("initial chrf:",  df[df["initial_bleu_score"]!=0]["initial_chrf"].mean())
    print("adv chrf:",  df[df["initial_bleu_score"]!=0]["min_bleu_score_chrf"].mean())

    print("initial bertscore:",  df[df["initial_bleu_score"]!=0]["initial_bertscore_tr"].mean())
    print("adv bertscore:",  df[df["initial_bleu_score"]!=0]["min_bleu_score_bertscore_tr"].mean())

    print("bleu ratio:",  df[df["initial_bleu_score"]!=0]["min_bleu_score_ratio"].mean())
    print("chrf ratio:", df[df["initial_bleu_score"]!=0]["min_bleu_score_chrf_ratio"].mean())
    print("bertscore ratio:", df[df["initial_bleu_score"]!=0]["min_bleu_score_bertscore_ratio"].mean())


    print("similarity:", df[df["initial_bleu_score"]!=0]["min_bleu_similarity"].mean())
    print("chrf for source similarity:",  chrf_src)
    # print("source chrf:",  df[df["initial_bleu_score"]!=0]["source_chrf"].mean())
    print("source bertscore:",  df[df["initial_bleu_score"]!=0]["source_bertscore"].mean())


    print("token error rate:", (df[df["initial_bleu_score"]!=0]["ter"]).mean())

    print("adv lm loss:",  df[df["initial_bleu_score"]!=0]["adv_lm_loss"].mean())
    print("org lm loss:",  df[df["initial_bleu_score"]!=0]["org_lm_loss"].mean())

    print("adv perp score:",  math.exp(df[df["initial_bleu_score"]!=0]["adv_lm_loss"].mean()))
    print("org perp score:",  math.exp(df[df["initial_bleu_score"]!=0]["org_lm_loss"].mean()))

    print("optimization time:", df[df["initial_bleu_score"]!=0]["optimization_time"].mean())
    print("sampling time:", df[df["initial_bleu_score"]!=0]["adv_sampling_time"].mean())

    df.to_csv(filename)