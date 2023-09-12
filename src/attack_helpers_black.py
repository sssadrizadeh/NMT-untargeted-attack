import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import numpy as np
import time
# from scipy.stats import entropy
import pandas as pd
import pdb
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bert_score(refs, cands, weights=None, type="avg"):

      refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
      if weights is not None:
          refs_norm *= weights[:, None]
      else:
          refs_norm /= refs.size(1)
      cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
      cosines = refs_norm @ cands_norm.transpose(1, 2)
      # remove first and last tokens; only works when refs and cands all have equal length (!!!)
      cosines = cosines[:, 1:-1, 1:-1]
      
      if type=="avg":
        R = torch.diagonal(cosines[0],0).unsqueeze(0).sum(1)
      else:
        R = cosines.max(-1)[0].sum(1)
      return R


def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()


def shift_tokens_right_marianmt(input_ids: torch.Tensor, model):
    """
    Shift input ids one token to the right.
    """
    pad_token_id=model.config.pad_token_id
    decoder_start_token_id=model.config.decoder_start_token_id
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def shift_tokens_right_mbart50(input_ids: torch.Tensor, model):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()
    pad_token_id=model.config.pad_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

def shift_tokens_for_decoder(input_ids: torch.Tensor, model,model_name):
  # Convert labels into decoder_input_ids according to the model
  if model_name in ["t5-small","t5-base"]:
    return model._shift_right(input_ids)
  if model_name=="marianmt":
    return shift_tokens_right_marianmt(input_ids, model)
  if model_name=="mbart50":
    return shift_tokens_right_mbart50(input_ids, model)

def compute_margin_loss(logits,labels,kappa,batch_size):
  top_preds = logits.sort(descending=True)[0][:, :, 0]
  labels_pred=logits.gather(2,labels.repeat(batch_size,1).view(batch_size,-1,1).to(device))[:,:,0]
  return (labels_pred-top_preds-kappa).clamp(min=0).mean()

def get_model_name_or_path(model_name,target_lang):
  if model_name=="marianmt":
    return f'Helsinki-NLP/opus-mt-en-{target_lang}'
  elif model_name=="mbart50":
    return "facebook/mbart-large-50-one-to-many-mmt"



def get_ids_adv_text(data,num_ids,start_idx,seed=42):
  #Generate the ids of the sentences to attack in the dataset
  np.random.seed(seed)
  ids=np.arange(len(data))
  np.random.shuffle(ids)
  return ids[start_idx:num_ids]


def tokenizer_mbart(args,tokenizer,model_name):
  if model_name=="mbart50":
    tokenizer.src_lang= "en_XX"
    if args.target_lang=='fr':
        tokenizer.tgt_lang= "fr_XX"
    elif args.target_lang=='de':
        tokenizer.tgt_lang= "de_DE"
        
  tokenizer.model_max_length = 512
  return tokenizer


def adv_attack(args,source_tokenizer,target_model,target_tokenizer,data,ids_to_attack,source_task_prefix,target_task_prefix,sentencepiece_token,metric):
  
  with open(args.pkl_addr,"rb") as f:
    all_log_coeefs = pickle.load(f)

  dictionnary_list=[]

  count=-1
  for idx in tqdm(ids_to_attack):
    count+=1
    dict_attack={}

    dict_attack.update({"source_model_name":args.source_model_name,"target_model_name":args.target_model_name,
                        "idx":idx,"num_iters":args.num_iters,
                        "batch_size":args.batch_size,"lr":args.lr,"lam_sim":args.lam_sim,
                        "lam_perp":args.lam_perp,"margin_loss":args.margin_loss,
                        "adapt_teacher_forcing":args.adapt_teacher_forcing,"kappa":args.kappa})

    
    original_translation=data['translation'][idx][args.target_lang]
    original_text=data['translation'][idx]['en']

    target_initial_tokens= target_tokenizer(target_task_prefix+data['translation'][idx]['en'], max_length=256, truncation=True,return_tensors="pt")["input_ids"]
    target_initial_translation_tokens = target_model.generate(target_initial_tokens.to(device),max_length=512)
    initial_translation=[target_tokenizer.decode(t, skip_special_tokens=True) for t in target_initial_translation_tokens][0]
    initial_bleu_score=metric.compute(predictions=[initial_translation],references=[[original_translation]])["score"]

    dict_attack.update({"original_text":original_text,"original_translation":original_translation,
                        "initial_translation":initial_translation,"initial_bleu_score":initial_bleu_score})


    log_coeffs = all_log_coeefs[count]

    time_end_optimization = time.time()
    #Generation of the adversarial samples after the optimisation 
    dict_attack=generate_adv_examples_batch_black(args,source_tokenizer,target_model,target_tokenizer,log_coeffs,dict_attack,sentencepiece_token, target_task_prefix, source_task_prefix)


    time_end_adv_attack=time.time()
    dict_attack.update({"optimization_time":0,"adv_sampling_time":time_end_adv_attack-time_end_optimization})
    dictionnary_list.append(dict_attack)
    torch.cuda.empty_cache()

  target_model = 0
  
  timestamp = time.strftime('_%b-%d-%Y_%H%M', time.localtime())

  print("Evaluation")
  dataframe_results = eval_(dictionnary_list,metric,target_task_prefix,args)
  dataframe_results = pd.DataFrame(dataframe_results)


  timestamp = time.strftime('_%b-%d-%Y_%H%M', time.localtime())
  dataframe_results.to_csv(args.output_path+"/"+args.source_model_name+"_to"+args.target_model_name+"_"+args.experiment_name+"_"+timestamp+".csv")

  return dataframe_results


def generate_adv_examples_batch_black(args,source_tokenizer,target_model,target_tokenizer,log_coeffs,dict_attack,sentencepiece_token, target_task_prefix, source_task_prefix):
  
  args.batch_size_generation=10
  
  with torch.no_grad():
    adv_ids_list=[]
    for i in range(args.num_samples//args.batch_size_generation*args.batch_size_generation):
      adv_ids_list.append(F.gumbel_softmax(torch.tensor(log_coeffs), hard=True).argmax(1).unsqueeze(0))
      
    adv_ids_samples = torch.cat(adv_ids_list).to(device)
    adv_text_samples = [target_task_prefix+source_tokenizer.decode(t, skip_special_tokens=True).replace(sentencepiece_token," ")[len(source_task_prefix):] for t in adv_ids_samples]
    adv_retokenized_samples = target_tokenizer(adv_text_samples, max_length=256, truncation=True, padding=True, return_tensors='pt')["input_ids"].to(device)

    num_batches=args.num_samples//args.batch_size_generation
    translated_samples=[]
    for i in range(num_batches):
      translated = target_model.generate(input_ids=adv_retokenized_samples[i*args.batch_size_generation:(i+1)*args.batch_size_generation],max_length=512)
      for translation_ids in translated:
        translated_samples.append(translation_ids)

      translation=[target_tokenizer.decode(t, skip_special_tokens=True) for t in translated_samples]
      adv_text_retokenized=[target_tokenizer.decode(t, skip_special_tokens=True).replace(sentencepiece_token," ") for t in adv_retokenized_samples]
    
    dict_attack.update({"translation":translation})
    dict_attack.update({"adv_text_retokenized":adv_text_retokenized})
  
  return dict_attack




def eval_(dictionnary_list,metric,task_prefix,args):
  import tensorflow_hub as hub
  import tensorflow_text

  for i,dict_attack in enumerate(dictionnary_list):

    bleu_score_list=[]
    bleu_score_ratio_list=[]
    adv_text_list=[]
    translation_list=[]
    semantic_similarity_list=[]
    is_identic_list=[]

    universal_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

    translation = dict_attack["translation"]
    original_translation = dict_attack["original_translation"]
    initial_bleu_score = dict_attack["initial_bleu_score"]
    adv_text_retokenized = dict_attack["adv_text_retokenized"]
    original_text = dict_attack["original_text"]

    with torch.no_grad():
      for j in range(len(translation)):

        bleu_score=metric.compute(predictions=[translation[j]],references=[[original_translation]])
        bleu_score_list.append(bleu_score["score"])
        if initial_bleu_score>0:
          bleu_score_ratio_list.append(bleu_score["score"]/initial_bleu_score)
        else:
          bleu_score_ratio_list.append(1)



        adv_text_list.append(adv_text_retokenized[j])
        translation_list.append(translation[j])
        semantic_similarity=np.inner(universal_encoder(original_text),  universal_encoder(adv_text_retokenized[j][len(task_prefix):]))[0][0]
        semantic_similarity_list.append(semantic_similarity)
        #print({"Adversarial Text": adv_text_retokenized, "Adversarial Translation": translation})
        is_identic_list.append(adv_text_retokenized==original_text)

    dict_attack.update({"identic_proportion":np.mean(is_identic_list),"bleu_score_ratio_list":bleu_score_ratio_list,
                        "semantic_similarity_list":semantic_similarity_list})

    argmin_bleu_score=np.argmin(bleu_score_list)
    dict_attack.update({"min_bleu_score":bleu_score_list[argmin_bleu_score],"min_bleu_score_ratio":bleu_score_ratio_list[argmin_bleu_score],
                        "min_bleu_similarity":semantic_similarity_list[argmin_bleu_score],
                        "min_bleu_score_adv_text":adv_text_list[argmin_bleu_score],"min_bleu_score_translation":translation_list[argmin_bleu_score]})
    
    close_semantic_samples=[i for i in range(args.num_samples) if (semantic_similarity_list[i]>0.8 and is_identic_list[i]==False)]
    if len(close_semantic_samples)>0:
      close_semantic_bleu_score_list=np.array(bleu_score_list)[close_semantic_samples]
      close_semantic_bleu_score_ratio_list=np.array(bleu_score_ratio_list)[close_semantic_samples]
      close_semantic_semantic_similarity_list=np.array(semantic_similarity_list)[close_semantic_samples]
      argmin_close_semantic_bleu_score=close_semantic_samples[np.argmin(close_semantic_bleu_score_list)]
      dict_attack.update({"close_semantic_min_bleu_score":bleu_score_list[argmin_close_semantic_bleu_score],
                          "close_semantic_min_bleu_score_ratio":bleu_score_ratio_list[argmin_close_semantic_bleu_score],
                          "close_semantic_min_bleu_similarity":semantic_similarity_list[argmin_close_semantic_bleu_score],
                          "close_semantic_min_bleu_score_adv_text":adv_text_list[argmin_close_semantic_bleu_score],
                          "close_semantic_min_bleu_score_translation":translation_list[argmin_close_semantic_bleu_score]})

    else:
      dict_attack.update({"close_semantic_min_bleu_score":-1,"close_semantic_min_bleu_score_ratio":-1,
                          "close_semantic_min_bleu_similarity":-1,
                          "close_semantic_min_bleu_score_adv_text":"","close_semantic_min_bleu_score_translation":""})

    dictionnary_list[i] = dict_attack
  return dictionnary_list

  