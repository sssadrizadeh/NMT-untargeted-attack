# NMT-untargeted-attack

This is the official repository of the paper [**"A Relaxed Optimization Approach for Adversarial Attacks against Neural Machine Translation Models"**](http://arxiv.org/abs/2306.08492), an unargeted adversarial attack against NMT systems.

## Installation
Install [conda](https://conda.io) and run the steps below:
```
$ git clone https://github.com/sssadrizadeh/NMT-untargeted-attack.git
$ cd NMT-untargeted-attack
$ conda env create --file=env.yml
$ conda activate attack
```

The datsets and models are available in the HuggingFace transformers package. You can download the language models used in our attack from [here](https://zenodo.org/record/8337474). Please copy these files in the [`language_model`](models/language_model) folder. 

## Performing adversarial attack against NMT models
To attack a translation model, Marian NMT or mBART50, run the following code:
```sh
$ python whitebox_attack.py \
  --num_sentences 1000 \
  --model_name marianmt \
  --language_model_path models/language_model/marian_de \
  --idf_dict_path models/idf_dict/idf_dict_marian_de.json \
  --output_path results  \
  --experiment_name marian_en_de_1000 \
  --lam_sim 45 \
  --lam_perp 0 \
  --target_lang de
```
This code generates adversarial examples against Marian NMT for the samples 0-1000 of the WMT14 (De-En) dataset. You can change the code according to your target model and trasnlation task.

After running the code, a cvs file of the results is generated which can be evaluted by:
```sh
$ python evaluate.py --path <path_csv_file> --model_name <model_name> --target_lang <target_language>
```
This code evaluates the attack in terms of the average semantic similarity between the original sentences  and the adversarial, the token error rate, the success attack rate, and relative decrease in translation quality.


## Citation
If you found this repository helpful, please don't forget to cite our paper:
```BibTeX
@inproceedings{sadrizadeh2023Relaxed,
  title = {A Relaxed Optimization Approach for Adversarial Attacks against Neural Machine Translation Models},
  author = {Sahar Sadrizadeh, Clément Barbier, Ljiljana Dolamic, and Pascal Frossard},
  booktitle = {2023 31th European Signal Processing Conference (EUSIPCO)},
  year = {2023}
}
```
In case of any question, please feel free to contact  [sahar.sadrizadeh@epfl.ch](mailto:sahar.sadrizadeh@epfl.ch).
