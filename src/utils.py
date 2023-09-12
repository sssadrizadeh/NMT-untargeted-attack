import json
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from math import log
import numpy as np
import pdb


def sent_encode(tokenizer, sent):
    "Encoding as sentence based on the tokenizer"
    sent = sent.strip()
    if sent == "":
        return tokenizer.build_inputs_with_special_tokens([])
    else:
        return tokenizer.encode(sent, add_special_tokens=True, max_length=tokenizer.model_max_length, truncation=True)


def process(a, tokenizer=None):
    if tokenizer is not None:
        a = sent_encode(tokenizer, a['translation']['en']) #[ex['en'] for ex in a['translation']]
    return set(a)

def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.
    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    # pdb.set_trace()
    # with Pool(nthreads) as p:
    #     idf_count.update(chain.from_iterable(p.map(process_partial, arr)))
    
    all_sent = map(process_partial, arr)
    for s in all_sent:
        idf_count.update(s)

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
    return idf_dict

