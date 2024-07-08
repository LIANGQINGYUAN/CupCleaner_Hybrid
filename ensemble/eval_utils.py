from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Tuple, Union
import re
from SARI import SARIsent
from pycocoevalcap.meteor.meteor import Meteor
import os
import subprocess
import numpy as np

def subtokenize_comment(comment_line: str, remove_tag=True) -> str:
    """Subtokenize comments from https://github.com/panthap2/deep-jit-inconsistency-detection/blob/master/data_processing/data_formatting_utils.py"""

    if remove_tag:
        comment_line = remove_tag_string(comment_line)
    comment_line = remove_html_tag(
        comment_line.replace("/**", "")
        .replace("**/", "")
        .replace("/*", "")
        .replace("*/", "")
        .replace("*", "")
        .strip()
    )
    comment_line = re.findall(
        r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", comment_line.strip()
    )
    comment_line = " ".join(comment_line)
    comment_line = comment_line.replace("\n", " ").strip()

    tokens = comment_line.split(" ")
    subtokens = []
    for token in tokens:
        curr = re.sub("([a-z0-9])([A-Z])", r"\1 \2", token).split()
        try:
            new_curr = []
            for c in curr:
                by_symbol = re.findall(
                    r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip()
                )
                new_curr = new_curr + by_symbol

            curr = new_curr
        except:
            curr = []
        subtokens = subtokens + [c.lower() for c in curr]

    comment_line = " ".join(subtokens)
    return comment_line.lower()

def remove_tag_string(line: str) -> str:
    search_strings = [
        "@return",
        "@ return",
        "@param",
        "@ param",
        "@throws",
        "@ throws",
    ]
    for s in search_strings:
        line = line.replace(s, "").strip()
    return line


def remove_html_tag(line: str):
    SPECIAL_TAGS = [
        "{",
        "}",
        "@code",
        "@docRoot",
        "@inheritDoc",
        "@link",
        "@linkplain",
        "@value",
    ]
    clean = re.compile("<.*?>")
    line = re.sub(clean, "", line)

    for tag in SPECIAL_TAGS:
        line = line.replace(tag, "")

    return line

def tokenized_xmatch(
    code_refs: List[str], code_preds: List[str]
) -> Tuple[float, List]:
    """Check xMatch for preds and refs after tokenization."""

    length = len(code_refs)
    count = 0
    xmatch_results = []
    for i in range(length):
        r = code_refs[i]
        p = code_preds[i]
        
        p_subtokens = subtokenize_comment(p)
        r_subtokens = subtokenize_comment(r)
        
        if r_subtokens == p_subtokens:
            xmatch_results.append(1)
            count += 1
        else:
            xmatch_results.append(0)
    # end for
    return count / length * 100, xmatch_results

def compute_bleu_scores(
    references: List[str], hypotheses: List[str], dataset: str,
) -> Tuple[float, List]:
    """Compute BLEU score and return the Tuple[average BLEU, list of bleu]"""

    if "comment-update" in dataset:
        refs = [subtokenize_comment(ref) for ref in references]
        hypos = [subtokenize_comment(hyp) for hyp in hypotheses]
    else:
        refs = references
        hypos = hypotheses
    bleu_4_sentence_scores = []
    for ref, hyp in zip(refs, hypos):
        if hyp == "":
            hyp = "<EMPTY>"
        bleu_4_sentence_scores.append(
            sentence_bleu(
                [ref.split()],
                hyp.split(),
                smoothing_function=SmoothingFunction().method2,
                auto_reweigh=True,
            )
            * 100
        )
    return (
        sum(bleu_4_sentence_scores) / float(len(bleu_4_sentence_scores)),
        bleu_4_sentence_scores,
    )

def compute_sari(
    src_corpus: List[str], tgt_corpus: List[str], pred_corpus: List[str], dataset=None
) -> Tuple[float, List]:
    """Computer SARI metrics for edit-related tasks. Note predictions should be predicted string sequences."""

    inp = zip(src_corpus, tgt_corpus, pred_corpus)
    scores = []

    for source, target, predicted in inp:
        if "comment-update" in dataset:
            predicted = subtokenize_comment(predicted)
            source = subtokenize_comment(source)
            target = subtokenize_comment(target)
        scores.append(SARIsent(source, predicted, [target]) * 100)

    return sum(scores) / float(len(scores)), scores

def compute_sentence_meteor(reference_list, sentences):
    preds = dict()
    refs = dict()

    for i in range(len(sentences)):
        preds[i] = [' '.join([s for s in sentences[i]])]
        refs[i] = [' '.join(l) for l in reference_list[i]]

    final_scores = dict()

    scorers = [
        (Meteor(),"METEOR")
    ]

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                final_scores[m] = scs
        else:
            final_scores[method] = scores

    meteor_scores = final_scores["METEOR"]
    return meteor_scores

def compute_meteor(reference_list, sentences):
    
    reference_list = [[subtokenize_comment(g).split()] for g in reference_list]
    sentences = [subtokenize_comment(p).split() for p in sentences]

    meteor_scores = compute_sentence_meteor(reference_list, sentences)
    return 100 * sum(meteor_scores)/len(meteor_scores)

def compute_gleu_score(len_data, orig_file, ref_file, pred_file):
    # command = 'python2.7 gleu/scripts/compute_gleu -s {} -r {} -o {} -d'.format(
    #     os.path.join(PREDICTION_DIR, orig_file), os.path.join(PREDICTION_DIR, ref_file),
    #     os.path.join(PREDICTION_DIR, pred_file))
    g_path = os.path.join(os.getcwd(), './gleu/scripts/compute_gleu')
    command = 'python2.7  {} -s {} -r {} -o {} -d'.format(
        g_path, orig_file, ref_file, pred_file)
    output = subprocess.check_output(command.split())

    output_lines = [l.strip() for l in output.decode("utf-8").split('\n') if len(l.strip()) > 0]
    l = 0
    while l < len(output_lines):
        if output_lines[l][0] == '0':
            break
        l += 1

    scores = np.zeros(len_data, dtype=np.float32)
    while l < len_data:
        terms = output_lines[l].split()
        idx = int(terms[0])
        val = float(terms[1])
        scores[idx] = val
        l += 1
    scores = np.ndarray.tolist(scores)
    return 100*sum(scores)/float(len(scores))

def write2file(srclist, goldlist, predlist):
    with open("src.txt", "w+") as f1, open("gold.txt", "w+") as f2, open("pred.txt", "w+") as f3:
        for s, g, p in zip(srclist, goldlist, predlist):
            f1.write(subtokenize_comment(s)+'\n')
            f2.write(subtokenize_comment(g)+'\n')
            f3.write(subtokenize_comment(p)+'\n')

def compute_gleu(srclist, goldlist, predlist):
    write2file(srclist, goldlist, predlist)
    return compute_gleu_score(len(goldlist),"src.txt", "gold.txt", "pred.txt")


import json
import pandas as pd

def read_data(file_name):
    items = []
    for i in open(file_name,'r').readlines():
        items.append(json.loads(i))
    return pd.DataFrame(items)

def get_em(gold, pred):
    acc = []
    for g, p in zip(gold, pred):
        if g == p:
            acc.append(1)
    return round(sum(acc)/len(gold)*100,2)

def evaluation(gold, pred, src=None):
    # return bleu_from_list(gold,pred), get_em(gold,pred)
    metrics = {
        "xMatch":round(tokenized_xmatch(gold,pred)[0],2),
        "BLEU-4":round(compute_bleu_scores(gold,pred, 'comment_update')[0],2),
        "METEOR":round(compute_meteor(gold,pred),2),
        "GLEU":round(compute_gleu(src, gold,pred),2),
        "SARI":round(compute_sari(src, gold,pred, 'comment_update')[0],2),
    }
    return metrics
    