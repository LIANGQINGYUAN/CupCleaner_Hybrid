import os
import json
import pandas as pd
import diff_utils


def read_examples(data_dir, split_tag, tokenizer, args):
# def read_examples(filename, data_num):
    """Read examples from filename."""
    task = args.task
    # if 'ACL' in data_dir:
    if 'codesum' in task:
        examples = read_comment_generation_examples(data_dir, split_tag, tokenizer, task)
    elif 'nonl' in task:
        examples = read_comment_update_examples_nonl(data_dir, split_tag, tokenizer, task)
    else:
        examples = read_comment_update_examples(data_dir, split_tag, tokenizer, task)
    return examples


def read_comment_generation_examples(data_dir, split_tag, tokenizer, task):
    examples = []
    if split_tag == 'train':
        filename = os.path.join(data_dir, f"train_static_cleaned_b25.json")
    elif split_tag == 'test':
        filename = os.path.join(data_dir, f"test.json")
    else:
        filename = os.path.join(data_dir, f"train_static_cleaned_b25.json")

    with open(filename) as f:
        index = 0
        for idx, line in enumerate(f):
            x = json.loads(line)
            comm_1 = x["old_comment"]
            comm_2 = x["new_comment"]
            code_1 = x["old_code"]
            code_2 = x["new_code"]
            if idx == 0:
                print("comm_1: ",comm_1)
                print("comm_2: ",comm_2)
                print("code_1: ",code_1)
                print("code_2: ",code_2)

            examples.append(
                Example(
                    idx=index,
                    source= ' '.join(code_1.strip().split(" ")),
                    target= ' '.join(comm_1.replace("\n"," ").strip().split(" "))
                )
            )
            index += 1
            examples.append(
                Example(
                    idx=index,
                    source= ' '.join(code_2.strip().split(" ")),
                    target= ' '.join(comm_2.replace("\n"," ").strip().split(" "))
                )
            )
            index += 1
    return examples

def read_comment_update_examples(data_dir, split_tag, tokenizer, task):
    examples = []
    if split_tag == 'train':
        if 'IQR' in task:
            IQR_TAG = '_'+task.split("_")[-1]
        else:
            IQR_TAG = ""
        if 'static' in task:
            print(f"Reading cleaned data of {task}")
            filename = os.path.join(data_dir, f"train_static_cleaned{IQR_TAG}.json")
        elif 'dynamic' in task:
            print(f"Reading cleaned data of {task}")
            filename = os.path.join(data_dir, f"train_dynamic{IQR_TAG}_cleaned.json")
        elif 'clean' in task:
            print(f"Reading normal data of {task}")
            filename = os.path.join(data_dir, f"train_clean.json")
        else:
            print(f"Reading normal data of {task}")
            filename = os.path.join(data_dir, f"train.json")
    elif split_tag == 'test':
        if 'test150' in task:
            filename = os.path.join(data_dir, f"test150.json")
        else:
            filename = os.path.join(data_dir, f"test.json")
    else:
        if 'static' in task:
            if 'IQR' in task:
                IQR_TAG = '_'+task.split("_")[-1]
            else:
                IQR_TAG = ""
            filename = os.path.join(data_dir, f"valid_static_cleaned{IQR_TAG}.json")
        elif 'dynamic' in task and 'use_cv' in task :
            filename = os.path.join(data_dir, f"valid_static_cleaned.json")
        elif 'clean' in task:
            filename = os.path.join(data_dir, f"valid_clean.json")
        else:
            filename = os.path.join(data_dir, f"valid.json")
    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            comm_1 = x["old_comment"]
            comm_2 = x["new_comment"]
            code_1 = x["old_code"]
            code_2 = x["new_code"]
            cm1_tokens = tokenizer.tokenize(comm_1)
            cd1_tokens = tokenizer.tokenize(code_1)
            cd2_tokens = tokenizer.tokenize(code_2)
            comm_1 = tokenizer.convert_tokens_to_string(cm1_tokens[:100])
            code_1 = tokenizer.convert_tokens_to_string(cd1_tokens[:int(500-len(comm_1)/2)])
            code_2 = tokenizer.convert_tokens_to_string(cd2_tokens[:int(500-len(comm_1)/2)])
            if "CDT" in task:
                code_diff = ' '.join(diff_utils.compute_code_diffs(code_1.split(),code_2.split())[0])
                source= ' '.join((comm_1.strip()+ " <s> " + code_diff.replace("\n"," ")).split(" "))
            else:
                source= ' '.join((comm_1.strip()+ " <s> " + code_1.strip()+ " <s> "+code_2.strip().replace("\n"," ")).split(" "))
            if idx == 0:
                print("comm_1: ",comm_1)
                print("code_1: ",code_1)
                print("code_2: ",code_2)
            
            examples.append(
                Example(
                    idx=idx,
                    source= source,
                    target= ' '.join(comm_2.replace("\n"," ").strip().split(" ")),
                    old_comm=' '.join(x["old_comment"].replace("\n"," ").strip().split(" "))
                )
            )
    return examples

def read_comment_update_examples_nonl(data_dir, split_tag, tokenizer, task):
    examples = []
    if split_tag == 'train':
        filename = os.path.join(data_dir, f"train_static_cleaned.json")
    elif split_tag == 'test':
        filename = os.path.join(data_dir, f"test.json")
    else:
        filename = os.path.join(data_dir, f"valid_static_cleaned.json")
    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            comm_1 = x["old_comment"]
            comm_2 = x["new_comment"]
            code_1 = x["old_code"]
            code_2 = x["new_code"]
            cd1_tokens = tokenizer.tokenize(code_1)
            cd2_tokens = tokenizer.tokenize(code_2)
            code_1 = tokenizer.convert_tokens_to_string(cd1_tokens[:250])
            code_2 = tokenizer.convert_tokens_to_string(cd2_tokens[:250])
            if idx == 0:
                print("comm_1: ",comm_1)
                print("code_1: ",code_1)
                print("code_2: ",code_2)

            examples.append(
                Example(
                    idx=idx,
                    source= ' '.join((code_2.strip()+ " <s> "+code_1.strip().replace("\n"," ")).split(" ")),
                    target= ' '.join(comm_2.replace("\n"," ").strip().split(" ")),
                    old_comm=' '.join(x["old_comment"].replace("\n"," ").strip().split(" "))
                )
            )
    return examples

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 old_comm='',
                 old_code='',
                 new_comm='',
                 new_code=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.old_comm = old_comm
        self.old_code = old_code
        self.new_comm = new_comm
        self.new_code = new_code

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids

def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item
    if args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_len, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_len, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
    )