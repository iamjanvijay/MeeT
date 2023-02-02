import argparse
from ast import arg
import os
import json
import sys
import numpy as np
import random
import torch
from tqdm import tqdm, trange

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import  T5Tokenizer, T5ForConditionalGeneration, AdamW

ID2LABEL = ['outside', 'exist'] # TODO: experiment with distinct "outside create" and "outside destroy" labels
LABEL2ID = dict([(value, key) for key, value in enumerate(ID2LABEL)])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_model", default=None, type=str, required=True)
    parser.add_argument("--tokeniser", default=None, type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16, required=True)
    parser.add_argument('--max_i_len', type=int, default=512, help="max input length")
    parser.add_argument('--max_o_len', type=int, default=20, help="max target length")
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument("--output_dir", default='./out/', type=str)
    parser.add_argument('--random_seed', type=int, default=1234, help="random seed for random library")

    args = parser.parse_args()
    return args

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False

def get_model(args):
    print("Load checkpoints form: ", args.lm_model)
    tokenizer = T5Tokenizer.from_pretrained(args.tokeniser)
    tokenizer.add_tokens(['<outside>', '<create>', '<exist>', '<move>', '<destroy>']) # added to maintain reproducibility of previous experiments
    tokenizer.add_tokens([f'<start>', f'<options>', f'</options>']) # add start state
    model = T5ForConditionalGeneration.from_pretrained(args.lm_model)
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

def load_data(data_list):
    pid_list, passage_list, ent_list, state_list, loc_list = [], [], [], [], []
    for ent_and_passage in data_list:
        pid_list.append(ent_and_passage['id'])
        ent_list.append(ent_and_passage['entity'])
        text = [step['sentence'] for step in ent_and_passage['sentence_list']]
        passage_list.append(text)        
        label_to_t5_state_label = {'O': 'outside', 'E': 'exist'}
        state_list.append([LABEL2ID[label_to_t5_state_label[state]] for state in ent_and_passage['gold_state_seq']]) # state labels
        def map_location_labels(loc): # mapping "-" to unknown and "?" to none
            if loc == '?':
                return 'unknown'
            elif loc == '-':
                return 'none'
            else:
                return loc
        loc_list.append([map_location_labels(location) for location in ent_and_passage['gold_loc_seq']]) # location labels
    labels = {'state': state_list, 'location': loc_list}

    assert [len(item) for item in passage_list] == [len(item) for item in labels['state']] \
        and [len(item)+1 for item in passage_list] == [len(item) for item in labels['location']] \
        and len(passage_list) == len(ent_list)

    return pid_list, passage_list, ent_list, labels

def convert_examples_to_features(args, passages, ents, labels, tokenizer, file_train=False):
    source_list = []
    target_list = []

    def trim_spaces(source_str_lists):
        questions = " ".join(" ".join(source_str_lists).strip().split()).split("$")
        return questions

    for passage, ent, label in zip(passages, ents, labels['state']):
        for sen_idx in range(len(passage)):
            source = [f"what is the state of {ent} in sent {sen_idx+1}? \\n (a) exist (b) outside \\n"] # without prepending the last_state
            for in_sen_idx in range(len(passage)):
                    source += [f" sent {in_sen_idx+1}: {passage[in_sen_idx]}"]                
            source_list.extend(trim_spaces(source)) 
            target_list.append(f"{ID2LABEL[label[sen_idx]]}")

    print("*"*50)
    print(f"INPUT: {source_list[-1]}")
    print(f"OUTPUT: {target_list[-1]}")
    print("*"*50)
            
    max_source_length = args.max_i_len
    max_target_length = args.max_o_len

    print("Start tokenizing source...")
    # encode the sources
    source_encoding = tokenizer(source_list[:],
                                padding='longest',
                                max_length=max_source_length,
                                truncation=True,
                                return_tensors="pt")
    input_ids, attention_mask = source_encoding.input_ids, source_encoding.attention_mask

    print("Start tokenizing target...")
    # encode the targets
    target_encoding = tokenizer(target_list[:],
                                padding='longest',
                                max_length=max_target_length,
                                truncation=True)
    labels = target_encoding.input_ids

    # replace padding token id's of the labels by -100
    # In PyTorch and Tensorflow, -100 is the ignore_index of the
    labels = torch.tensor(labels)
    labels[labels == tokenizer.pad_token_id] = -100

    print(input_ids.size(), attention_mask.size(), labels.size())
    assert(labels.size()[1] < max_target_length), f"Increase the max target length to : {1+labels.size()[1]}"

    final_data = TensorDataset(input_ids, attention_mask, labels)
    final_sampler = RandomSampler(final_data) if file_train else SequentialSampler(final_data)
    final_dataloader = DataLoader(final_data, sampler=final_sampler, batch_size=args.batch_size)
    return final_dataloader

def print_and_log(f_log, string):
    string = str(string)
    print(string)
    f_log.write(string + '\n')

def compute_modified_metric(gold, pred, selected_classes=['outside', 'exist']):
    precision, recall, fscore, support = precision_recall_fscore_support(gold, pred, average=None)
    final_precision, final_recall, final_fscore, total_support = 0., 0., 0., 0.
    for cur_class, cur_id in LABEL2ID.items():
        if cur_class in selected_classes:
            final_precision += (support[cur_id] * precision[cur_id])
            final_recall += (support[cur_id] * recall[cur_id])
            final_fscore += (support[cur_id] * fscore[cur_id])
            total_support += support[cur_id]
    return final_precision/total_support, final_recall/total_support, final_fscore/total_support

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    f_log = open(os.path.join(args.output_dir, 'inference.log'), 'w')
    print_and_log(f_log, f"device: {device} | n_gpu: {n_gpu}")

    # setting seeds
    set_global_seed(args.random_seed)

    # model and optimiser
    tokenizer, model = get_model(args) # create model
    model.to(device)

    # loading and pre-process the dataset
    dev_file, test_file = [json.load(open(data_path, 'r')) for data_path in ['data/recipes/dev.json', 'data/recipes/test.json']]
    print_and_log(f_log, "Total # of dev/test points: {}/{}".format(len(dev_file), len(test_file)))

    dev_pid, dev_passage, dev_ent, dev_labels = load_data(dev_file) # Prepare the dev data
    test_pid, test_passage, test_ent, test_labels = load_data(test_file) # Prepare the test data

    print("Start pre-processing dev data")
    dev_dataloader = convert_examples_to_features(args, dev_passage, dev_ent, dev_labels, tokenizer)
    print("Start pre-processing test data")
    test_dataloader = convert_examples_to_features(args, test_passage, test_ent, test_labels, tokenizer)

    # Evaluation
    model.eval()

    # Dev set
    dev_gold = []
    dev_pred = []
    with tqdm(total=len(dev_dataloader), file=sys.stdout) as pbar:
        for batch_idx, batch in enumerate(dev_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                output_sequences = model.generate(input_ids=b_input_ids, attention_mask=b_input_mask)  
                gold_sens = tokenizer.batch_decode([[label if label != -100 else 1 for label in labels] for labels in b_labels], skip_special_tokens=True)
                pred_sens = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                for gold_sen, pred_sen in zip(gold_sens, pred_sens):
                    if pred_sen not in ID2LABEL:
                        print(f"Warning!!! something invalid predicted: {pred_sen}")
                        pred_sen = 'outside'
                    dev_gold += [LABEL2ID[gold_sen]]
                    dev_pred += [LABEL2ID[pred_sen]]

            pbar.update(1)

    print_and_log(f_log, f"gold_sens: {gold_sens[0]}")
    print_and_log(f_log, f"pred_sens: {pred_sens[0]}")
    print_and_log(f_log, "The performance on dev set: ")
    dev_acc = accuracy_score(dev_gold, dev_pred)
    dev_prec, dev_rec, dev_f1 = compute_modified_metric(dev_gold, dev_pred)
    print_and_log(f_log, " Accuracy: {:.3%}".format(dev_acc))
    print_and_log(f_log, "Precision: {:.3%}".format(dev_prec))
    print_and_log(f_log, "   Recall: {:.3%}".format(dev_rec))
    print_and_log(f_log, "       F1: {:.3%}".format(dev_f1))
    print_and_log(f_log, classification_report(dev_gold, dev_pred, target_names=ID2LABEL))

    best_dev_f1_test = True
    if best_dev_f1_test:

        # Output the prediction for evaluation on dev set
        dev_plens = [len(passage) for passage in dev_passage]
        assert(sum(dev_plens)==len(dev_pred))
        with open(os.path.join(args.output_dir, f'dev_predictions_best.tsv'), 'w') as f:
            for para_num, dev_plen in enumerate(dev_plens):
                para_id, para_entity = dev_pid[para_num], dev_ent[para_num]
                start_id = sum(dev_plens[:para_num])
                para_dev_pred = dev_pred[start_id: start_id+dev_plen]
                map_labels = {'outside': 'O', 'exist': 'E'}
                out_labels = {'E': 'EXIST', 'O': 'OUTSIDE'}
                for sen_id, pred_id in enumerate(para_dev_pred):
                    before_loc, after_loc = "?", "?"
                    f.write(f"{para_id}\t{sen_id+1}\t{para_entity}\t{out_labels[map_labels[ID2LABEL[pred_id]]]}\t{ID2LABEL[pred_id]}\t{after_loc}\n")

        # Test set
        test_gold = []
        test_pred = []
        with tqdm(total=len(test_dataloader), file=sys.stdout) as pbar:
            for batch_idx, batch in enumerate(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    output_sequences = model.generate(input_ids=b_input_ids, attention_mask=b_input_mask)
                    gold_sens = tokenizer.batch_decode([[label if label != -100 else 1 for label in labels] for labels in b_labels], skip_special_tokens=True)
                    pred_sens = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                    for gold_sen, pred_sen in zip(gold_sens, pred_sens):
                        if pred_sen not in ID2LABEL:
                            print(f"Warning!!! something invalid predicted: {pred_sen}")
                            pred_sen = 'outside'
                        test_gold += [LABEL2ID[gold_sen]]
                        test_pred += [LABEL2ID[pred_sen]]   
                pbar.update(1)

        print_and_log(f_log, f"gold_sens: {gold_sens[0]}")
        print_and_log(f_log, f"pred_sens: {pred_sens[0]}")
        print_and_log(f_log, "The performance on test set: ")
        test_acc = accuracy_score(test_gold, test_pred)
        test_prec, test_rec, test_f1 = compute_modified_metric(test_gold, test_pred)
        print_and_log(f_log, " Accuracy: {:.3%}".format(test_acc))
        print_and_log(f_log, "Precision: {:.3%}".format(test_prec))
        print_and_log(f_log, "   Recall: {:.3%}".format(test_rec))
        print_and_log(f_log, "       F1: {:.3%}".format(test_f1))
        print_and_log(f_log, classification_report(test_gold, test_pred, target_names=ID2LABEL))

        # Output the prediction for evaluation on test set
        test_plens = [len(passage) for passage in test_passage]
        assert(sum(test_plens)==len(test_pred))
        with open(os.path.join(args.output_dir, f'test_predictions_best.tsv'), 'w') as f:
            for para_num, test_plen in enumerate(test_plens):
                para_id, para_entity = test_pid[para_num], test_ent[para_num]
                start_id = sum(test_plens[:para_num])
                para_test_pred = test_pred[start_id: start_id+test_plen]
                map_labels = {'outside': 'O', 'exist': 'E'}
                out_labels = {'E': 'EXIST', 'O': 'OUTSIDE'}
                for sen_id, pred_id in enumerate(para_test_pred):
                    before_loc, after_loc = "?", "?"
                    f.write(f"{para_id}\t{sen_id+1}\t{para_entity}\t{out_labels[map_labels[ID2LABEL[pred_id]]]}\t{ID2LABEL[pred_id]}\t{after_loc}\n")

    f_log.close()



if __name__ == '__main__':

    args = get_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)