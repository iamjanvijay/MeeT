import argparse
from ast import arg
import os
import json
import sys
import numpy as np
import random
import torch
from tqdm import tqdm, trange
from stemming.porter2 import stem

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import  T5Tokenizer, T5ForConditionalGeneration, AdamW

ID2LABEL = ['outside', 'create', 'exist', 'move', 'destroy'] # TODO: experiment with distinct "outside create" and "outside destroy" labels
LABEL2ID = dict([(value, key) for key, value in enumerate(ID2LABEL)])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_model", default=None, type=str, required=True)
    parser.add_argument("--tokeniser", default=None, type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n_sen', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16, required=True)
    parser.add_argument('--max_i_len', type=int, default=512, help="max input length")
    parser.add_argument('--max_o_len', type=int, default=20, help="max target length")
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument("--output_dir", default='./out/', type=str)
    parser.add_argument('--random_seed', type=int, default=1234, help="random seed for random library")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")

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
        map_state_labels = lambda x: 'O' if (x in ['O_C', 'O_D']) else x # mapping O_D and O_C to common label O
        label_to_t5_state_label = {'O': 'outside', 'C': 'create', 'E': 'exist', 'M': 'move', 'D': 'destroy'}
        state_list.append([LABEL2ID[label_to_t5_state_label[map_state_labels(state)]] for state in ent_and_passage['gold_state_seq']]) # state labels
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

    for passage, ent, label in zip(passages, ents, labels['location']):
        for sen_idx in range(len(passage)):
            source = [f"where is {ent} located in sent {sen_idx+1}? \\n"]
            for in_sen_idx in range(len(passage)):
                source += [f" sent {in_sen_idx+1}: {passage[in_sen_idx]}"]
            source += [f" \"end\" . other locations: none, unknown ."]
            if sen_idx==len(passage)-1: # last sentence
                source += [f"$where is {ent} located in the \"end\"? \\n"]
                for in_sen_idx in range(len(passage)):
                    source += [f" sent {in_sen_idx+1}: {passage[in_sen_idx]}"]
                source += [f" \"end\" . other locations: none, unknown ."]
                
            source_list.extend(trim_spaces(source)) 
            target_list.extend([label[sen_idx]])
            if sen_idx==len(passage)-1: # location of the entity after the process ends
                target_list.extend([label[len(passage)]]) 

    print("*"*50)
    print(f"INPUT: {source_list[-2]}")
    print(f"OUTPUT: {target_list[-2]}")
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
    assert(labels.size()[1] < max_target_length), f"Increaset the max target length to : {1+labels.size()[1]}"

    final_data = TensorDataset(input_ids, attention_mask, labels)
    final_sampler = RandomSampler(final_data) if file_train else SequentialSampler(final_data)
    final_dataloader = DataLoader(final_data, sampler=final_sampler, batch_size=args.batch_size)
    return final_dataloader

def print_and_log(f_log, string):
    string = str(string)
    print(string)
    f_log.write(string + '\n')

# metrics for the location task
def normalize_text(location, stemming=False):
    if stemming:
        loc_string = ' %s ' % ' '.join([stem(x) for x in location.lower().replace('"','').split()])
    else:
        loc_string = ' %s ' % ' '.join([x for x in location.lower().replace('"','').split()])
    return loc_string

def compute_exact_match(prediction, truth, stemming=False):
    return int(normalize_text(prediction, stemming) == normalize_text(truth, stemming))

def compute_f1(prediction, truth, stemming=False):
    pred_tokens = normalize_text(prediction, stemming).split()
    truth_tokens = normalize_text(truth, stemming).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    f1 = 2 * (prec * rec) / (prec + rec)
    if f1 > 1.0:
        print(f"*****Warning: F1-score {f1} greater than 1.")
    return f1

def compute_location_metric(gold_locations, pred_locations, mtype='f1', stemming=False):
    if mtype=='em':
        total, matched = 0., 0.
        for gold_location, pred_location in zip(gold_locations, pred_locations):
            matched += compute_exact_match(pred_location, gold_location, stemming)
            total += 1
        score = matched / total
    else: # F1
        total, total_f1 = 0., 0.
        for gold_location, pred_location in zip(gold_locations, pred_locations):
            total_f1 += compute_f1(pred_location, gold_location, stemming)
            total += 1
        score = total_f1 / total
    return score

def process_location(location):
    if location == 'unknown':
        return '?'
    elif location == 'none':
        return '-'
    else:
        return location

def save_model(f_log, ckpt_path, model):
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    model.save_pretrained(ckpt_path)
    print_and_log(f_log, f"Saved checkpoint at: {ckpt_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    f_log = open(os.path.join(args.output_dir, 'train.log'), 'w')
    print_and_log(f_log, f"device: {device} | n_gpu: {n_gpu}")

    # setting seeds
    set_global_seed(args.random_seed)

    # model and optimiser
    tokenizer, model = get_model(args) # create model
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # loading and pre-processing data
    train_file, dev_file, test_file = [json.load(open(data_path, 'r')) for data_path in ['data/propara/train.json', 'data/propara/dev.json', 'data/propara/test.json']]
    print_and_log(f_log, "Total # of train/dev/test points: {}/{}/{}".format(len(train_file), len(dev_file), len(test_file)))

    train_pid, train_passage, train_ent, train_labels = load_data(train_file) # Prepare the train data
    dev_pid, dev_passage, dev_ent, dev_labels = load_data(dev_file) # Prepare the dev data
    test_pid, test_passage, test_ent, test_labels = load_data(test_file) # Prepare the test data

    print("Start pre-processing training data")
    train_dataloader = convert_examples_to_features(args, train_passage, train_ent, train_labels, tokenizer, file_train=True)
    print("Start pre-processing dev data")
    dev_dataloader = convert_examples_to_features(args, dev_passage, dev_ent, dev_labels, tokenizer)
    print("Start pre-processing test data")
    test_dataloader = convert_examples_to_features(args, test_passage, test_ent, test_labels, tokenizer)

    best_stemmed_em = -1
    # trange is a tqdm wrapper around the normal python range
    for num_epoch in trange(args.epochs, desc="Epoch"):

        # Training
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        with tqdm(total=len(train_dataloader), file=sys.stdout) as pbar:
            for step, batch in enumerate(train_dataloader):

                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                # forward pass
                loss = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)[0]

                # backward pass
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)
                # update parameters
                optimizer.step()
                model.zero_grad()
                pbar.update(1)

        print_and_log(f_log, f"Epoch: {num_epoch}")
        print_and_log(f_log, "Train loss: {}".format(tr_loss / nb_tr_steps))
            
        # Evaluation
        model.eval()

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
                        dev_gold += [gold_sen]
                        dev_pred += [pred_sen]

                pbar.update(1)

        print_and_log(f_log, f"gold_sens: {gold_sens[0]}")
        print_and_log(f_log, f"pred_sens: {pred_sens[0]}")
        print_and_log(f_log, "The performance on dev set: ")
        dev_f1 = compute_location_metric(dev_gold, dev_pred, mtype='f1', stemming=False)
        dev_stemmed_f1 = compute_location_metric(dev_gold, dev_pred, mtype='f1', stemming=True)
        dev_em = compute_location_metric(dev_gold, dev_pred, mtype='em', stemming=False)
        dev_stemmed_em = compute_location_metric(dev_gold, dev_pred, mtype='em', stemming=True)

        print_and_log(f_log, "Stemmed-EM: {:.3%}".format(dev_stemmed_em))
        print_and_log(f_log, "        EM: {:.3%}".format(dev_em))
        print_and_log(f_log, "Stemmed-F1: {:.3%}".format(dev_stemmed_f1))
        print_and_log(f_log, "        F1: {:.3%}".format(dev_f1))

        best_dev_stemmed_em = False
        if dev_stemmed_em > best_stemmed_em:
            best_stemmed_em = dev_stemmed_em
            best_dev_stemmed_em = True

        if best_dev_stemmed_em:
            # save model
            save_model(f_log, os.path.join(args.output_dir, 'ckpts', f'best.ckpt'), model)

            # Output the prediction for evaluation on dev set
            dev_plens = [len(passage) for passage in dev_passage]
            with open(os.path.join(args.output_dir, f'dev_predictions_best.tsv'), 'w') as f:
                for para_num, dev_plen in enumerate(dev_plens):
                    para_id, para_entity = dev_pid[para_num], dev_ent[para_num]
                    start_id = sum(dev_plens[:para_num]) + len(dev_plens[:para_num])
                    para_dev_pred = dev_pred[start_id: start_id+dev_plen+1]
                    para_dev_gold = dev_gold[start_id: start_id+dev_plen+1]
                    for sen_id in range(1, len(para_dev_gold)):
                        pred_before_loc, pred_after_loc = process_location(para_dev_pred[sen_id-1]), process_location(para_dev_pred[sen_id])
                        f.write(f"{para_id}\t{sen_id}\t{para_entity}\tNo-State\t{pred_before_loc}\t{pred_after_loc}\n")

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
                            test_gold += [gold_sen]
                            test_pred += [pred_sen]   
                    pbar.update(1)

            print_and_log(f_log, f"gold_sens: {gold_sens[0]}")
            print_and_log(f_log, f"pred_sens: {pred_sens[0]}")
            print_and_log(f_log, "The performance on test set: ")
            test_f1 = compute_location_metric(test_gold, test_pred, mtype='f1', stemming=False)
            test_stemmed_f1 = compute_location_metric(test_gold, test_pred, mtype='f1', stemming=True)
            test_em = compute_location_metric(test_gold, test_pred, mtype='em', stemming=False)
            test_stemmed_em = compute_location_metric(test_gold, test_pred, mtype='em', stemming=True)

            print_and_log(f_log, "Stemmed-EM: {:.3%}".format(test_stemmed_em))
            print_and_log(f_log, "        EM: {:.3%}".format(test_em))
            print_and_log(f_log, "Stemmed-F1: {:.3%}".format(test_stemmed_f1))
            print_and_log(f_log, "        F1: {:.3%}".format(test_f1))

            # Output the prediction for evaluation on test set
            test_plens = [len(passage) for passage in test_passage]
            with open(os.path.join(args.output_dir, f'test_predictions_best.tsv'), 'w') as f:
                for para_num, test_plen in enumerate(test_plens):
                    para_id, para_entity = test_pid[para_num], test_ent[para_num]
                    start_id = sum(test_plens[:para_num]) + len(test_plens[:para_num])
                    para_test_pred = test_pred[start_id: start_id+test_plen+1]
                    para_test_gold = test_gold[start_id: start_id+test_plen+1]
                    for sen_id in range(1, len(para_test_gold)):
                        pred_before_loc, pred_after_loc = process_location(para_test_pred[sen_id-1]), process_location(para_test_pred[sen_id])
                        f.write(f"{para_id}\t{sen_id}\t{para_entity}\tNo-State\t{pred_before_loc}\t{pred_after_loc}\n")

            # only print the config for best checkpoint
            config = vars(args).copy()
            config['test-Stemmed-EM'] = test_stemmed_em
            config['test-EM'] = test_em
            config['test-Stemmed-F1'] = test_stemmed_f1
            config['test-F1'] = test_f1

            config['dev-Stemmed-EM'] = dev_stemmed_em
            config['dev-EM'] = dev_em
            config['dev-Stemmed-F1'] = dev_stemmed_f1
            config['dev-F1'] = dev_f1

            config['best_epoch'] = num_epoch
            print_and_log(f_log, config)

    f_log.close()


if __name__ == '__main__':

    args = get_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)