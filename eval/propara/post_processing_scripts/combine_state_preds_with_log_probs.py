import sys
import os
import json

logs_path = sys.argv[1] # state logs path (either from train or inference): logs/propara/train/state

ID2LABEL = ['prior', 'create', 'exist', 'move', 'destroy', 'post'] 
LABEL2ID = dict([(value, key) for key, value in enumerate(ID2LABEL)])
split_types = ['dev', 'test']

def load_data(data_list):
    pid_list, passage_list, ent_list, state_list, loc_list = [], [], [], [], []
    for ent_and_passage in data_list:
        pid_list.append(ent_and_passage['id'])
        ent_list.append(ent_and_passage['entity'])
        text = [step['sentence'] for step in ent_and_passage['sentence_list']]
        passage_list.append(text)        
        label_to_t5_state_label = {'O_C': 'prior', 'O_D': 'post', 'C': 'create', 'E': 'exist', 'M': 'move', 'D': 'destroy'}
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

for split_type in split_types:
    # read the 
    pred_fpath = os.path.join(logs_path, f'{split_type}_predictions_best.tsv')
    preds_seqs = []
    with open(pred_fpath) as f:
        last_key, curr_preds_seqs = "", []
        for line in f:
            para_id, sen_id, entity, final_state, state, nothing = line.strip().split('\t')
            curr_key = f"{para_id}-{entity}"
            if curr_key != last_key and last_key != "":
                if len(curr_preds_seqs) > 0:
                    preds_seqs.append(curr_preds_seqs)
                curr_preds_seqs = []
            curr_preds_seqs.append([para_id, sen_id, entity, final_state, state, nothing])
            last_key = curr_key
    preds_seqs.append(curr_preds_seqs)

    # read the log-probabilites of different states
    state_logprobs_fpath = os.path.join(logs_path, f'{split_type}_preds_best_state_log_probs.csv')
    pred_state_scores = [[] for seq in preds_seqs]
    curr_para_ent_pair = 0
    index_to_state = []
    with open(state_logprobs_fpath) as f:
        for i, line in enumerate(f):
            if i == 0:
                index_to_state = line.strip().split(',')
            else:
                state_scores = [float(x) for x in line.strip().split(',')]
                pred_state_scores[curr_para_ent_pair].append(state_scores)
                if len(pred_state_scores[curr_para_ent_pair]) == len(preds_seqs[curr_para_ent_pair]):
                    curr_para_ent_pair += 1
    state_to_index = {state: index for index, state in enumerate(index_to_state)}

    # some assertions
    for para_ent_index in range(len(preds_seqs)):
        curr_pred_seq, curr_state_scores = preds_seqs[para_ent_index], pred_state_scores[para_ent_index]

        for i in range(len(curr_pred_seq)):
            assert(i+1 == int(curr_pred_seq[i][1])) # in sequence of steps?

            final_state, state = curr_pred_seq[i][3], curr_pred_seq[i][4]
            max_score = max(curr_state_scores[i])
            for j, score in enumerate(curr_state_scores[i]): 
                if score==max_score:
                    assert(index_to_state[j]==state) # state corresponding to max score is the predicted state?

    # load the data and write reformatted state predictions
    test_file = json.load(open(f'data/propara/{split_type}.json', 'r')) 
    test_pid, test_passage, test_ent, test_labels = load_data(test_file) # Prepare the test data

    gold_state_seqs = []
    for pid, passage, ent, label in zip(test_pid, test_passage, test_ent, test_labels['state']):
        para_seq = []
        for sen_idx in range(len(passage)):
            para_seq.append([pid, ent, sen_idx+1, ID2LABEL[label[sen_idx]], passage[sen_idx]])
        gold_state_seqs.append(para_seq)
        
    order_of_logit_scores = ['create', 'exist', 'move', 'destroy' ,'prior', 'post']
    proced_file = open(os.path.join(logs_path, f'{split_type}_best_state_preds_with_logprobs.tsv'), 'w')
    for para_ent_index in range(len(preds_seqs)):
        curr_pred_seq, curr_state_scores, gold_state = preds_seqs[para_ent_index], pred_state_scores[para_ent_index], gold_state_seqs[para_ent_index]
        for sid in range(len(curr_pred_seq)):
            entity, sentence = gold_state[sid][1].lower(), gold_state[sid][4].lower()
            mention_type = 'implicit'
            for ent in entity.split(';'):
                ent = ent.strip()
                if ent in sentence:
                    mention_type = 'explicit'

            print_tuple = [str(gold_state[sid][0]), str(gold_state[sid][2]), gold_state[sid][1],  curr_pred_seq[sid][4], gold_state[sid][3]]
            print_tuple.extend([f'{state}@{score}' for state,score in zip(order_of_logit_scores, curr_state_scores[sid])])
            print_tuple.append(mention_type)            
            proced_file.write('\t'.join(print_tuple) + '\n')

    proced_file.close()