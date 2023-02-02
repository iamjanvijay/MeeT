# usage command: python evaluate.py --loc_preds ../../fan/t5_small_recipes_location/test_predictions_13.tsv --state_preds ../../fan/t5_small_recipes_state/test_predictions_32.tsv --gold_json ../../fan/data/test_recipes_task_koala_split.json

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--loc_preds', type=str, required=True, help="file with formatted location predictions as produced by training code.")
parser.add_argument('--state_preds', type=str, required=True, help="file with formatted state predictions as produced by training code.")
parser.add_argument("--gold_json", type=str, required=True, help="gold json file as included in recipe dataset.")
opt = parser.parse_args()

def make_entry(pid, entity, sen_id, input_dict):
    if pid not in input_dict:
        input_dict[pid] = dict()
    if entity not in input_dict[pid]:
        input_dict[pid][entity] = dict()
    if sen_id not in input_dict[pid][entity]:
        input_dict[pid][entity][sen_id] = []
    return input_dict

def format_predictions(loc_preds_path, state_preds_path):
    state_preds, loc_preds = dict(), dict()

    # reading location predictions
    with open(loc_preds_path) as f:
        for line in f:
            pid, sen_id, entity, _, before_loc, after_loc = line.strip().split('\t')
            sen_id = int(sen_id)
            loc_preds = make_entry(pid, entity, sen_id, loc_preds)
            loc_preds[pid][entity][sen_id].extend([before_loc, after_loc])

    # reading state predictions
    with open(state_preds_path) as f: 
        for line in f:
            pid, sen_id, entity, state, _, _ = line.strip().split('\t')
            sen_id = int(sen_id)
            state_preds = make_entry(pid, entity, sen_id, state_preds)
            state_preds[pid][entity][sen_id].extend([state]) # state: EXIST or OUTSIDE

    # idenitfying location changes
    loc_change_preds = dict()
    for pid in loc_preds:
        for entity in loc_preds[pid]:
            num_sents = len(loc_preds[pid][entity])
            assert(pid in state_preds and entity in state_preds[pid] and num_sents==len(state_preds[pid][entity]))

            location_changes = []
            for sen_id in range(1, num_sents+1):
                state, before_loc, after_loc = state_preds[pid][entity][sen_id][0], loc_preds[pid][entity][sen_id][0], loc_preds[pid][entity][sen_id][1]
                if before_loc != after_loc and state != 'OUTSIDE':
                    location_changes.append({'step': sen_id, 'location': after_loc})
            
            key = f"{pid}#{entity}"
            assert(key not in loc_change_preds), f"Multiple entires for para-id \"{pid}\" and entity \"{entity}\" pair in predictions"
            loc_change_preds[key] = location_changes

    test_key = sorted(loc_change_preds)[0]
    print(f"predicted location changes for {test_key}:", loc_change_preds[test_key])
    return loc_change_preds

def format_gold(gold_json_path):
    data_list = json.load(open(gold_json_path, 'r'))
    
    loc_change_golds = dict()
    for ent_and_passage in data_list:
        pid = ent_and_passage['id']
        entity = ent_and_passage['entity']
        loc_seq = ent_and_passage['gold_loc_seq']
        state_seq = ent_and_passage['gold_state_seq']
        num_sent = len(ent_and_passage['gold_loc_seq']) - 1
        assert(len(state_seq)==num_sent)

        location_changes = []
        for sen_id in range(num_sent):
            state, before_loc, after_loc = state_seq[sen_id], loc_seq[sen_id], loc_seq[sen_id+1] 
            if before_loc != after_loc and state != "O":
                location_changes.append({'step': sen_id+1, 'location': after_loc})

        key = f"{pid}#{entity}"
        assert(key not in loc_change_golds), f"Multiple entires for para-id \"{pid}\" and entity \"{entity}\" pair in gold"
        loc_change_golds[key] = location_changes

    test_key = sorted(loc_change_golds)[0]
    print(f"gold location changes for {test_key}:", loc_change_golds[test_key])
    return loc_change_golds

def main(loc_preds_path, state_preds_path, gold_json_path):
    loc_change_preds, loc_change_golds = format_predictions(loc_preds_path, state_preds_path), format_gold(gold_json_path)
    assert(len(loc_change_preds) == len(loc_change_golds))
    num_data = len(loc_change_preds)

    total_pred, total_ans, total_correct = 0, 0, 0
    for key in loc_change_golds:
        loc_change_gold, loc_change_pred = loc_change_golds[key], loc_change_preds[key]
        
        num_gold, num_pred = len(loc_change_gold), len(loc_change_pred)
        if num_pred == 0 or num_gold == 0:
            num_correct = 0
        else:
            num_correct = len([loc for loc in loc_change_pred if loc in loc_change_gold])

        total_pred += num_pred
        total_ans += num_gold
        total_correct += num_correct

    if total_pred==0:
        precision = 0.0
    else:
        precision = total_correct / float(total_pred)
    recall = total_correct / float(total_ans)
    if (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    print(f'{num_data} instances evaluated.')
    print(f'Total predictions: {total_pred}, total answers: {total_ans}, total correct predictions: {total_correct}')
    print(f'Precision: {precision*100:.1f}, Recall: {recall*100:.1f}, F1: {f1*100:.1f}')

if __name__ == '__main__':
    main(opt.loc_preds, opt.state_preds, opt.gold_json)

                
