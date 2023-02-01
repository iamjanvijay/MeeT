import sys

def format_output_actions(spans, state_change_answer, entity, para_id):
    # idx2state = {0:'O_C', 1:'O_D', 2:'E', 3:'M', 4:'C', 5:'D'}
    # state_change_answer = [idx2state[c] for c in state_change_answer]
    data = []
    destroy = 0
    for step in range(1,len(spans)):
        item = {}
        item['para_id'] = para_id
        item['step'] = step
        item['entity'] = entity
        change = state_change_answer[step-1]
        if destroy == 1 and change != 'O_D':
            change = 'O_D'
        if change == 'E':
            item['action'] = 'NONE'
            if step == 1:
                item['before'] = spans[0]
            else:
                item['before'] = data[step-2]['after']
            item['after'] = item['before']
        elif change == 'C':
            item['action'] = 'CREATE'
            item['before'] = '-'
            item['after'] = spans[step]
        elif change == 'D':
            item['action'] = 'DESTROY'
            if step == 1:
                item['before'] = spans[0]
            else:
                item['before'] = data[step-2]['after']
            item['after'] = '-'
            destroy = 1
        elif change == 'M':
            item['action'] = 'MOVE'
            if step == 1:
                item['before'] = spans[0]
            else:
                item['before'] = data[step-2]['after']
            item['after'] = spans[step]
            if item['before'] == item['after']:
                item['action'] = 'NONE'
        elif change == 'O_C':
            item['action'] = 'NONE'
            item['before'] = '-'
            item['after'] = '-'
        else:
            if step == 1:
                print ('this should not happen')
            item['action'] = 'NONE'
            item['before'] = '-'
            item['after'] = '-'       
        data.append(item)
    return data   

def get_output(metadata, pred_state_seq, pred_loc_seq, two_output_tags):
    """
    Get the predicted output from generated sequences by the model.
    """
    para_id = metadata['para_id']
    entity_name = metadata['entity']
    total_sents = len(pred_state_seq)

    # replace all the - with ?
    for i, pred_loc in enumerate(pred_loc_seq):
        if pred_loc=='-':
            pred_loc_seq[i] = '?'
    
    # {'para_id': '37', 'step': 6, 'entity': 'mineral', 'action': 'NONE', 'before': ' rock', 'after': ' rock'}
    data = format_output_actions(pred_loc_seq, pred_state_seq, entity_name, para_id)
    

    result = dict()
    prediction = [None for _ in range(total_sents)]

    result['id'] = int(data[0]['para_id'])
    result['entity'] = entity_name
    result['total_sents'] = total_sents
    result['prediction'] = prediction

    # {'id': 933, 'entity': 'rock', 'total_sents': 5, 'prediction': [('NONE', '-', '-'), ('NONE', '-', '-'), ('NONE', '-', '-'), ('CREATE', '-', '?'), ('NONE', '?', '?')]}
    for sen_tuple in data:
        sid = sen_tuple['step']-1
        result['prediction'][sid] = (sen_tuple['action'], sen_tuple['before'], sen_tuple['after'])
    return result

def main_merge(state_preds_fpath, location_preds_fpath, merged_preds_fpath):
    two_output_tags = None
    predictions = dict() # {para-id + entity => {metadata => {para_id => int, entity => str}, pred_state_seq => [state_seq], pred_loc_seq => [loc_seq]}}

    with open(state_preds_fpath) as f:
        for line in f:
            para_id, sen_id, entity, final_state, model_state, _ = line.strip().split('\t')
            if two_output_tags is None:
                if model_state == 'outside':
                    two_output_tags = False
                if model_state in ['prior', 'post']:
                    two_output_tags = True
            else:
                assert((two_output_tags == False and model_state not in ['prior', 'post']) or (two_output_tags == True and model_state not in ['outside']))
            map_key = para_id + '-' + entity
            if map_key not in predictions:
                predictions[map_key] = dict()
                predictions[map_key]['metadata'] = dict()
                predictions[map_key]['metadata']['para_id'] = int(para_id)
                predictions[map_key]['metadata']['entity'] = entity
                predictions[map_key]['pred_state_seq'] = []
                predictions[map_key]['pred_loc_seq'] = []
            predictions[map_key]['pred_state_seq'].append((int(sen_id), model_state))

    with open(location_preds_fpath) as f:
        for line in f:
            para_id, sen_id, entity, final_state, before_location, after_location = line.strip().split('\t')
            map_key = para_id + '-' + entity
            predictions[map_key]['pred_loc_seq'].append((int(sen_id), before_location, after_location))

    state_mapper = {'exist': 'E', '<exist>': 'E', 'create': 'C', '<create>': 'C', 'move': 'M', '<move>': 'M', 'destroy': 'D', '<destroy>': 'D', 'outside': 'O', '<outside>': 'O', 'prior': 'O_C', 'post': 'O_D'}
    for map_key in predictions:
        predictions[map_key]['pred_state_seq'] = [state_mapper[x[1]] for x in sorted(predictions[map_key]['pred_state_seq'], key=lambda x: x[0])]
        temp_pred_loc_seq = sorted(predictions[map_key]['pred_loc_seq'], key=lambda x: x[0])
        predictions[map_key]['pred_loc_seq'] = [temp_pred_loc_seq[0][1]]
        predictions[map_key]['pred_loc_seq'] = predictions[map_key]['pred_loc_seq'] + [x[2] for x in temp_pred_loc_seq]
        
    output_result = dict()
    for map_key in predictions:
        assert(len(predictions[map_key]['pred_state_seq']) + 1 == len(predictions[map_key]['pred_loc_seq']))
        pred_instance = get_output(metadata = predictions[map_key]['metadata'], pred_state_seq = predictions[map_key]['pred_state_seq'], pred_loc_seq = predictions[map_key]['pred_loc_seq'], two_output_tags = two_output_tags)
        para_id = pred_instance['id']
        entity_name = pred_instance['entity']
        output_result[str(para_id) + '-' + entity_name] = pred_instance

    with open(merged_preds_fpath, 'w') as fw:
        for map_key in output_result:
            pred_instance = output_result[map_key]
            para_id, entity, total_sents, prediction = pred_instance['id'], pred_instance['entity'], pred_instance['total_sents'], pred_instance['prediction']
            for sen_id in range(total_sents):
                final_state, before_loc, after_loc = prediction[sen_id]
                fw.write(f'{para_id}\t{sen_id+1}\t{entity}\t{final_state}\t{before_loc}\t{after_loc}\n')

if __name__ == '__main__':

    state_preds_fpath = sys.argv[1]
    location_preds_fpath = sys.argv[2]
    merged_preds_fpath = sys.argv[3]

    main_merge(state_preds_fpath, location_preds_fpath, merged_preds_fpath)
