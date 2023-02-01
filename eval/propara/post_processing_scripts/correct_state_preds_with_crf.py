import os
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import json 
from collections import Counter, defaultdict
import math

label_to_index = {'prior':0, 'post':1, 'exist':2, 'move':3, 'create':4, 'destroy':5}
index_to_label = ['prior', 'post', 'exist', 'move', 'create', 'destroy']
label_to_final_state = {'prior':'NONE', 'post':'NONE', 'exist':'NONE', 'move':'MOVE', 'create':'CREATE', 'destroy':'DESTROY'}
num_states = len(index_to_label)

class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator, sumed_scores = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.type_as(emissions).sum(), sumed_scores

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        sumed_scores = [score]
        max_score = score
        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)

            next_max_score, _ = next_score.max(dim=1)
            next_score = torch.logsumexp(next_score, dim=1)
            
            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            max_score = torch.where(mask[i].unsqueeze(1), next_max_score, max_score)
            sumed_scores.append(max_score)
        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions
        sumed_scores[-1] = score
        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1), sumed_scores

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        sumed_scores = [score]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            sumed_scores.append(score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions
        sumed_scores[-1] = score
        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list, sumed_scores

def read_propara_data(file):
    with open(file, 'r') as f:
        lines = []
        for line in f:
            if line.strip() == '':
                continue
            lines.append(json.loads(str(line)))
    return lines

def compute_priors():
    state2idx = {'O_C': 0, 'O_D': 1, 'E': 2, 'M': 3, 'C': 4, 'D': 5}
    start_transitions = Counter()
    end_transitions = Counter()
    transitions = defaultdict(Counter)
    data = read_propara_data('data/propara/grids.v1.train.json')
    for sample in data:
        for entity_id, states in enumerate(sample['states']):
            state_transition = compute_state_change_seq(states)
            start_transitions[state_transition[0]] += 1
            end_transitions[state_transition[-1]] += 1
            for i in range(1, len(state_transition)):
                transitions[state_transition[i-1]][state_transition[i]] += 1
    
    start_prior = np.full(6, -1e9)
    end_prior = np.full(6, -1e9)
    transition_prior = np.full((6, 6), -1e9)
    for k, v in start_transitions.items():
        start_prior[state2idx[k]] = math.log(v/sum(start_transitions.values()))
    for k, v in end_transitions.items():
        end_prior[state2idx[k]] = math.log(v/sum(end_transitions.values()))
    for k, v in transitions.items():
        for kk, vv in v.items():
            transition_prior[state2idx[k]][state2idx[kk]] = math.log(vv/sum(v.values()))
    return start_prior, end_prior, transition_prior

def compute_state_change_seq(gold_loc_seq):
    num_states = len(gold_loc_seq)
    # whether the entity has been created. (if exists from the beginning, then it should be True)
    create = False if gold_loc_seq[0] == '-' else True
    gold_state_seq = []

    for i in range(1, num_states):

        if gold_loc_seq[i] == '-':  # could be O_C, O_D or D
            if create == True and gold_loc_seq[i-1] == '-':
                gold_state_seq.append('O_D')
            elif create == True and gold_loc_seq[i-1] != '-':
                gold_state_seq.append('D')
            else:
                gold_state_seq.append('O_C')

        elif gold_loc_seq[i] == gold_loc_seq[i-1]:
            gold_state_seq.append('E')

        else:  # location change, could be C or M
            if gold_loc_seq[i-1] == '-':
                create = True
                gold_state_seq.append('C')
            else:
                gold_state_seq.append('M')
    
    assert len(gold_state_seq) == len(gold_loc_seq) - 1
    
    return gold_state_seq

def function_to_decode(crf_layer, key, logit_scores):
    num_sens = len(logit_scores[key])
    # state_changes should be of shape: 1 * num_sens * num_states
    state_changes = []
    for sid in range(1, num_sens+1):
        scores = []
        for state_id, state_label in enumerate(index_to_label):
            scores.append(logit_scores[key][sid][state_label])
        state_changes.append(scores)
    state_changes = [state_changes]

    state_changes = torch.from_numpy(np.array(state_changes))
    state_change_answer, sumed_scores = crf_layer.decode(emissions=state_changes)
    state_change_answer = state_change_answer[0]

    return [index_to_label[state_change] for state_change in state_change_answer]

# defining the CRF layer with probs computed from counts of state-transitions
CRFLayer = CRF(num_states, batch_first = True)
start_prior, end_prior, transition_prior = compute_priors()
assert len(start_prior) == num_states
CRFLayer.start_transitions = torch.nn.Parameter(torch.tensor(start_prior))
CRFLayer.end_transitions = torch.nn.Parameter(torch.tensor(end_prior))
CRFLayer.transitions = torch.nn.Parameter(torch.tensor(transition_prior))

def write_lines_to_file(lines, filename):
    with open(filename, 'w') as fw:
        for line in lines:
            fw.write(line)

# all_factor, implicit_factor, explicit_factor = 0.5, 1.0, 1.0 : 71.0
implicit_factor, explicit_factor = float(sys.argv[1]), float(sys.argv[2])
split_type = sys.argv[3]
logs_path = sys.argv[4] # state logs path (either from train or inference): logs/propara/train/state

logit_scores = dict() # logit_scores[para_ent][sid][state_type] = score | logit_scores[para_ent][sid][metion_type] = explicit/implicti
dumped_realigned_scores = os.path.join(logs_path, f'{split_type}_best_state_preds_with_logprobs.tsv')
with open(dumped_realigned_scores, 'r') as f:
    for line in f:
        pid, sid, ent, pred_state, gold_state, create_score, exist_score, move_score, destroy_score, prior_score, post_score, mention_type = line.strip().split('\t')  
        
        key = f'{pid}@{ent}'
        if key not in logit_scores:
            logit_scores[key] = dict()
        assert(int(sid) not in logit_scores[key])
        logit_scores[key][int(sid)] = dict()

        logit_scores[key][int(sid)]['metion_type'] = mention_type

        mul_factor = 1.0
        if mention_type == 'implicit':
            mul_factor *= implicit_factor
        else: # explicit
            mul_factor *= explicit_factor

        logit_scores[key][int(sid)]['create'] = float(create_score.split('@')[1]) 
        logit_scores[key][int(sid)]['exist'] = float(exist_score.split('@')[1]) 
        logit_scores[key][int(sid)]['move'] = float(move_score.split('@')[1]) 
        logit_scores[key][int(sid)]['destroy'] = float(destroy_score.split('@')[1]) 
        logit_scores[key][int(sid)]['prior'] = float(prior_score.split('@')[1]) 
        logit_scores[key][int(sid)]['post'] = float(post_score.split('@')[1]) 

        best_state, best_score = 'none', float(-np.inf)
        for state_type in ['create', 'exist', 'move', 'destroy', 'prior', 'post']:
            if best_score < logit_scores[key][int(sid)][state_type]:
                best_state = state_type
                best_score = logit_scores[key][int(sid)][state_type]
            logit_scores[key][int(sid)][state_type] *= mul_factor
            
        assert(pred_state == best_state), f"pred_state \"{pred_state}\" doesn't match best_state \"{best_state}\" for: {line}"

lines = []
for key in logit_scores:
    pid, ent = key.split('@')
    state_preds = function_to_decode(CRFLayer, key, logit_scores)
    for sid, state_pred in enumerate(state_preds):
        sid += 1
        line = f'{pid}\t{sid}\t{ent}\t{label_to_final_state[state_pred]}\t{state_pred}\t{"?"}\n'
        lines.append(line)

write_lines_to_file(lines, os.path.join(logs_path, f'{split_type}_predictions_best_post_crf.tsv'))




