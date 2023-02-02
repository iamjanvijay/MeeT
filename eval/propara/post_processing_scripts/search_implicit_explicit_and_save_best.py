import os
import sys
sys.path.append('eval/propara/evaluator')
from evaluator import main
from tqdm import tqdm

states_logs_folder = sys.argv[1]
locations_logs_folder = sys.argv[2]

search_best = True
if search_best:
    print("searching for the best-ratio of implicit-explicit weights...")
    implicit_explicit_cands = []
    for imp_weight in range(0, 11):
        for exp_weight in range(0, 11):
            implicit_explicit_cands.append((imp_weight / 10.0, exp_weight / 10.0))

    best_dev_f1_score, best_weights = -1, -1
    for imp_weight, exp_weight in tqdm(implicit_explicit_cands):
        os.system(f'python eval/propara/post_processing_scripts/correct_state_preds_with_crf.py {imp_weight} {exp_weight} dev {states_logs_folder};')
        os.system(f'python eval/propara/post_processing_scripts/merge_state_and_location_cgli_style.py {os.path.join(states_logs_folder, "dev_predictions_best_post_crf.tsv")} {os.path.join(locations_logs_folder, "dev_predictions_best.tsv")} eval/propara/final_preds/dev_merged_state_and_location.tsv;')

        dev_f1_score = main('eval/propara/golds/answers_dev.tsv', 'eval/propara/final_preds/dev_merged_state_and_location.tsv', None, None, None, only_return_f1=True)
        # print("implicit-weight: {}; explicit-weight: {}".format(imp_weight, exp_weight), "dev-f1-score: {}".format(dev_f1_score))
        if best_dev_f1_score == -1 or dev_f1_score > best_dev_f1_score:
            best_dev_f1_score = dev_f1_score
            best_weights = (imp_weight, exp_weight)

imp_weight, exp_weight = best_weights
print("best setting for implicit-explicit weights:")
print("implicit-weight: {}; explicit-weight: {}".format(imp_weight, exp_weight), "dev-f1-score: {}".format(best_dev_f1_score))

os.system(f'python eval/propara/post_processing_scripts/correct_state_preds_with_crf.py {imp_weight} {exp_weight} dev {states_logs_folder};')
os.system(f'python eval/propara/post_processing_scripts/merge_state_and_location_cgli_style.py {os.path.join(states_logs_folder, "dev_predictions_best_post_crf.tsv")} {os.path.join(locations_logs_folder, "dev_predictions_best.tsv")} eval/propara/final_preds/dev_merged_state_and_location.tsv;')
print("Best dev results:")
main('eval/propara/golds/answers_dev.tsv', 'eval/propara/final_preds/dev_merged_state_and_location.tsv', None, None, None, only_return_f1=False)
print("Saved final dev predictions at:", "eval/propara/final_preds/dev_merged_state_and_location.tsv")

os.system(f'python eval/propara/post_processing_scripts/correct_state_preds_with_crf.py {imp_weight} {exp_weight} test {states_logs_folder};')
os.system(f'python eval/propara/post_processing_scripts/merge_state_and_location_cgli_style.py {os.path.join(states_logs_folder, "test_predictions_best_post_crf.tsv")} {os.path.join(locations_logs_folder, "test_predictions_best.tsv")} eval/propara/final_preds/test_merged_state_and_location.tsv;')
print("\nBest test results:")
main('eval/propara/golds/answers_test.tsv', 'eval/propara/final_preds/test_merged_state_and_location.tsv', None, None, None, only_return_f1=False)
print("Saved final test predictions at:", "eval/propara/final_preds/test_merged_state_and_location.tsv")

