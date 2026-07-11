import json
import os
import sys
import re
from glob import glob
from typing import List, Dict

def parse_absa_string(text: str) -> List[Dict[str, str]]:
    pattern = r"\[(\w+)\]\s*([^[]+)"
    matches = re.findall(pattern, text)
    result = []
    current_dict = {}
    for tag, content in matches:
        if tag == "SSEP":
            result.append(current_dict)
            current_dict = {}
        else:
            current_dict[tag] = content.strip()
    if current_dict:
        result.append(current_dict)
    return result

def calculate_metrics(predictions: List[List[Dict[str, str]]], targets: List[List[Dict[str, str]]]) -> Dict[str, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for prediction,target in zip(predictions,targets):
        for target_tuple in target:
            if target_tuple in prediction:
                true_positive += 1
            else:
                false_negative += 1
        false_positive += sum(1 for pred in prediction if pred not in target)
    precision = true_positive/(true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive/(true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * recall * precision)/(recall + precision) if (recall + precision) > 0 else 0
    return {
        f"precision" : precision,
        f"recall" : recall,
        f"f1" : f1
    }

def convert_absadict_to_tuples(absa_dict_list: List[Dict[str, str]]) -> List[tuple]:
    tuples_list = []
    for absa_dict in absa_dict_list:
        aspect = absa_dict.get('A', '')
        opinion = absa_dict.get('O', '')
        sentiment = absa_dict.get('S', '')
        tuples_list.append((aspect, opinion, sentiment))
    return tuples_list

def convert_tuples_to_absadict(tuples_list: List[tuple], order: str) -> List[Dict[str, str]]:
    absa_dict_list = []
    for aspect, opinion, sentiment in tuples_list:
        absa_dict = {}
        for char in order:
            if char == 'A':
                absa_dict['A'] = aspect
            elif char == 'O':
                absa_dict['O'] = opinion
            elif char == 'S':
                absa_dict['S'] = sentiment
            else:
                raise ValueError(f"Invalid character '{char}' in order string. Only 'A', 'O', and 'S' are allowed.")
        absa_dict_list.append(absa_dict)
    return absa_dict_list

def convert_output_to_mvp_format(data_list: List[Dict[str, str]], order='aos') -> str:
    result_str = []
    for triplet in data_list:
        triplet_str = [f"[{element.upper()}] {triplet[element.upper()]}" for element in order]
        triplet_str = ' '.join(triplet_str)
        result_str.append(triplet_str)
    return ' [SSEP] '.join(result_str)

def process_file(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    i = 0
    n_sample = 5
    selected_preds = []
    selected_targets = []
    table_freq = {}
    for idx, instance in enumerate(data):
        tuples = convert_absadict_to_tuples(parse_absa_string(instance.get('prediction', '')))
        for t in tuples:
            table_freq[t] = table_freq.get(t, 0) + 1
        i += 1
        if i >= n_sample:
            # Select majority
            majority_preds = [t for t, freq in table_freq.items() if freq > n_sample//2]
            selected_preds.append(majority_preds)

            # Append target
            target_tuples = convert_absadict_to_tuples(parse_absa_string(instance.get('target', '')))
            selected_targets.append(target_tuples)
            
            # Reset for next sample
            i = 0
            table_freq = {}
    
    folder_data_path = os.path.dirname(data_path)
    save_path = os.path.join(folder_data_path, 'voting_results.json')
    
    absa_str_selected_preds = [convert_output_to_mvp_format(convert_tuples_to_absadict(pred, order='AOS'), order='aos') for pred in selected_preds]
    absa_str_selected_targets = [convert_output_to_mvp_format(convert_tuples_to_absadict(target, order='AOS'), order='aos') for target in selected_targets]
    
    voting_results = []
    for j, pred in enumerate(selected_preds):
        voting_results.append(
            {
                'target': absa_str_selected_targets[j],
                'prediction': absa_str_selected_preds[j],
                'target_list': [f'[A] {target[0]} [O] {target[1]} [S] {target[2]}' for target in selected_targets[j]],
                'prediction_list': [f'[A] {p[0]} [O] {p[1]} [S] {p[2]}' for p in pred]
            }
        )
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(voting_results, f, ensure_ascii=False, indent=4)
    
    print(f"Written {save_path}")

def main():
    if len(sys.argv) > 1:
        data_paths = [sys.argv[1]]
    else:
        data_paths = glob('outputs/evals/*/*/mvp/seed_*/*/*/*/inference_results.json')

    for data_path in data_paths:
        process_file(data_path)

if __name__ == '__main__':
    main()
