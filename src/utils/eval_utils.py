from typing import List, Dict
import re
def calculate_metrics(predictions: List[List[Dict[str, str]]], targets: List[List[Dict[str, str]]], task='') -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for the given predictions and targets for ABSA.

    Args:
        predictions (List[List[Dict[str, str]]]): List of predicted triplets.
        targets (List[List[Dict[str, str]]]): List of target triplets.
        task (str): The task name for which metrics are calculated.
    
    Returns:
        Dict[str, float]: A dictionary containing precision, recall, and F1 score.
    """
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
        f"precision_{task}" : precision,
        f"recall_{task}" : recall,
        f"f1_{task}" : f1
    }

def parse_absa_string(text: str) -> List[Dict[str, str]]:
	"""
	Parses a string formatted as "[A] aspect [O] opinion [S] sentiment" into a list of dictionaries.
	Each dictionary contains the tag as the key and the corresponding value.
	For example, "[A] [O] [S] [A] harga [O] terjangkau [S] positive [SSEP] [A] fasilitas [O] nyaman [S] positive" becomes:
	[{'A': 'harga', 'S': 'positive', 'O': 'terjangkau'},
	{'A': 'fasilitas', 'S': 'positive', 'O': 'nyaman'}].

	Args:
		text (str): ABSA string output to be parsed.

	Returns:
		List[Dict[str, str]]: List of dictionaries of parsed ABSA output.

	"""
	pattern = r"\[(\w+)\]\s*([^[]+)"
	matches = re.findall(pattern, text)

	result = []
	current_dict = {}

	for tag, content in matches:
		if tag == "SSEP":  # Sentence separator -> Start a new dictionary
			result.append(current_dict)
			current_dict = {}
		else:
			current_dict[tag] = content.strip()

	if current_dict:  # Append the last sentence if it exists
		result.append(current_dict)

	return result

def parse_aoste(text: str) -> List[Dict[str, str]]:
	"""
	Format this: check in:dari jam 8 malam cek out jam 11 malam . karena tidak ada air:negative, wc:kotor:negative
	To this: [{"A": "check in", "O": "dari jam 8 malam cek out jam 11 malam . karena tidak ada air", "S": "negative"},
			  {"A": "wc", "O": "kotor", "S": "negative"}]
	Args:
		text (str): AOSTE string output to be parsed.

	Returns:
		List[Dict[str, str]]: List of dictionaries of parsed AOSTE output.

	"""
	pattern = r"([^:]+):([^,]+):(\w+)"
	matches = re.findall(pattern, text)

	result = []
	for aspect, opinion, sentiment in matches:
		result.append({
			"A": aspect.strip(),
			"O": opinion.strip(),
			"S": sentiment.strip()
		})

	return result

def metrics_instructabsa(y_true, y_pred, is_triplet_extraction=False):
    total_pred = 0
    total_gt = 0
    tp = 0
    if not is_triplet_extraction:
        for gt, pred in zip(y_true, y_pred):
            gt_list = gt.split(', ')
            pred_list = pred.split(', ')
            total_pred+=len(pred_list)
            total_gt+=len(gt_list)
            for gt_val in gt_list:
                for pred_val in pred_list:
                    if pred_val in gt_val or gt_val in pred_val:
                        tp+=1
                        break

    else:
        for gt, pred in zip(y_true, y_pred):
            gt_list = gt.split(', ')
            pred_list = pred.split(', ')
            total_pred+=len(pred_list)
            total_gt+=len(gt_list)
            for gt_val in gt_list:
                gt_asp = gt_val.split(':')[0].strip()

                try:
                    gt_op = gt_val.split(':')[1].strip()
                except:
                    continue

                try:
                    gt_sent = gt_val.split(':')[2].strip()
                except:
                    continue

                for pred_val in pred_list:
                    pr_asp = pred_val.split(':')[0].strip()

                    try:
                        pr_op = pred_val.split(':')[1].strip()
                    except:
                        continue

                    try:
                        pr_sent = gt_val.split(':')[2].strip()
                    except:
                        continue

                    if pr_asp in gt_asp and pr_op in gt_op and gt_sent == pr_sent:
                        tp+=1

    p = tp/total_pred
    r = tp/total_gt
    return p, r, 2*p*r/(p+r), None