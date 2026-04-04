import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

lemmatizer = WordNetLemmatizer()

# adopted from opencompass
def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r"[\n.,]", text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r"[^\w\s]", "", truncated_text)

    # Remove article
    no_articles = re.sub(r"\b(a|an|the)\b", "", no_punctuation, flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r"\s+", " ", no_articles).strip()

    return cleaned_text


def tokenize_and_lemmatize(text: str) -> list:
    text = text.lower()
    text = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in text]
    return tokens


# adopted from zero_shot_cot
def answer_cleansing(
    method: str,
    answer_trigger: str,
    prediction: str,
    dataset: str,
    task: str,
    input_sample: dict,
) -> str:
    if "{start}" in answer_trigger and "{end}" in answer_trigger and "options" in input_sample:
        options_num = len(input_sample["options"])
        answer_trigger = answer_trigger.format(start="A", end=chr(65 + options_num - 1))
        # print("Answer Trigger: " + answer_trigger)

    # Extract the prediction using the answer_trigger or alternative pattern
    if method == "few_shot":
        preds = prediction.split(answer_trigger)
        answer_flag = True if len(preds) > 1 else False
        prediction = preds[-1]
    elif method == "zero_shot":
        try:
            prediction = prediction.split(answer_trigger)[1].strip()
        except IndexError:
            try:
                prediction = prediction.split(" answer is ")[1].strip()
            except IndexError:
                pass

    # Clean up unwanted phrases in the prediction
    for unwanted_phrase in [
        "I understand",
        "A through J",
        "A through E",
        "A through D",
    ]:
        prediction = prediction.replace(unwanted_phrase, "")

    # Dataset-specific processing
    if dataset in ["slake", "vqa-rad", "path-vqa"]:  # Open-ended and Closed-ended QA
        prediction = general_postprocess(prediction)
        prediction = tokenize_and_lemmatize(prediction)
        return prediction
    elif dataset in ["medqa", "medmcqa", "mmlu_medical"]:  # A-D
        prediction = re.findall(r"\b(A|B|C|D)\b", prediction)
    elif "options" in input_sample and len(input_sample["options"]) > 0:  # Multiple Choice MedXpertQA and sampled
        options_num = len(input_sample["options"])
        options = [chr(65 + i) for i in range(options_num)]
        options_str = r"\b(" + "|".join(options) + r")\b"
        prediction = re.findall(options_str, prediction)
    else:
        raise ValueError("Dataset is not properly defined ...")

    if len(prediction) == 0:
        prediction = []
    else:
        # If there is a "label" and its length is 1, process prediction accordingly
        if "label" in input_sample and len(input_sample["label"]) == 1:
            if method == "few_shot":
                # choose the first or last element based on the answer_flag
                prediction = [prediction[0]] if answer_flag else [prediction[-1]]
            elif method == "zero_shot":
                # choose the first element in list
                prediction = [prediction[0]]
            else:
                raise ValueError("Method is not properly defined ...")

            # Remove trailing period if it exists
            if prediction[0] and prediction[0].endswith("."):
                prediction[0] = prediction[0][:-1]

    return prediction


def compute_accuracy(pred_tokens, gt_tokens):
    return 1 if pred_tokens == gt_tokens else 0


def compute_precision(pred_tokens, gt_tokens):
    label_counter = Counter(gt_tokens)
    predict_counter = Counter(pred_tokens)
    common = label_counter & predict_counter
    true_positive = sum(common.values())

    if len(pred_tokens) > 0:
        return true_positive / len(pred_tokens)
    else:
        return 0.0


def compute_recall(pred_tokens, gt_tokens):
    label_counter = Counter(gt_tokens)
    predict_counter = Counter(pred_tokens)
    common = label_counter & predict_counter
    true_positive = sum(common.values())

    if len(gt_tokens) > 0:
        return true_positive / len(gt_tokens)
    else:
        return 0.0


def compute_f1(pred_tokens, gt_tokens):
    common_tokens = set(pred_tokens) & set(gt_tokens)
    num_common = len(common_tokens)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def calculate_metrics(data):
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []

    for index, item in enumerate(data):
        f1 = compute_f1(item["predict_list"], item["label_list"])
        f1_scores.append(f1)

        accuracy = compute_accuracy(item["predict_list"], item["label_list"])
        accuracies.append(accuracy)

        precision = compute_precision(item["predict_list"], item["label_list"])
        precisions.append(precision)

        recall = compute_recall(item["predict_list"], item["label_list"])
        recalls.append(recall)

    metrics = {
        "accuracy": sum(accuracies) / len(accuracies),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1_scores) / len(f1_scores),
    }
    return metrics
