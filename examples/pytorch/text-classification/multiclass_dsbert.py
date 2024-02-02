# from transformers import DistilBertForMultiLabelClassification
from transformers.models.distilbert.configuration_distilbert import DistilBertForMultiLabelConfig
from transformers.models.distilbert.modeling_distilbert_multilabel import DistilBertForMultiLabelClassification
from transformers import AutoTokenizer
import pandas as pd 
import torch.nn.functional as F
import torch
from sklearn.metrics import classification_report


def generate_taxonomy(categories):
    taxonomy_dict = {}

    for category in categories:
        main_category, sub_category = category.split(" >> ")

        if main_category not in taxonomy_dict:
            taxonomy_dict[main_category] = [sub_category]
        else:
            taxonomy_dict[main_category].append(sub_category)

    return taxonomy_dict

def encode_list_of_lists(list_of_lists, label2id):
    encoded_lists = []
    for inner_list in list_of_lists:
        encoded_inner_list = [label2id.index(item) for item in inner_list]
        encoded_lists.append(encoded_inner_list)
    return encoded_lists


def batch(iterable: list, n: int = 1):
    """
    iterate over batches with a certain length
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


data_path = "/Users/filippotessaro/Desktop/wrk/WISEcode/product-categorization/data/processed"
df_train = pd.read_csv(f"{data_path}/catalogs-20240102.clean.train.tsv", sep="\t")
df_test = pd.read_csv(f"{data_path}/catalogs-20240102.clean.test.tsv", sep="\t")

# build categories from train
categories = list(set(df_train["cat2"].to_list()))
categories.sort()
print(f"there are {len(categories)} categories; {categories[:5]}")
taxonomy_dict = generate_taxonomy(categories)
print(f"taxonomy dict: {taxonomy_dict}")

# load model
print("Start loading model and tokenizer")
model_name_or_path = "/Users/filippotessaro/models/wisecat/exp_MLMdoordash30ep_pino_data_clean_10jan_multilevel_ep30"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = DistilBertForMultiLabelClassification.from_pretrained(model_name_or_path)

# labels lists L1 and L2
l1_label2ids = model.config.label2id_l1
id2label_l1 = {id: label for id, label in enumerate(l1_label2ids)}

l2_label2ids = model.config.label2id_l2
id2label_l2 = {id: label for id, label in enumerate(l2_label2ids)}


data = df_test["norm_name"].to_list()
max_seq_len = 256
batch_size = 8


def mask_logits(logits, allowed_indexes):
    batch_size, num_classes = logits.size()

    # Create a mask with ones at allowed indexes
    mask = torch.zeros(batch_size, num_classes)
    for i, allowed_index_list in enumerate(allowed_indexes):
        mask[i, allowed_index_list] = 1

    # Apply the mask to the logits
    masked_logits = logits * mask

    return masked_logits

final_categories_list = []
for batch_ in batch(data, batch_size):
    print("processing batch")
    inputs = tokenizer(batch_, padding=True, truncation=True, return_tensors="pt", max_length=max_seq_len)
    # Make predictions
    outputs = model(**inputs, return_dict=True)
    level_1_logits, level_2_logits = outputs.logits[0], outputs.logits[1]
    
    # normalise L2 logits in range [0,1]
    level_2_logits = F.softmax(level_2_logits, dim=-1)
    
    # argmax of L1 category
    level_1_prediction = torch.argmax(level_1_logits, dim=-1).tolist()
    # get categories
    level_1_categories = [id2label_l1[elem] for elem in level_1_prediction]
    # get list of indexes
    categories_to_keep_in_l2 = [taxonomy_dict[elem] for elem in level_1_categories]
    index_to_keep_in_l2 = encode_list_of_lists(categories_to_keep_in_l2, l2_label2ids)

    revisited_l2_logits = mask_logits(level_2_logits, index_to_keep_in_l2)
    level_2_prediction = torch.argmax(revisited_l2_logits, dim=-1).tolist()
    level_2_categories = [id2label_l2[elem] for elem in level_2_prediction]

    final_categories = [f"{l1} >> {l2}" for l1,l2 in zip(level_1_categories,level_2_categories)]
    
    final_categories_list.extend(final_categories)

assert len(final_categories_list) == len(data) == len(df_test)

df_test["prediction"] = final_categories_list

print("### Classification Report ###")
print(classification_report(df_test["cat2"], df_test["prediction"]))

