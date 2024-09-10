import os
import random
import numpy as np
import torch
import re
import gc
import pickle

from datasets import load_dataset

from transformers import BertTokenizer, BertModel

dev = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

print(dev)

seed = 42

os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cache_dir = "./.cache/datasets"

# datasets = {
#     "imdb": load_dataset("imdb", cache_dir=cache_dir),
#     "yelp": load_dataset("yelp_polarity", cache_dir=cache_dir),
#     "amazon": load_dataset("amazon_polarity", cache_dir=cache_dir),
# }

model_name = "bert-base-uncased"
cache_folder = "./.cache/huggingface"
# bert_tokenizer = BertTokenizer.from_pretrained(
#     model_name,
#     cache_dir=cache_folder,
#     device_map="auto",
# )
# bert_model = BertModel.from_pretrained(
#     model_name,
#     cache_dir=cache_folder,
#     device_map="auto",
# )


def clean_text(text):
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Replace single quotes that are not preceded by a backslash
    text = re.sub(r"(?<!\\)'", '"', text)
    return text


def preprocess(data, data_name, sample_size=600_000):
    bert_tokenizer = BertTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_folder,
        device_map="auto",
    )
    bert_model = BertModel.from_pretrained(
        model_name,
        cache_dir=cache_folder,
        device_map="auto",
    )

    data_train_val = data["train"].train_test_split(test_size=0.2, seed=42)

    datasets = {
        "train": data_train_val["train"],
        "val": data_train_val["test"],
        "test": data["test"],
    }

    for dataset_type, dataset in datasets.items():
        print(f"Processing {data_name} {dataset_type} dataset")

        data_labels = dataset["label"]

        if "text" in dataset.column_names:
            data_text = [clean_text(text) for text in dataset["text"]]
        else:
            data_text = [
                clean_text(title).upper() + ": " + clean_text(content)
                for title, content in zip(dataset["title"], dataset["content"])
            ]

        data = list(zip(data_labels, data_text))

        print(f"Checking {data_name} {dataset_type} dataset size")
        print(f"Number of samples before: {len(data)}")
        if len(data) > sample_size:
            print(f"Sampling {data_name} {dataset_type} dataset")
            data = random.sample(data, sample_size)
        print(f"Number of samples after: {len(data)}")

        data_labels, data_text = zip(*data)

        data_labels = list(data_labels)
        data_text = list(data_text)

        print(f"Saving {data_name} {dataset_type} labels")
        labels_file_path = f"word_labels/{data_name}_{dataset_type}_labels.pkl"

        with open(labels_file_path, "wb") as f:
            pickle.dump(data_labels, f)

        token_file_path = f"word_tokens/{data_name}_{dataset_type}_tokens.pt"
        batch_size = 1024 * 3

        if not os.path.exists(token_file_path):
            print(f"Tokenizing {data_name} {dataset_type} dataset")
            tokens = bert_tokenizer(
                data_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )

            print(f"Saving {data_name} {dataset_type} word embeddings")
            torch.save(tokens, token_file_path)
        else:
            tokens = torch.load(token_file_path)

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        print(f"Computing {data_name} {dataset_type} word embeddings")
        with torch.no_grad():
            for i in range(0, input_ids.size(0), batch_size):
                batch_input_ids = input_ids[i : i + batch_size].to(dev)
                batch_attention_mask = attention_mask[i : i + batch_size].to(dev)
                outputs = bert_model(
                    batch_input_ids, attention_mask=batch_attention_mask
                )
                batch_word_embeddings = outputs.last_hidden_state.cpu()

                torch.save(
                    batch_word_embeddings,
                    f"word_embeddings/{data_name}_{dataset_type}_batch_{i//batch_size}.pt",
                )

                del (
                    batch_word_embeddings,
                    outputs,
                    batch_input_ids,
                    batch_attention_mask,
                )
                torch.cuda.empty_cache()
                gc.collect()


# for dataset_name, dataset in datasets.items():
#     preprocess(dataset, dataset_name)

preprocess(load_dataset("imdb", cache_dir=cache_dir), "imdb")

# import gc

# gc.collect()

# torch.cuda.empty_cache()

# preprocess(load_dataset("yelp_polarity", cache_dir=cache_dir), "yelp")

# gc.collect()

# torch.cuda.empty_cache()

# preprocess(load_dataset("amazon_polarity", cache_dir=cache_dir), "amazon")
