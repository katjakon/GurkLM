import ast

from datasets import load_dataset
from torch.utils.data import DataLoader

POS = "upos"
CODE = "lang-code"

def get_pos_mapping(examples):
    """Get mapping of tag indices to human-readable tags."""
    upos = examples.features["upos"].feature
    # Return as dicitonary
    return {i: upos.int2str(i) for i in range(len(upos.names))}

def get_code_mapping(): 
    return {0: 'TR', 1: 'DE', 2: 'OTHER', 3: 'MIXED'}# Hard Coding labels for language code identification.
    
def tokenize_and_align_labels(examples, label_type, label_all_tokens=False, skip_index=-100, tokenizer=None):
    if tokenizer is None:
        raise ValueError("Please set tokenizer!")
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=256)
    labels = []

    if label_type == POS:
        col_name = "upos"
    elif label_type == CODE:
        col_name = "misc"
    else:
        raise ValueError(f"Unknown label type: {label_type}. Choose {POS} or {CODE}")
    for i, label in enumerate(examples[col_name]):

        if label_type == CODE:
            label = [ast.literal_eval(el).get("LangID", "OTHER") if ast.literal_eval(el) is not None else "OTHER" for el in label]
            rev_mapping = {v: k for k, v in get_code_mapping().items()}
            label = [rev_mapping.get(l, 2) for l in label]

        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids : list[int] = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(skip_index)

            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else skip_index)

            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def get_ud_data(ud_name, batch_size, tokenizer, label_type):
    ds = load_dataset("universal_dependencies", name=ud_name, trust_remote_code=True)
    train, val, test = ds.values()
    if label_type in {POS, CODE}:
        # Tokenize
        train_tokenized = train.map(lambda example: tokenize_and_align_labels(example, tokenizer=tokenizer, label_type=label_type),
                                    batched=True, batch_size=batch_size)

        val_tokenized = val.map(lambda example: tokenize_and_align_labels(example, tokenizer=tokenizer,  label_type=label_type),
                                batched=True, batch_size=batch_size)
        
        test_tokenized = test.map(lambda example: tokenize_and_align_labels(example, tokenizer=tokenizer, label_type=label_type),
                            batched=True, batch_size=batch_size,)

        # Filter as needed for subsequent processing
        train_tokenized.set_format(type='torch', columns=list(["input_ids",
                                                            "labels",
                                                            "attention_mask"]))
        val_tokenized.set_format(type='torch', columns=list(["input_ids",
                                                            "labels",
                                                            "attention_mask"]))
        test_tokenized.set_format(type='torch', columns=list(["input_ids",
                                                        "labels",
                                                        "attention_mask"]))

        # Create DataLoaders
        train_dataloader = DataLoader(train_tokenized, batch_size=batch_size)
        val_dataloader = DataLoader(val_tokenized, batch_size=batch_size)
        test_dataloader = DataLoader(test_tokenized, batch_size=batch_size)
    else:
        # Create DataLoaders
        train_dataloader = DataLoader(train, batch_size=batch_size)
        val_dataloader = DataLoader(val, batch_size=batch_size)
        test_dataloader = DataLoader(test, batch_size=batch_size)


    return train_dataloader, val_dataloader, test_dataloader