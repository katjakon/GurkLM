import os
import random
import re


def make_locative(word):
    """Add -da to word, with some vowel harmony."""
    # Special case: hard final sound
    if word[-1] in {"c", "d", "f", "k", "p", "q", "s", "t", "x", "z"}:
        consonant = "t"
    else:
        consonant = "d"
    vowel = "e"  # fallback, also the right option for {"e", "i", "ü", "ä", "ö", "y"}
    # Check for last vowel in word
    for c in word[::-1]:
        if c in {"a", "o", "u"}:
            vowel = "a"
            break
    return f"{word}'{consonant}{vowel}"


def make_ablative(word):
    """Turkish ablative is very similar to Turkish Locative case."""
    return make_locative(word) + "n"


def turk_locative(de_sents_raw):
    """Go through German sentences and change some phrases to Turkish locative."""
    new_sents = []
    unaltered_sents = []
    # "auf dem", "auf der"
    pattern = r"(auf|an) (de[mr]) (\w+)"
    for sent in de_sents_raw:
        m = re.findall(pattern, sent)
        if m:
            # replace all eligible phrases
            for i in m:
                new_word = make_locative(i[-1])
                new_sents.append(sent.replace(" ".join(i), new_word))
        else:
            unaltered_sents.append(sent)
    return new_sents, unaltered_sents


def turk_ablative(de_sents_raw):
    """Go through German sentences and change some phrases to Turkish ablative."""
    new_sents = []
    unaltered_sents = []
    # "aus dem", "aus der", "von dem", "von der"
    pattern = r"(aus|von) (de[mr]) ([A-Z][A-Za-z]+)"
    for sent in de_sents_raw:
        m = re.findall(pattern, sent)
        if m:
            # replace all eligible phrases
            for i in m:
                new_word = make_ablative(i[-1])
                new_sents.append(sent.replace(" ".join(i), new_word))
        else:
            unaltered_sents.append(sent)
    return new_sents, unaltered_sents


# Get sentences to to alterations on from directory 
sents_raw = []
for file in os.listdir("data/to_wordform_augment"):
    with open(f"data/to_wordform_augment/{file}", encoding="utf-8") as file_in:
        for line in file_in:
            sents_raw.append(line)

# Split into groups
random.shuffle(sents_raw)
loc_sents = sents_raw[:500000]
abl_sents = sents_raw[500000:1000000]
both_sents = sents_raw[1000000:]

# Apply wordform augmentation
loc_sents, unaltered1 = turk_locative(loc_sents)
abl_sents, unaltered2 = turk_ablative(abl_sents)
loc_sents_interm, unaltered_step1 = turk_locative(both_sents)
both_sents, unaltered3 = turk_ablative(loc_sents_interm + unaltered_step1)

unaltered = unaltered1 + unaltered2 + unaltered3

all_sents = unaltered + loc_sents + abl_sents + both_sents
random.shuffle(all_sents)

# Save all sentences to files of length 10000 sentences
file_lim = 10000
tokens = 0
sents = 0
sent_list = []
for x, sent in enumerate(all_sents, start=1):
    sent_list.append(sent)
    if len(sent_list) == file_lim:
        with open(f"data/wordform_augmented/word_augmented_{x-file_lim}_{x}.txt", "w", encoding="utf-8") as file_out:
            for sent in sent_list:
                file_out.write(sent)
                sents += 1
                tokens += len(sent.split())
            sent_list = []

# ...And the final sentences
x = len(sent_list)
with open(f"data/wordform_augmented/word_augmented_{len(all_sents)-x}_{len(all_sents)}.txt", "w", encoding="utf-8") as file_out:
    for sent in sent_list:
        file_out.write(sent)
        sents += 1
        tokens += len(sent.split())
    sent_list = []

# Write log file
with open("data/augmentation_log.txt", "w", encoding="utf-8") as file_out:
    file_out.write("Augmentation log\n")
    file_out.write(f"Added Turkish locative endings: {len(loc_sents)}\n")
    file_out.write(f"Added Turkish ablative endings: {len(abl_sents)}\n")
    file_out.write(f"Added Turkish both endings: {len(both_sents)}\n")
    file_out.write(f"Unaltered: {len(unaltered)}\n")
    file_out.write(f"Total number of sentences: {sents}\n")
    file_out.write(f"Total number of tokens: {tokens}\n")

