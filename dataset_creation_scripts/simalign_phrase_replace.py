import re
from string import punctuation

from nltk.translate.phrase_based import phrase_extraction
from nltk.tokenize import word_tokenize
from simalign import SentenceAligner


punct = set(punctuation)

# Load aligner
aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="maifr")

# Read in Gernam sentences
de_sents = []
de_sents_raw = []
with open("data/OpenSubtitles3/OpenSubtitles.de-tr.de", encoding="utf-8") as file_in:
    for line in file_in:
        de_sents_raw.append(line.strip())
        line = re.sub(r"\(\w+\) ", "", line)
        sent = word_tokenize(line)
        de_sents.append(sent)

# Read in Turkish sentences
tr_sents = []
tr_sents_raw = []
with open("data/OpenSubtitles3/OpenSubtitles.de-tr.tr", encoding="utf-8") as file_in:
    for line in file_in:
        tr_sents_raw.append(line.strip())
        sent = word_tokenize(line)
        tr_sents.append(sent)


print(len(de_sents))
print(len(tr_sents))

mm = ["rev"]  # This method seemed to work best
orig_sents = set(de_sents_raw + tr_sents_raw)

print("Now starting phrase swapping")

x = 0 # Sentence counter
alignments = [1]
new = []
while alignments:
    with open(f"data/new_sents/new_open_subtitles3/phrase_switch_opensubtitles3_{x}_{x+1000}.txt", "w", encoding="utf-8") as file_out:
        # Get alignments
        alignments = [
            aligner.get_word_aligns(d, t) if d and t else {i: [] for i in mm}
            for d, t in zip(de_sents[b:b+1000], tr_sents[b:b+1000])
        ]
        alignments = [list(set.intersection(*[set(i[m]) for m in mm])) for i in alignments]
        # Swap all options
        for i in range(len(alignments)):
            d = de_sents[i+b]  # German sentence
            t = tr_sents[i+b]  # Turkish sentence
            # Filter out alignments that are probably faulty
            if abs(len(d) - len(t)) > 3 or len(t) < 3 or len(d) < 3:
                continue
            a = alignments[i]
            # For the phrase extraction, the sentences need to be joined
            de = " ".join(d)
            tr = " ".join(t)
            p = phrase_extraction(de, tr, a)
            # Additional filtering to hopefully arrive at some sensible options
            j0 = set(j[0] for j in alignments[i])
            j1 = set(j[1] for j in alignments[i])
            p = [x for x in p if (x[0][0] in j0) and (x[1][0] in j0) and (x[0][1] in j1) and (x[1][1] in j1)]
            # Filter out phrases with punctuation in them
            p = [j for j in p if not any(x in punct for x in " ".join([j[2], j[3]]))]
            # Filter out phrases that just swap the entire sentence
            p = [x for x in p if not x[1][1] == len(t) - 1]
            # Filter out phrases with absolute difference in lenght more that 2
            p = [j for j in p if abs((j[0][1] - j[0][0]) - (j[1][1] - j[1][0])) < 3]
            if p:
                # Replace all options
                for phrase in p:
                    to_replace = phrase
                    new = de_sents_raw[i+b].replace(to_replace[2], to_replace[3])
                    if not new in orig_sents:
                        file_out.write(f"{new}\n")
        print(f"sents {x} bis {x+1000} done")
        x += 1000

