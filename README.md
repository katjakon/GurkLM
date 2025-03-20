# GurkLM 🥒
Pretraining a language model for **G**erman-T**urk**ish Code-Switching (Gurk)

## Pretraining Data
With the help of parallel data of German and Turkish from the [Opus project](https://opus.nlpl.eu/results/tr&de/corpus-result-table), we perform augmentation to create synthetic code-switching data. In particular:
- Replacing **individual Turkish words** with aligned German words and vice versa
- Replacing **continuous Turkish sequences** with aligned German continuous sequences and vice versa
- Creating **mixed German-Turkish** words via rule-based augmentation

For the data augmenation, we have to first perform alignment for the parallel sentences. The OpenSubtitles corpus is a similar domain as our validation task. It contains around 17M sentences for German (106M tokens) and Turkish (81M tokens). Other corpora are available.

## Validation Tasks

We are using the [Universal Dependency treebank for Turkish-German Code-Switching](https://github.com/UniversalDependencies/UD_Turkish_German-SAGT/tree/master) as a test set for evaluating performance. 
The corpora contains transcriptions of bilingual conversations which are annotated for instance with morphological features, POS tags and languages codes. <br>
The following tasks are used to evaluate the performance of the pretraining procedure:
- **POS Tagging**: A MLP is trained on the learned token representations to predict POS tags.
- **Language Code Predicition**: A MLP is trained on the learned token represenations to predict the language codes (German, Turkish, Mixed & Other)
- **Masked Language Modelling**: Correct prediction of masked tokens in the test set / Perplexity?

A notebook for evaluating the BERT models and the random baseline can be found here: 
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/katjakon/GurkLM/blob/dev2/eval_notebook.ipynb)<br>


# more is more (data)

- https://github.com/ozlemcek/TuGeBiC
  + \# of tokens 	116688
  + \# of monolingual sentences 	10141
  + \# of bilingual sentences 	4510
  + \# of CS points in bilingual sentences 	8180
