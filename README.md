# GurkLM 🥒
Pretraining a language model for **G**erman-T**urk**ish Code-Switching (Gurk)

## Architecture

GURK is an encoder model based on the Transformer architecture.

## Pretraining Data
With the help of parallel data of German and Turkish from the [Opus project](https://opus.nlpl.eu/results/tr&de/corpus-result-table), we perform augmentation to create synthetic code-switching data. In particular:

- Replacing **continuous Turkish sequences** with aligned German continuous sequences and vice versa
- Creating **mixed German-Turkish** words via rule-based augmentation

## Validation Tasks

We are using the [Universal Dependency treebank for Turkish-German Code-Switching](https://github.com/UniversalDependencies/UD_Turkish_German-SAGT/tree/master) as a test set for evaluating performance. 
The corpora contains transcriptions of bilingual conversations which are annotated for instance with morphological features, POS tags and languages codes. <br>
The following tasks are used to evaluate the performance of the pretraining procedure:
- **POS Tagging**: A MLP is trained on the learned token representations to predict POS tags.
- **Language Code Predicition**: A MLP is trained on the learned token represenations to predict the language codes (German, Turkish, Mixed & Other)
- **Masked Language Modelling**: Correct prediction of masked tokens in the test set

## Usage

### Requirements

In order to run the scripts in this repository, please install the requirements as listed in `requirements.txt`.
Additionally, install [pytorch with CUDA](https://pytorch.org/get-started/locally/).

### Packed Dataset creation

To create a packed dataset from a directory of raw text files:

```
python create_dataset.py data_dir out_dir --seq-len SEQ_LEN --file-len FILE_LEN --tokenizer TOKENIZER
```

Arguments:

- `data_dir`: Path to the directory containing the data files, each consisting lines of raw sentences
- `out_dir`: Directory to store the output files at, will be created
- `--seq-len`: Length to which the sequences are to be packed
- `--file-len`: How many lines to store in a file, defaults to 100000
- `--tokenizer`: Identifier of the tokenizer to use, defaults to 'bert-base-cased'

### Model training

To train a model, run the following:

```
python train.py --config path/to/config.json
```

### Learning curve plotting

To plot learning curves for a particular training run, use the following:

```
python plot_learning_curve.py log_file config_file out_dir --plot-steps PLOT_STEPS
```

Arguments:

- `log_file`: Path to the log file containing the log output of the model
- `config_file`: Path to the config file used for the training run
- `out_dir`: Directory to store the output files at
- `--plot-steps`: Number of steps to average over for plotting the training data learning curve

### Evaluation scripts

To run evaluation tasks, use the following:

```
python evaluate.py --config CONFIG --checkpoint CHECKPOINT --output-path OUTPUT_PATH --task TASK --ud-data UD_DATA --lr LR --epochs EPOCHS --batch-size BATCH_SIZE --data-tokenizer DATA_TOKENIZER --model-type MODEL_TYPE
```

Arguments:

- `--config`: Path to the config file used for the training run
- `--checkpoint`: Path to the checkpoint to evaluate
- `--output-path`: Directory to store the evaluation output at
- `--task`: choice from `upos`, `lang-code` and `mlm`
- `--ud-data`: identifier of the UD dataset to use for evaluation
- `--lr`: learning rate to use for training the MLP for `upos` and `lang-code` task
- `--epochs`: number of epochs to train the MLP for `upos` and `lang-code` task for
- `--batch-size`: batch size to use for said training
- `--data-tokenizer`: identifier of the tokenizer to use for tokenization of the evaluation data
- `--model-type`: Either use 'gurk' for custom model or choose a pre-trained BERT model from huggingface.

Example configuration files can be seen in the `configs/` directory.

