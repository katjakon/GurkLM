# GurkLM 🥒
Pretraining a language model for **G**erman-T**urk**ish Code-Switching (Gurk)

# To Do

## Implementation transformer & training
- [ ] Choose and use pre-trained tokenizer
- [ ] Implement masking regiment
- [ ] Implement saving checkpoints
- [x] Implement positional encodings

## Pretraining data

- [ ] Choose & download corpora
- [ ] Preprocessing & filtering 
    - very short and very long sequences
    - de-duplication
    - tokenization (splitting punctuation?)
- [ ] Train alignment model
- [ ] Perform data augmentation
- [ ] Create rules & perform rule-based augmentation

## Validation tasks
- [ ] Construct validation tasks
- [ ] Establish upper bound (existing pretrained model)
- [ ] Establish lower bound (random baseline, majority baseline)