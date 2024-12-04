# GurkLM 🥒
Pretraining a language model for **G**erman-T**urk**ish Code-Switching (Gurk)

# To Do

## Implementation transformer & training
- [x] Choose and use pre-trained tokenizer
- [x] Masking: Not only use [MASK], but random token ids & same token id
- [x] Implement saving checkpoints
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

# more is more (data)

- https://github.com/ozlemcek/TuGeBiC
  + \# of tokens 	116688
  + \# of monolingual sentences 	10141
  + \# of bilingual sentences 	4510
  + \# of CS points in bilingual sentences 	8180
  
