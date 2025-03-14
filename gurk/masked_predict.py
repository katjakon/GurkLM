import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mask_each_token(input_ids, mask_token_id):
    mask = torch.eye(input_ids.size(1)).bool().to(device)
    masked_ids = input_ids.repeat(mask.size(1), 1).to(device)
    masked_ids[mask] = mask_token_id
    mask_token_index = torch.where(masked_ids == mask_token_id)[1]
    return masked_ids, mask_token_index, mask

def accuracy_at_n(y_true, y_pred, n=3):
    top_n = torch.topk(y_pred, k=n)[1]
    y_true = y_true.view(-1, 1).to(device)
    correct = (y_true == top_n).any(1)
    return torch.sum(correct) / correct.size(0)

def predict_masked_gurk(input_ids, model, mask_token_id):
    masked_ids, _, mask = mask_each_token(input_ids, mask_token_id)
    with torch.no_grad():
        output = model(masked_ids, key_padding_mask=None, pred_mask=torch.flatten(mask))
    pred = output["masked_preds"]
    return pred

def predict_masked_bert(input_ids, model, mask_token_id):
    masked_ids, mask_token_index, _ = mask_each_token(input_ids, mask_token_id)
    output = model(masked_ids).logits
    mask_token_logits = output[0, mask_token_index, :]
    return mask_token_logits
