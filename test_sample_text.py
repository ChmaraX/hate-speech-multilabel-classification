import os
from constants import MAX_TOKEN_COUNT, LABEL_COLUMNS
from transformers import BertTokenizerFast as BertTokenizer, logging

from models.bert_based.Model_BERT_Linear_v0 import Model_BERT_Linear_v0

logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BEST_MODEL_PATH = 'benchmark/models/v0_linear_freeze_emb.ckpt'


def load_model():
    trained_model = Model_BERT_Linear_v0.load_from_checkpoint(BEST_MODEL_PATH)
    trained_model.eval()
    trained_model.freeze()
    return trained_model


def test_sample_text(trained_model, tokenizer, sample_text):
    encoding = tokenizer.encode_plus(
        sample_text,
        add_special_tokens=True,
        max_length=MAX_TOKEN_COUNT,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    )

    test_prediction = trained_model(
        encoding["input_ids"], encoding["attention_mask"])
    test_prediction = test_prediction.flatten().numpy()

    print(f"metrics for sample text '{sample_text}':")

    for label, prediction in zip(LABEL_COLUMNS, test_prediction):
        print(f"{label}: {prediction}")


trained_model = load_model()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Women are made for making babies and cooking dinner and nothing else!!!"

test_sample_text(trained_model, tokenizer, text)
