WANDB = False

BERT_MODEL_NAME = 'bert-base-uncased'

MODEL_CKPT_PATH = './checkpoints/'

DATASET_DIR = './datasets'

LABEL_COLUMNS = ['disability', 'gender', 'non hate',
                 'origin',  'religion', 'sexual_orientation']

TEXT_COLUMN = 'clean_text'

TRAIN_SIZE = 0.7

VAL_SIZE = 0.15

TEST_SIZE = 0.15

LEARNING_RATE = 2e-5

MAX_TOKEN_COUNT = 64

N_EPOCHS = 1

BATCH_SIZE = 32

FREEZE_LAYERS_COUNT = None
