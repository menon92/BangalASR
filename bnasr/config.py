import os


# dataset path
UTT_SPK_PATH = 'data/asr_bengali/utt_spk_text.tsv'
FLAC_AUDIO_DIR = 'data/asr_bengali/data'

# training params
EPOCHS = 5
INITIAL_EPOCHS = 0
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

# 99% training and 1% validation
TRAINING_DATA_PERCENTAGES = 0.99
MODEL_SAVE_DIR = 'models'
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, 'bnasr-{epoch:02d}')