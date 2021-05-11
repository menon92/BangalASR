'''Training transformer netwrok on bangla speech data 
'''
import os
import tensorflow as tf
from tensorflow import keras

from . model import Transformer
from . import utils
from . utils import VectorizeChar
from . import dataset
from . import config as cfg


# Set seed for experiment reproducibility
seed = 777
tf.random.set_seed(seed)


class DisplayOutputs(keras.callbacks.Callback):
    '''Display model outut after each specefied epochs'''
    def __init__(
        self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28
    ):
        """Displays a batch of outputs after every epoch

        Args:
            batch: A test batch containing the keys "source" and "target"
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0:
            return
        source = self.batch["source"]
        target = self.batch["target"].numpy()
        bs = tf.shape(source)[0]
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()
        for i in range(bs):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text.replace('-','')}")
            print(f"prediction: {prediction}\n")


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    '''Learning reate scheduler'''
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """ linear warm up - linear decay """
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / (self.decay_epochs),
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        return self.calculate_lr(epoch)


def train():
    '''Training speech model 
    '''
    # load data from raw audio file
    data, unique_chars = dataset.get_data(
        utt_path=cfg.UTT_SPK_PATH,
        flac_audio_dir=cfg.FLAC_AUDIO_DIR
    )

    max_target_len = 200  # all transcripts in out data are < 200 characters
    vectorizer = VectorizeChar(max_target_len)
    print("vocab size", len(vectorizer.get_vocabulary()))
    
    # split data into train validation
    data = data
    split = int(len(data) * cfg.TRAINING_DATA_PERCENTAGES)
    train_data = data # data[:split]
    test_data = data[split:]
    
    ds = utils.create_tf_dataset(train_data, bs=cfg.TRAIN_BATCH_SIZE)
    val_ds = utils.create_tf_dataset(test_data, bs=cfg.VALID_BATCH_SIZE)
    
    # take test sample
    batch = next(iter(val_ds))

    # The vocabulary to convert predicted indices into characters
    idx_to_char = vectorizer.get_vocabulary()
    display_cb = DisplayOutputs(
        batch, idx_to_char, target_start_token_idx=2, target_end_token_idx=3
    )  # set the arguments as per vocabulary index for '<' and '>'

    # init transformer model
    model = Transformer(
        num_hid=200,
        num_head=2,
        num_feed_forward=400,
        target_maxlen=max_target_len,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=108,
    )

    # define loss matric. label_smoothing is important because 
    # class distribution is not equal
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1,
    )

    # learning reate scheduler
    learning_rate = CustomSchedule(
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=len(ds),
    )

    # model checpoint saving callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=cfg.CHECKPOINT_PATH, 
        verbose=0, 
        save_weights_only=True,
        save_freq=5
    )
    # set optimzer
    optimizer = keras.optimizers.Adam(learning_rate)

    # compile the model
    model.compile(optimizer=optimizer, loss=loss_fn)

    # resueme already trained model
    if cfg.RESUME_TRAINING:
        print(f'Model is resuming from {cfg.RESUEM_MODEL_DIR} ...')
        latest = tf.train.latest_checkpoint(cfg.RESUEM_MODEL_DIR)
        model.load_weights(latest)

    # start training the model
    model.fit(
    	ds,
    	validation_data=val_ds,
    	callbacks=[display_cb, checkpoint_cb],
    	initial_epoch=cfg.INITIAL_EPOCHS,
        epochs=cfg.EPOCHS
    )
