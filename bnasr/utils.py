'''Contain utility code used in both training and inferance time'''

import tensorflow as tf


class VectorizeChar:
    def __init__(self, unique_chars, max_len=50):
        '''Initialize vectorize char object

        Args:
            unique_chars: list of unique charaster used
            max_len: int, maximum length of text sequnece from your dataset
        '''
        self.vocab = (
            [
                "-",
                "#", 
                "<", # use as start token
                ">"  # use as end token
            ]
            + unique_chars
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        '''Make vectorizer as callable object on text

        Args:
            text: str, text sequence
        Returns:
            text sequence represent as number
        '''
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        '''Return all the available vocabulary'''
        return self.vocab


def create_text_ds(data, vectorizer):
    '''Create text Dataset using vectorizer
    
    Args:
        data: list of dictionary in audio: text format
        vectorizer: VectorizeChar object to convert text 
                    sequence to it's vector representation
    
    Returns:
        Tensorflwo Dataset object
    '''
    texts = [_["text"] for _ in data]
    text_ds = [vectorizer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    
    return text_ds


def path_to_audio(path):
    '''Create spectogram from raw audio file. Is also normalize
    the spectogram as positive signals and add padding to create
    batchs of audio data sampes

    Args:
        path: str, path of audio wav file

    Returns:
        Normalized audio data. For detail check `notebooks/EDA-Speech_data.ipynb`

    '''
    # spectrogram using stft
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)    

    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    
    audio_len = tf.shape(x)[0]
    
    # padding to 10 seconds
    pad_len = 2000
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    
    return x


def create_audio_ds(data):
    '''Create Dataset from data audio files

    Args:
        data: list of dictionary in audio: text format
    
    Returns:
        Tensorflwo Dataset object
    '''
    flist = [_["audio"] for _ in data]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    audio_ds = audio_ds.map(
        path_to_audio, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return audio_ds


def create_tf_dataset(data, vectorizer, bs=4):
    '''Create tensorflow compitable data, so that we can directly feed it
    into our asr pipeline
    
    Args:
        data: list of dictionary in audio: text format
        bs: int, batch size

    Returns:
        Tensorflwo Dataset object
    '''
    audio_ds = create_audio_ds(data)
    text_ds = create_text_ds(data, vectorizer)

    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds
