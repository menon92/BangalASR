import os
from glob import glob
from tqdm import tqdm

# character we will filter from text sequence, this can be change according
# to your needs
FILTER_CHARS = [
    '"', '%', "'", ',', '-', '.', '/', '\x93', '\x94', '\u200c', '\u200d', '‘', 
    '’', '“', '”', '…', '!', ':'
]

# english character set to discard
ENGLISH = set([
    '0', '1', '2', '3', '4', '5', 'B', 'L', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 
    'g', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z',
])

def clean(text):
    '''Clean text'''
    for c in FILTER_CHARS:
        if c in text:
            text = text.replace(c, '')
    return text


def is_contrain_english_chars(text):
    if ENGLISH.intersection(set(text)):
       return True
    return False


def convert_flac_to_wav(flac_audio_files):
    '''convert flac file to wav file
    sudo apt install sox
    '''
    try:
        print(f"Converting flac to wav")
        for f in tqdm(flac_audio_files):
            cmd = f"sox {f} {f.split('.')[0] + '.wav'}"
            os.system(cmd)
        print('done')
    except Exception as e:
        print(e)


def get_data(
    utt_path,
    flac_audio_dir,
    n_data_to_take=1000,
    max_sequece_to_take=40,
    show_summary=True
):
    '''Get data summary from utt, flac audio dir
    
    Args:
        utt_path: str, path to utt csv file
        flac_audio_dir: str, path to flac audio dir
        n_data_to_take: int, number of data to take
        max_sequece_to_take: int, max text sequence to take
        show_summary: bool, whether to show summary or not
    
    Returns:
        Tuple: data, unique_chars
    '''
    _take = 1000
    flac_audio_files = glob(flac_audio_dir + '/*/*.flac')[:_take*2]

    print(f"Converting flac to wav")
    for f in tqdm(flac_audio_files):
        cmd = f"sox {f} {f.split('.')[0] + '.wav'}"
        os.system(cmd)
    print('done')
    
    flac_list = [
        os.path.splitext(os.path.basename(_file))[0] 
        for _file in flac_audio_files
    ]
    flac_set = set(flac_list)

    data = []
    unique_chars = set()
    max_text_len = 0
    max_text = ''
    en_bn_mixed = 0
    takes = 0
    with open(utt_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip(' \n')
            line = line.split('\t')
            file_name, text= line[0], line[2]
    
            if takes >= _take:
                break

            if file_name in flac_set:
                text = clean(text)
                # skip text which has > 30 chars
                if len(text) > max_sequece_to_take:
                    continue
                
                # skip english text
                if is_contrain_english_chars(text):
                    en_bn_mixed += 1
                    continue
                
                file_abs_path = flac_audio_files[flac_list.index(file_name)].split('.')[0] + '.wav'
                data.append({'audio': file_abs_path, 'text': text})
                
                # create unique chars set
                for c in text:
                    unique_chars.add(c)
                
                # find max text sequence lenght, text
                text_len = len(text)
                if max_text_len < text_len:
                    max_text_len = text_len
                    max_text = text
                
                takes += 1
                
    unique_chars = sorted(unique_chars)

    if show_summary:
        print(f'flac audio files: {len(flac_audio_files)}')
        print(f'flac_dic        : {len(flac_set)}')
        print(f'utt entry       : {len(lines)}')
        print(f'unique chars    : {len(unique_chars)}')
        print(f'data            : {len(data)}')
        print(f"max text length : {max_text_len}")
        print(f'max text        : {max_text}')
        print(f'en bn mixed     : {en_bn_mixed}')

    return data, unique_chars
