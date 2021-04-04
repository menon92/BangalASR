import bnasr


def test_get_data():
	audio_dir =  'data/asr-bengali/data'
	utt_spk_text = 'data/asr-bengali/utt_spk_text.tsv'

	data, chars = bnasr.dataset.get_data(utt_spk_text, audio_dir)
	print('-' * 24)
	print(chars)


if __name__ == '__main__':
	test_get_data()