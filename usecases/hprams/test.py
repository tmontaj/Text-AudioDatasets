"""
test hprams file
"""
hprams = {
    "splits": ["dev-clean", "dev-other"],
    "text_audio": {
        "reverse": "False",
        "len_": True,
        "batch": 2,
        "threshold": 5,
        "is_spectrogram": False,
        "remove_comma": True,
        "alphabet_size": 26,
        "first_letter": 97,
        "sampling_rate": 16000,
        "buffer_size": 1000,
        "melspectrogram": {
            "spectrogram": {
                "nfft": 800,
                "window": 512,
                "stride": 200
            },
            "melscale": {
                "mels": 80,
                "fmin": 0,
                "fmax": 8000
            }
        }
    },
    "audio_audio": {
        "reverse": "False",
        "batch": 2,
        "threshold": 5,
        "buffer_size": 1000,
        "melspectrogram": {
                "nfft": 800,
                "window": 512,
                "stride": 200,
                "mels": 80,
                "fmin": 0,
                "fmax": 8000
        }
    },"speaker_verification": {
        "batch": 32,
        "threshold": 5,
        "sampling_rate": 16000,
        "buffer_size": 1000,
        "num_recordes": 3,
        "melspectrogram": {
                "nfft": 800,
                "window": 512,
                "stride": 200,
                "mels": 80,
                "fmin": 0,
                "fmax": 8000
        }
    },

}
