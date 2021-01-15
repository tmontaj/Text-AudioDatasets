hprams = {
    "splits": ["dev-clean"],
    "text_audio": { 
        "reverse": "False", 
        "batch": 2, 
        "threshold": 5,
        "is_spectrogram": False, 
        "remove_comma": True, 
        "alphabet_size": 26, 
        "first_letter": 96, 
        "sampling_rate": 16000, 
        "buffer_size": 1000,
        "melspectrogram":{
            "spectrogram": {
                "nfft":800, 
                "window":512, 
                "stride":200
            },
            "melscale": {
                "mels": 80, 
                "fmin": 0, 
                "fmax": 8000
            }  
        }
    },
    "audio_audio":{
        "split": "dev-clean",
        "reverse": "False", 
        "batch": 2, 
        "threshold": 5,
        "sampling_rate": 16000, 
        "buffer_size": 1000,
        "melspectrogram":{
            "spectrogram": {
                "nfft":800, 
                "window":512, 
                "stride":200
            },
            "melscale": {
                "mels": 80, 
                "fmin": 0, 
                "fmax": 8000
            }  
        }
    },
    
}