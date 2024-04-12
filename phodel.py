import requests
import json
import torch
import eng_to_ipa as ipa
import soundfile as sf

def map_to_array(batch):
    speech_array, _ = sf.read(batch)
    return speech_array

def getTranscription(originalText):
    # Replace this with the URL of your deployed model on Hugging Face's Inference API
    inference_api_url = "https://api-inference.huggingface.co/models/facebook/wav2vec2-lv-60-espeak-cv-ft"

    # Load the audio file
    data = map_to_array('output.flac')

    # Prepare the input for the Inference API
    input_data = json.dumps({
        "inputs": {
            "source": data.tolist()
        }
    })

    # Send a POST request to the Inference API
    headers = {"Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "Content-Type": "application/json"}
    try:
        response = requests.post(inference_api_url, headers=headers, data=input_data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(e)
        return [], []

    # Process the response
    response_data = response.json()
    if 'errors' in response_data:
        print(response_data['errors'])
        return [], []

    transcription_s = response_data[0]['transcription']

    # Convert the original text to IPA
    transcription = [ipa.convert(originalText)]

    return transcription, transcription_s

def getPhonemes(trans, trans_s):
    searched = set()
    searched.add(" ")
    fluencyScore = 0
    stutterPhonemes = []
    for phoneme in trans[0]:
        if phoneme not in searched:
            searched.add(phoneme)
            og = trans[0].count(phoneme)
            vc = trans_s[0].count(phoneme)
            score = float(vc) / float(og)
            if score > 1.5:
                stutterPhonemes.append(phoneme)
                fluencyScore += 1
                if score > 2:
                    fluencyScore += 1
    fluencyScore = (len(searched)  - fluencyScore) / len(searched)
    return stutterPhonemes, fluencyScore

def extract_features(audio):
    y, sr = librosa.load(audio_path, sr=None)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_features = np.concatenate((mfcc_mean, mfcc_std))

    # Extract Spectrogram features
    spectrogram = np.abs(librosa.stft(y))
    spectrogram_mean = np.mean(spectrogram, axis=1)
    spectrogram_std = np.std(spectrogram, axis=1)
    spectrogram_features = np.concatenate((spectrogram_mean, spectrogram_std))

    # Extract additional features (e.g., pitch, energy, zero-crossing rate)
    pitch = librosa.core.piptrack(y=y, sr=sr)[0]
    pitch_mean = np.mean(pitch, axis=1)
    pitch_std = np.std(pitch, axis=1)

    # Combine all features
    combined_features = np.concatenate((mfcc_features, spectrogram_features, pitch_mean, pitch_std)).reshape(1, -1)

    return combined_features

def count_attributes(model1, features):
    predictions = model.predict(features)
    attribute_counts = {}
    for i, label in enumerate(target_labels):
        attribute_counts[label] = int(predictions[0][i])
    return attribute_counts
