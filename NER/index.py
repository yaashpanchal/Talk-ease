from itertools import count
import streamlit as st
import sounddevice as sd
import scipy as sc
import soundfile as sf
import numpy as np
from multiprocessing import Process
import requests
from streamlit_lottie import st_lottie
import pickle
import random
import phodel
import os
from io import BytesIO
import io
import streamlit.components.v1 as components
import librosa
from pydub import AudioSegment

parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "st_audiorec")
st_audiorec = components.declare_component("st_audiorec", path=build_dir)
import eng_to_ipa as ipa

with (open("NER/labeledParagraphs.pickle", "rb")) as openfile:
    labeledParagraphs = pickle.load(openfile)

with (open("NER/modifiableWords.pickle", "rb")) as openfile:
    modifiableWords = pickle.load(openfile)

with (open("NER/wordPosPhonemeDict.pickle", "rb")) as openfile:
    wordPosPhonemeDict = pickle.load(openfile)


def load_lottieurl(url):
    request = requests.get(url)
    if request.status_code != 200:
        return None
    return request.json()

def record():
    print("record started")
    fs = 16000  # sample rate 16000 Hz
    recording = sd.rec(int(SAMPLE_TIME * fs), samplerate=fs, channels=1)
    sd.wait()
    sc.io.wavfile.write(r'C:\Users\Lenovo\Desktop\Speech\datasets\stuttering\dub\Logue-master\output.wav', fs, recording)

    # Convert WAV to FLAC using pydub
    sound = AudioSegment.from_wav(r'C:\Users\Lenovo\Desktop\Speech\datasets\stuttering\dub\Logue-master\output.wav')
    sound.export(r'C:\Users\Lenovo\Desktop\Speech\datasets\stuttering\dub\Logue-master\output.flac', format='flac')

    print("record ended")

def next(prev, curr):
    st.session_state[prev] = False
    st.session_state[curr] = True
    st.experimental_rerun()

def substitute_paragraph(phoenemes):
    phoenemes = phoenemes[0]
    # paragraph = labeledParagraphs[random.randint(0,4)]
    paragraph = labeledParagraphs[0]
    paragraph_text = ""
    index = 0
    for word in paragraph:
        if word[1] == "PUNCT":
            paragraph_text = paragraph_text[0: -1]
        if word[1] in modifiableWords:
            i = index
            for j in range(len(phoenemes)):
                if (word[1], phoenemes[i]) in wordPosPhonemeDict.keys():
                    possibleWords = wordPosPhonemeDict[(word[1], phoenemes[i])]
                    paragraph_text += possibleWords[random.randint(0, len(possibleWords) - 1)]
                    paragraph_text += " "
                    index = (index + 1) % len(phoenemes)
                    break
                i = (i + 1) % len(phoenemes)
            else:
                paragraph_text += (word[0] + " ")
        else:
            paragraph_text += (word[0] + " ")
    return paragraph_text

def predict_stutter():
    stuttered_phonemes = st.session_state.phoenemes[0]
    stuttered_phonemes_maps = {}
    print(stuttered_phonemes)
    words = SAMPLE_PARAGRAPH.split()
    print("hi")
    print(words)
    processed_words = []
    for word in words:
        phonemesOfWord = ipa.convert(word)
        for phoneme in phonemesOfWord:
            print(phoneme)
            if phoneme in stuttered_phonemes:
                processed_words.append("<u>" + word + "</u>")
                break
        else:
            processed_words.append(word)
    return (" ").join(processed_words)

SAMPLE_PARAGRAPH = """
"Dad who is talking on the phone mentioned eating bug eggs with jam this summer. My funny and dippy cat living in the ocean likes to listen to the rhyme of the flute. This is why you don’t give him tips with carrots. Yesterday, at eight, the sky was pink. Five bees and one wolf fought with three monkeys and four birds next to the gym. A boy ended up shouting in their ears and cured the wolf’s arm with pencils and scissors. In the future, I will buy a pair of leather thongs and hide them on a beach where no one can open this treasure."  
"""

# time user has to read the sample paragraph
SAMPLE_TIME = 60
fs = 16000

book_animation = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_4XmSkB.json")

# initialize state variables
if 'read_expended' not in st.session_state:
    st.session_state.read_expended = True

if 'analyze_expended' not in st.session_state:
    st.session_state.analyze_expended = False

if 'practice_expended' not in st.session_state:
    st.session_state.practice_expended = False

if 'result_expended' not in st.session_state:
    st.session_state.result_expended = False

if 'stuttered_text' not in st.session_state:
    st.session_state.stuttered_text = SAMPLE_PARAGRAPH

if 'phoenemes' not in st.session_state:
    st.session_state.phoenemes = ["", 0]

if 'paragraph' not in st.session_state:
    st.session_state.paragraph = ""

if 'finish_record_start' not in st.session_state:
    st.session_state.finish_record_start = False

if 'start_loading_start' not in st.session_state:
    st.session_state.start_loading_start = False

if 'finish_record_prac' not in st.session_state:
    st.session_state.finish_record_prac = False

if 'start_loading_prac' not in st.session_state:
    st.session_state.start_loading_prac = False


st.write("""
<style>
u {
  color: red
}
</style>
""", unsafe_allow_html=True)

# hero
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st_lottie(book_animation)
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.title("Logue")
        st.caption("AI Powered Speech Therapist")

# Task: Intro (maybe use card)
st.text("made by Alan, Henry, Willy")
st.write("Logue is an AI powered digital speech therapist that helps stutterers become better at speaking in an intelligent way: with transformer AI models, phoneme detection, and NLP text generation.")

# step 1
with st.container():
    read = st.expander("Step 1.",
                       expanded=st.session_state.read_expended
                       )
    with read:
        st.title("Read 📖")
        st.markdown("Hello! Welcome to the first step of the therapy. In this section, please read the following paragraph so that we can detect which <u>phonemes</u> you struggle to pronounce. The paragraph is designed to test all the phonemes, so it may not make semantic sense. Please relax and click the 'Start Recording' button and start speaking when you are ready. Click 'Stop' when you finish talking. Click 'Reset' if you want to start over. Click 'Download' if you want to hear what you said. Finally, click 'Submit' button to sumbit your sound file to our slutter phonemes detector.", unsafe_allow_html=True)
        st.markdown("<strong>" + SAMPLE_PARAGRAPH + "</strong>", unsafe_allow_html=True)
        # optional task: can add countdown feature on button
        # optional task: allow user to download the recorded audio
        # record()
        val = st_audiorec(key="start-rec")
        if isinstance(val, dict):
            with st.spinner('retrieving audio-recording...'):
                ind, val = zip(*val['arr'].items())
                ind = np.array(ind, dtype=int)
                val = np.array(val)
                sorted_ints = val[ind]
                stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
                wav_bytes = stream.read()
            # st.audio(wav_bytes, format='audio/wav')
            data, samplerate = sf.read(io.BytesIO(wav_bytes))
            sf.write(r'C:\Users\Lenovo\Desktop\Speech\datasets\stuttering\dub\Logue-master\stereo.flac', data, samplerate)

            st.session_state.finish_record_start = True

        if st.session_state.finish_record_start:
            read_clicked = st.button("Submit",
                                     key="next-button"
                                     )
            if read_clicked:
                st.session_state.start_loading_start = True
                if st.session_state.start_loading_start:
                    st.caption("loading...")

                # Remove the ffmpeg command for conversion ffmpeg.input('stereo.flac').output('output.flac', ac=1).run(overwrite_output=True)

                sound = AudioSegment.from_file('stereo.flac')
                sound.export('output.flac', format='flac')

                audio, sr = sf.read(r'C:\Users\Lenovo\Desktop\Speech\datasets\stuttering\dub\Logue-master\output.flac')
                sf.write(r'C:\Users\Lenovo\Desktop\Speech\datasets\stuttering\dub\Logue-master\output.wav', audio, sr, 'PCM_16')

                x, sr = librosa.load('output.wav', sr=48000)
                y = librosa.resample(x, 48000, 16000)
                sf.write(r'C:\Users\Lenovo\Desktop\Speech\datasets\stuttering\dub\Logue-master\output.flac', y, 16000)

                t, t_s = phodel.getTranscription(SAMPLE_PARAGRAPH)
                phoenemes = phodel.getPhonemes(t, t_s)
                print(phoenemes)
                st.session_state.phoenemes = phoenemes
                paragraph = substitute_paragraph(phoenemes)
                st.session_state.paragraph = paragraph
                print(paragraph)
                st.session_state.start_loading_start = False
                # task: predict_stutter()
                next("read_expended", "analyze_expended")

# step 2
with st.container():
    read = st.expander("Step 2.",
                       expanded=st.session_state.analyze_expended
                       )
    with read:
        st.title("Analyze 📋")
        st.write("Words you stuttered on:")
        st.markdown(predict_stutter(), unsafe_allow_html=True)  # Task: underline words stuttered on
        st.write("Phonemes you stuttered on:")
        st.text(st.session_state.phoenemes[0])  # Task: show phonemes
        st.write("Fluency score out of 100(The higher you get, the less you stuttered):" + str(int(st.session_state.phoenemes[1] * 100)))
        analyze_clicked = st.button("Next",
                                    key="analyze-button"
                                    )
        if analyze_clicked:
            next("analyze_expended", "practice_expended")

# step 3
with st.container():
    read = st.expander("Step 3.",
                       expanded=st.session_state.practice_expended
                       )
    with read:
        st.title("Practice 🎙")
        st.write("Our AI generated a paragraph below based on the phonemes you stuttered on the most. The paragraph is designed to be a little diffcult for you to read because we reused phonemes you stuttered on the most when generating the paragraph. Practice reading out the paragraph will help you from stuttering. Click the 'Start Recording' button and start the practice when you are ready. Click 'Stop' when you finish talking. Click 'Reset' if you want to start over. Click 'Download' if you want to hear what you said. Finally, click 'Submit' button to sumbit your sound file to our slutter phonemes detector. You can do it!")
        st.markdown("<strong>" + st.session_state.paragraph + "</strong>", unsafe_allow_html=True)

        val = st_audiorec(key="prac-rec")
        if isinstance(val, dict):
            with st.spinner('retrieving audio-recording...'):
                ind, val = zip(*val['arr'].items())
                ind = np.array(ind, dtype=int)
                val = np.array(val)
                sorted_ints = val[ind]
                stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
                wav_bytes = stream.read()
            # st.audio(wav_bytes, format='audio/wav')
            data, samplerate = sf.read(io.BytesIO(wav_bytes))
            sf.write(r'C:\Users\Lenovo\Desktop\Speech\datasets\stuttering\dub\Logue-master\stereo.flac', data, samplerate)
            st.session_state.finish_record_prac = True

        if st.session_state.finish_record_prac:
            practice_clicked = st.button("Submit",
                                         key="practice-button"
                                         )
            if practice_clicked:
                st.session_state.start_loading_prac = True
                if st.session_state.start_loading_prac:
                    st.caption("loading...")

                # Remove the ffmpeg command for conversion
                t, t_s = phodel.getTranscription(SAMPLE_PARAGRAPH)
                phoenemes = phodel.getPhonemes(t, t_s)
                print(phoenemes)
                st.session_state.phoenemes = phoenemes
                paragraph = substitute_paragraph(phoenemes)
                st.session_state.paragraph = paragraph
                print(paragraph)
                st.session_state.start_loading_prac = False
                # task: predict_stutter()
                next("practice_expended", "result_expended")

# result
with st.container():
    read = st.expander("Result",
                       expanded=st.session_state.result_expended
                       )
    with read:
        st.title("Result 🤗")
        st.write("Fluency score out of 100(The higher you get, the less you stuttered):" + str(int(st.session_state.phoenemes[1] * 100)))


# Footer
# with st.container():
#   st.caption("Animation made by [Alessio Ciancio](https://lottiefiles.com/alessiociancio) from www.lottiefiles.com")
