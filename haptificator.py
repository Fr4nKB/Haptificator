# MIT License
# (c) 2025 Fr4nKB
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import sys
import librosa
import numpy as np
from pydub import AudioSegment
from pathlib import Path
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis

SAMPLE_RATE = 44100

def get_song_duration_in_ms(file_path):
    audio = AudioSegment.from_file(file_path)
    return len(audio)

def extract_beats_with_timestamps(file_path):
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # get beat frames
    _, beat_frames = librosa.beat.beat_track(y=y, sr=SAMPLE_RATE)
    
    # calculate timestamps for each beat
    beat_times = librosa.frames_to_time(beat_frames, sr=SAMPLE_RATE) * 1000
    
    # get the amplitude of each beat by extracting the rms for each beat frame
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    beat_values = rms[beat_frames]
    
    # combine beat values with respective timestamps
    beats_with_timestamps = list(zip(beat_times, beat_values))
    
    return beats_with_timestamps

def get_wave_data(t, values, start, stop, peak_factor, frequency):
    return peak_factor * np.sin(2 * np.pi * frequency * t[start:start+stop]) * values

def generate_wave(frequency, duration_ms, peak_factor, silence_perc=0.5):
    """
    Generate a sine wave with peak at start, stabilized amplitude, and peak at end, silence at the end.

    Parameters:
    frequency (float): The frequency of the wave in Hz.
    duration_ms (int): The duration of the wave in milliseconds.
    peak_factor (float): The peak amplitude factor.
    silence_perc (float): The percentage of the duration that will be silent, the silent portion is placed after the wave (default 50%).

    Returns:
    AudioSegment: The generated audio segment.
    """
    total_duration_s = duration_ms / 1000.0
    total_silence_duration_ms = duration_ms * silence_perc
    total_silence_duration_s = total_silence_duration_ms / 1000.0
    t = np.linspace(0, total_duration_s, int(SAMPLE_RATE * total_duration_s), endpoint=False)
    
    # calculate silence duration in samples
    silence_duration = int(total_silence_duration_s * SAMPLE_RATE)
    
    # adjust durations for attack, sustain, and release considering silence
    effective_duration = len(t) - silence_duration
    attack_duration = int(0.08 * effective_duration)
    sustain_duration = int(0.8 * effective_duration)
    release_duration = int(0.08 * effective_duration)
    transient_duration = int(0.02 * effective_duration)
    
    total_sound_duration = attack_duration + sustain_duration + release_duration + 2*transient_duration
    if total_sound_duration > effective_duration:
        sustain_duration -= total_sound_duration - effective_duration
    elif total_sound_duration < effective_duration:
        sustain_duration += effective_duration - total_sound_duration

    transient_start = np.linspace(0, 1, num=transient_duration)
    attack = np.linspace(1, 0.25, num=attack_duration)
    sustain = np.linspace(0.25, 0.25, num=sustain_duration)
    release = np.linspace(0.25, 1, num=release_duration)
    transient_end = np.linspace(1, 0, num=transient_duration)
    
    # Generate the sound wave with adjusted frequency for the sustain period
    sound_wave = np.zeros(effective_duration)
    start = 0

    sound_wave[:transient_duration] = get_wave_data(t, transient_start, 0, transient_duration, peak_factor, frequency)
    start += transient_duration

    sound_wave[start:start+attack_duration] = get_wave_data(t, attack, start, attack_duration, peak_factor, frequency)
    start += attack_duration

    sound_wave[start:start+sustain_duration] = get_wave_data(t, sustain, start, sustain_duration, peak_factor, frequency)
    start += sustain_duration

    sound_wave[start:start+release_duration] = get_wave_data(t, release, start, release_duration, peak_factor, frequency)
    start += release_duration

    sound_wave[start:] = get_wave_data(t, transient_end, start, transient_duration, peak_factor, frequency)
    sound_wave = (sound_wave * 32767).astype(np.int16)  # convert to 16-bit PCM format

    audio = AudioSegment(sound_wave.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)    
    # silence after wave
    silence = AudioSegment.silent(duration=total_silence_duration_ms)
    audio_with_silence = audio + silence
    
    return audio_with_silence


def create_vibration_wave(peaks_with_timestamps, song_duration):
    n_peaks = len(peaks_with_timestamps)
    if(n_peaks == 0):
        print("No peaks provided")
        return
    
    max_peak_value = max(peak for _, peak in peaks_with_timestamps)

    # add silent segment from start to first beat
    waves = [AudioSegment.silent(duration=peaks_with_timestamps[0][0], frame_rate=SAMPLE_RATE)]

    # generate a exp decaying wave for each beat
    for counter in range(0, n_peaks):
        timestamp, peak_value = peaks_with_timestamps[counter]
        next_timestamp, _ = (song_duration, 0) if counter + 1 == n_peaks else peaks_with_timestamps[counter + 1]

        duration_ms = next_timestamp - timestamp
        peak_factor = peak_value/max_peak_value

        # amplitude and frequency are adjusted dynamically based on the peak intensity
        wave = generate_wave(120 * peak_factor, duration_ms, peak_factor * 1.2)    # little boost to amplitude
        waves.append(wave)

    sound_sequence = AudioSegment.empty()
    for wave in waves:
        sound_sequence += wave

    return sound_sequence


def convert_to_wav(file_path):
    ogg_audio = AudioSegment.from_file(file_path, format="ogg")
    ogg_audio = ogg_audio.set_frame_rate(SAMPLE_RATE).set_channels(2).set_sample_width(2)
    ogg_audio.export("temp.wav", format="wav")


def merge_vibration_with_sound(original_file_path, wav_file_path, vibration_wave, output_file_path):
    """
    Merge the original audio with the generated vibration wave.
    
    Parameters:
    original_file_path (str): The path to the original audio file.
    wav_file_path (str): The path to the converted WAV file.
    vibration_wave (AudioSegment): The generated vibration wave.
    output_file_path (str): The path to the output audio file.

    """
    wav_audio = AudioSegment.from_file(wav_file_path, format="wav")
    wav_audio = wav_audio.set_frame_rate(SAMPLE_RATE).set_channels(2).set_sample_width(2)

    vibration_wave = vibration_wave.set_frame_rate(SAMPLE_RATE).set_channels(1).apply_gain(0)
    
    left = wav_audio.split_to_mono()[0]
    right = wav_audio.split_to_mono()[1]

    # the vibration wave duration is slightly shorter, adjust left and right channel accordingly
    duration = int(vibration_wave.duration_seconds * 1000)
    left = left[:duration]
    right = right[:duration]
    vibration_wave = vibration_wave[:duration]
    
    # for some reason we need to put vibration_wave in the middle to act as the third channel
    combined_audio = AudioSegment.from_mono_audiosegments(left, vibration_wave, right)
    combined_audio.export(output_file_path, format="ogg")
    
    tmp_file_path = Path("./temp.wav")
    if tmp_file_path.exists() and tmp_file_path.is_file():
        tmp_file_path.unlink()

    # ensure compatibility
    try:
        original_file = OggOpus(original_file_path)
    except:
        original_file = OggVorbis(original_file_path)

    vibration_file = OggVorbis(output_file_path)

    # copy all other tags from original file and add haptic
    vibration_file["ANDROID_HAPTIC"] = "1"
    for tag, value in original_file.tags:
        vibration_file[tag] = value

    vibration_file.save()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python haptificator.py <input_file_path> <output_file_path>")
        sys.exit(1)

    # convert the existing composition to a wav file
    file_path = sys.argv[1]
    output_path = sys.argv[2]
    convert_to_wav(file_path)

    # extract song duration in milliseconds and detect beats
    song_duration = get_song_duration_in_ms("./temp.wav")
    peaks_with_timestamps = extract_beats_with_timestamps("./temp.wav")

    # use beats to create vibration waves
    vibration_wave = create_vibration_wave(peaks_with_timestamps, song_duration)

    # finally merge everything
    merge_vibration_with_sound(file_path, "./temp.wav", vibration_wave, output_path)
