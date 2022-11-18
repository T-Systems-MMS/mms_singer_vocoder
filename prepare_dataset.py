import os
import argparse
import librosa
import soundfile as sf
from pathlib import Path

def write_filename_songtext_files(output_file: str, sample_dict: dict):
    with open(output_file, "w+") as f:
        for id, _filepath in sample_dict.items():
            songtext = ""
            out_line = id + "|" + songtext
            f.write(out_line + "\n")   

def get_filepaths(ds_folder, file_ending: str) -> list:
    filepaths = list()
    for root, _dirs, files in os.walk(ds_folder):
        for file in files:
            if file.endswith(file_ending):
                filepaths.append(os.path.join(root, file))
    return filepaths

def convert_sr_mono(sample_dict, destination_folder, sr: int):
    print("converting audio...")
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    for id, filepath in sample_dict.items():
        filename = os.path.basename(filepath)
        y, _sr = librosa.load(filepath)
        y = librosa.to_mono(y)
        sf.write(Path(destination_folder, filename), y, sr, subtype='PCM_16')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_folder', default="./data/opera_raw/")
    parser.add_argument('--convert_sr',type=int, default=None)

    a = parser.parse_args()

    filepaths = get_filepaths(a.dataset_folder,".wav")
    samples = dict()
    for id, filepath in enumerate(filepaths):
        samples[os.path.basename(filepath)[:-4]] = filepath
    print(f"{len(samples)} samples found in {a.dataset_folder}.")

    if a.convert_sr is not None:
        convert_sr_mono(samples, Path(a.dataset_folder,"wav_mono_22kHz"), a.convert_sr)

    write_filename_songtext_files(Path(a.dataset_folder, "training.txt"), samples)