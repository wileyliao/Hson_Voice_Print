import os
from speechbrain.inference import SpeakerRecognition
import torchaudio
import torch
import time
from concurrent.futures import ThreadPoolExecutor

local_model_dir = r"C:\python\torch\voice_print\ECAPA_TDNN\local_model"

# 初始化 ECAPA-TDNN 模型
model = SpeakerRecognition.from_hparams(
    source=local_model_dir,
    savedir="ecapa_model",
    run_opts={"device": "cuda"}
)


def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform[0, :].unsqueeze(0)
    return waveform, sample_rate

def extract_embedding(audio_path):
    waveform, sample_rate = load_audio(audio_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    embedding = model.encode_batch(waveform)
    return embedding.squeeze(0)

def compute_similarity(audio_path1, audio_path2):
    embedding1 = extract_embedding(audio_path1)
    embedding2 = extract_embedding(audio_path2)
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()

def compare_with_database(single_audio_file, subfolder):
    audio_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.wav')]
    similarities = [compute_similarity(single_audio_file, file) for file in audio_files]

    if len(similarities) > 4:
        similarities = sorted(similarities)[1:-1]


    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    folder_name = os.path.basename(subfolder)
    # print(f"{folder_name}: avg = {avg_similarity:.2f}")
    return folder_name, avg_similarity


def compare_main(audio_file):
    # 主目錄路徑，包含多個子資料夾
    start_time = time.time()
    main_directory = r"C:\python\torch\voice_print\dataset"
    subfolders = [os.path.join(main_directory, folder) for folder in os.listdir(main_directory) if
                  os.path.isdir(os.path.join(main_directory, folder))]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compare_with_database, [audio_file] * len(subfolders), subfolders))
        max_result = max(results, key=lambda x: x[1])
        results_json = "; ".join([f"{name}, {value:.2f}" for name, value in results])

        if (max_result[1]) ** 0.5 * 100 > 60:
            end_time = time.time()
            # print(f'Name: {max_result[0]}')
            # print(f'Conf: {(max_result[1]) ** 0.5 * 100}')

            data_dict = {
                "Data":[
                    {
                        "name": max_result[0],
                        "conf": str(max_result[1])
                    }
                ],
                "Result": results_json,
                "TimeTaken": f"{end_time - start_time:.2f} 秒"
            }

            return data_dict

        else:
            end_time = time.time()
            data_dict = {
                "Data": [
                    {
                        "name": "",
                        "conf": "0"
                    }
                ],
                "Result": results_json,
                "TimeTaken": f"{end_time - start_time:.2f} 秒"
            }

            return data_dict



if __name__=='__main__':
    # 單一音檔路徑
    audio_input = r"C:\python\torch\voice_print\ECAPA_TDNN\high_energy_recording.wav"

    compare_main(audio_input)
