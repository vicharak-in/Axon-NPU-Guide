import numpy as np
from rknnlite.api import RKNNLite
import argparse
import soundfile as sf
import sounddevice as sd
import torch
import torch.nn as nn
import time


MAX_LENGTH = 200

vocab = {' ': 19, "'": 1, '-': 14, '0': 23, '1': 15, '2': 28, '3': 11, '4': 27, '5': 35, '6': 36, '_': 30, 
        'a': 26, 'b': 24, 'c': 12, 'd': 5, 'e': 7, 'f': 20, 'g': 37, 'h': 6, 'i': 18, 'j': 16, 'k': 0, 'l': 21, 'm': 17, 
        'n': 29, 'o': 22, 'p': 13, 'q': 34, 'r': 25, 's': 8, 't': 33, 'u': 4, 'v': 32, 'w': 9, 'x': 31, 'y': 3, 'z': 2, '–': 10}

def init_model(model_path):
    if model_path.endswith(".rknn"):
        model = RKNNLite()

        print('--> Loading model')
        ret = model.load_rknn(model_path)
        if ret != 0:
            print('Load RKNN model \"{}\" failed!'.format(model_path))
            exit(ret)
        print('done')

        print('--> Init runtime environment')
        ret = model.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')
    else:
        print('Please provide rknn model only.')
        exit(0)

    return model

def release_model(model):
    model.release()
    model = None

def run_encoder(encoder_model, input_ids_array, attention_mask_array):
    (
        log_duration, 
        input_padding_mask, 
        prior_means, 
        prior_log_variances 
    ) = encoder_model.inference(inputs=[input_ids_array, attention_mask_array])
    return log_duration, input_padding_mask, prior_means, prior_log_variances

def run_decoder(decoder_model, attn, output_padding_mask, prior_means, prior_log_variances):
    waveform  = decoder_model.inference(
        inputs=[attn, output_padding_mask, prior_means, prior_log_variances]
        )[0]
    return waveform

def pad_or_trim(token_id, attention_mask, max_length):
    pad_len = max_length - len(token_id)
    if pad_len <= 0:
        token_id = token_id[:max_length]
        attention_mask = attention_mask[:max_length]

    if pad_len > 0:
        token_id = token_id + [0] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return token_id, attention_mask

def preprocess_input(text, vocab, max_length):
    text = list(text.lower())
    input_id = []
    for token in text:
        if token not in vocab:
            continue
        input_id.append(0)
        input_id.append(int(vocab[token]))
    input_id.append(0)
    attention_mask = [1] * len(input_id)

    input_id, attention_mask = pad_or_trim(input_id, attention_mask, max_length)

    input_ids_array = np.array(input_id)[None,...]
    attention_mask_array = np.array(attention_mask)[None,...]

    return input_ids_array, attention_mask_array

def middle_process(log_duration, input_padding_mask, max_length):
    log_duration = torch.tensor(log_duration)
    input_padding_mask = torch.tensor(input_padding_mask)

    speaking_rate = 1
    length_scale = 1.0 / speaking_rate
    duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
    predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

    predicted_lengths_max_real = predicted_lengths.max()
    predicted_lengths_max = max_length * 2

    indices = torch.arange(predicted_lengths_max, dtype=predicted_lengths.dtype)
    output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
    output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

    attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
    batch_size, _, output_length, input_length = attn_mask.shape
    cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
    indices = torch.arange(output_length, dtype=duration.dtype)
    valid_indices = indices.unsqueeze(0) < cum_duration
    valid_indices = valid_indices.to(attn_mask.dtype).view(batch_size, input_length, output_length)
    padded_indices = valid_indices - nn.functional.pad(valid_indices, [0, 0, 1, 0, 0, 0])[:, :-1]
    attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

    attn = attn.numpy()
    output_padding_mask = output_padding_mask.numpy()
    
    return attn, output_padding_mask, predicted_lengths_max_real

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMS_TTS Python Program', add_help=True)

    parser.add_argument(
        '--encoder_model_path', 
        type=str, 
        required=True, 
        help='model path, should be .rknn model file'
        )
    parser.add_argument(
        '--decoder_model_path', 
        type=str, 
        required=True, 
        help='model path, should be .rknn model file'
        )
    
    args = parser.parse_args()

    encoder_model = init_model(args.encoder_model_path)
    decoder_model = init_model(args.decoder_model_path)

    while(True):
        text = input("Enter text to be turned to speech:\n")
        if(text==''):
            continue
        elif(text=='\exit'):
            break

        st = time.time()

        input_ids_array, attention_mask_array = preprocess_input(text, vocab, max_length=MAX_LENGTH)

        (
            log_duration, 
            input_padding_mask, 
            prior_means, 
            prior_log_variances
        ) = run_encoder(encoder_model, input_ids_array, attention_mask_array)

        (
            attn, 
            output_padding_mask, 
            predicted_lengths_max_real
        ) = middle_process(log_duration, input_padding_mask, MAX_LENGTH)

        waveform = run_decoder(decoder_model, attn, output_padding_mask, 
                                prior_means, prior_log_variances)
        end = time.time()

        total_words = len(text.split())
        print(f"Processed {total_words} words in {(end-st)*1000:.2f}ms")
        print(f"Time per word to speech conversion: {(end-st)*1000/total_words:.2f}ms")

        print('\nPlaying the output...\n')
        sd.play(np.array(waveform[0][:predicted_lengths_max_real * 256]), samplerate=16000)
        sd.wait()  

        audio_save_path = "./output.wav"
        sf.write(
            file=audio_save_path, 
            data=np.array(waveform[0][:predicted_lengths_max_real * 256]), 
            samplerate=16000
            )
        print('The output wav file is saved:', audio_save_path)

    release_model(encoder_model)
    release_model(decoder_model)
    
