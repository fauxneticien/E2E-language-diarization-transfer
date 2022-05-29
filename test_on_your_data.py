import json
import argparse
from model import *
from data_load import *
import scipy.io.wavfile as wav
import python_speech_features as psf
from collections import OrderedDict
from torch.utils.data import DataLoader
from model_evaluation import *

def get_output(outputs, seq_len):
    output_ = 0
    for i in range(len(seq_len)):
        length = seq_len[i]
        output = outputs[i, :length, :]
        if i == 0:
            output_ = output
        else:
            output_ = torch.cat((output_, output), dim=0)
    return output_

def extract_log_fbank(audio, dim=23, log=True):
    (sr, signal) = wav.read(audio)
    high_frequency = sr / 2 - 200
    features = psf.logfbank(signal, sr, nfilt=dim, lowfreq=20, highfreq=high_frequency, winlen=0.025, winstep=0.01)
    return features


def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--json', type=str, default='test_on_your_data.json')
    parser.add_argument('--test', type=str, default="test.txt", help='testing data, in .txt')

    args = parser.parse_args()

    with open(args.json, 'r') as json_obj:
        config_proj = json.load(json_obj)


    gpu_id = config_proj["gpu_id"]
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    model = X_Transformer_E2E_LID(n_lang=2,
                                  dropout=0.1,
                                  input_dim=23,
                                  feat_dim=256,
                                  n_heads=4,
                                  d_k=256,
                                  d_v=256,
                                  d_ff=2048,
                                  max_seq_len=1000)
    model.to(device)
    pretrained_dict = torch.load(config_proj["pretrained"], map_location='cuda:0')
    new_state_dict = OrderedDict()
    model_dict = model.state_dict()
    dict_list = []
    for k, v in model_dict.items():
        dict_list.append(k)
    for k, v in pretrained_dict.items():
        if k.startswith('module.') and k[7:] in dict_list:
            new_state_dict[k[7:]] = v
        elif k in dict_list:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    test = args.test
    test_set = RawFeatures(test)
    test_data = DataLoader(dataset=test_set,
                            batch_size=1,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=collate_fn_cnn_atten)
    lang = 2
    eer = 0
    correct = 0
    total = 0
    FAR_list = torch.zeros(lang)
    FRR_list = torch.zeros(lang)
    with torch.no_grad():
        for step, (utt, labels, cnn_labels, seq_len) in enumerate(test_data):
            utt_ = utt.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)
            # Forward pass
            outputs, cnn_outputs = model(x=utt_, seq_len=seq_len, atten_mask=None)
            outputs = get_output(outputs, seq_len)
            predicted = torch.argmax(outputs, -1)
            total += labels.size(-1)
            correct += (predicted == labels).sum().item()
            FAR, FRR = compute_far_frr(lang, predicted, labels)
            FAR_list += FAR
            FRR_list += FRR
        acc = correct / total
    print('Current Acc.: {:.4f} %'.format(100 * acc))
    for i in range(lang):
        eer_ = (FAR_list[i] / total + FRR_list[i] / total) / 2
        eer += eer_
        print("EER for label {}: {:.4f}%".format(i, eer_ * 100))
    print('EER: {:.4f} %'.format(100 * eer / lang))

if __name__ == "__main__":
    main()
