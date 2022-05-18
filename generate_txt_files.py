import argparse
import csv
import os
import random

def process_tsv(tsv_filename):
    """
    This function takes label.txt and outputs textlines of the txt files.

    input file format:
    clip_id	text	label
    w5Jyq3XMbb3WwiKQ_0000	लिबर ऑफिस impress में एक प्रस्तुति document बनाना और बुनियादी formatting के इस spoken tutorial में आपका स्वागत है	0000000222221001112211001111221112221111100
    w5Jyq3XMbb3WwiKQ_0001	इस tutorial में हम impress window के भागों के बारे में सीखेंगे और कैसे स्लाइड इन्सर्ट करें और कॉपी करें फॉन्ट तथा फॉन्ट को फॉर्मेट करना सीखेंगे	002221000011022221111111000111122110000121100011111122111100000
    
    output example:
    [
        "w5Jyq3XMbb3WwiKQ_0000.npy 0000000222221001112211001111221112221111100 43",
        "w5Jyq3XMbb3WwiKQ_0001.npy 002221000011022221111111000111122110000121100011111122111100000 63",
    ]
    """
    textlines = []
    with open(tsv_filename) as tsv_f:
        for row in csv.reader(tsv_f, delimiter="\t"):
            textlines.append(os.path.join("npys", f'{row[0]+".npy"} {row[2]} {len(row[2])}'))
    return textlines[1:]

def split_data(textlines):
    random.seed(42)
    random.shuffle(textlines)

    split_ratios = [0.7, 0.85]
    split_indices = [int(split_ratios[0]*len(textlines)), int(split_ratios[1]*len(textlines))]
    
    return textlines[:split_indices[0]], textlines[split_indices[0]:split_indices[1]], textlines[split_indices[1]:]

def write_textlines_to_file(textlines, filename):
    with open(filename, "w") as f:
        f.write("\n".join(textlines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_filename", type=str)
    args = parser.parse_args()

    textlines = process_tsv(args.tsv_filename)

    train, dev, test = split_data(textlines[:])

    write_textlines_to_file(train, "train.txt")
    write_textlines_to_file(dev, "dev.txt")
    write_textlines_to_file(test, "test.txt")