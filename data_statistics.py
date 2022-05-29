import collections


def print_statistics_for_dataset(dataset='train', print_full=True):
    assert dataset in ['train', 'dev', 'test']
    filepath = f'./{dataset}.txt'
    files = collections.defaultdict(dict)
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for l in lines:
            contents = l.strip().split()
            length = int(contents[-1])
            if length == 0:
                filepath = contents[0]
                labels = ""
            else:
                filepath, labels = contents[:2]
            filepath = filepath.replace('binary_npys/', '')
            filepath = filepath.replace('.npy', '')
            filename, segment = filepath.split('_')
            files[filename][segment] = labels.strip()

        print(f"Dataset: {dataset}")
        total_len_hindi = 0
        total_len_english = 0
        for filename in files:
            labels = files[filename].values()
            labels_full = ''.join(labels)
            labels_len = len(labels_full)
            if print_full:
                print(f"  File name: {filename}")
                print(f"    Length of labels: {len(labels_full)}")

            len_hindi = sum([1 for c in labels_full if c == '0'])
            len_english = sum([1 for c in labels_full if c == '1'])
            assert len_hindi + len_english == labels_len

            if print_full:
                print(f"    Percentage of Hindi speech: {len_hindi / float(labels_len):.3f}")
                print(f"    Percentage of English speech: {len_english / float(labels_len):.3f}")
                print()

            total_len_hindi += len_hindi
            total_len_english += len_english

        total_length = total_len_hindi + total_len_english
        print(f"Percentage of Hindi speech: {total_len_hindi / float(total_length):.3f}")
        print(f"Percentage of English speech: {total_len_english / float(total_length):.3f}")
        print()


def print_data_statistics():
    datasets = ['train', 'dev', 'test']
    for ds in datasets:
        print_statistics_for_dataset(ds, print_full=False)


if __name__ == '__main__':
    print_data_statistics()
