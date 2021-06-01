def export_timestamps(start_labels, end_labels, save_file):
    assert len(start_labels) == len(end_labels)
    with open(save_file, 'w') as ff:
        for start_lab, end_lab in zip(start_labels, end_labels):
            ff.write(f'{start_lab}\t{end_lab}\n')
    return
