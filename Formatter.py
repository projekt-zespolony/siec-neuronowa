import numpy


class MesInputFormatter:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def convert_to_numbers(self, data):
        for elem in data:
            for s in elem:
                if len(s) > 8:
                    s = s[6:]


    def format(self, input_m_count, train_ratio):
        size = len(self.data)
        to_erase = size % input_m_count
        for index in [((size - 1) - i) for i in range(to_erase)]:
            self.data.pop(index)
            self.labels.pop(index)
        size -= to_erase

        all_data = numpy.array_split(numpy.array(self.data), int(size / input_m_count))
        all_labels = numpy.array_split(numpy.array(self.labels), int(size / input_m_count))

        all_dataf = []
        all_labelsf = []
        for elem in all_data:
            tmp = []
            for m in elem:
                tmp.extend(m)
            all_dataf.append(tmp)

        for elem in all_labels:
            tmp = []
            for m in elem:
                tmp.extend(m)
            all_labelsf.append(tmp)

        for elem in all_labelsf:
            for index in [(len(elem) - 1) - i for i in range(len(elem))]:
                if index > 1:
                    elem.pop(index)

        train_data_count = int(len(all_dataf) * train_ratio)
        train_data = all_dataf[:train_data_count]
        train_labels = all_labelsf[:train_data_count]
        test_data = all_dataf[train_data_count:]
        test_labels = all_labelsf[train_data_count:]

        return (train_data, train_labels), (test_data, test_labels)
