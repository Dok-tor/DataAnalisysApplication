import re
import numpy as np


def get_re_matches(input_string: str) -> list:
    pattern = re.compile(r'"([^"]*)"|(-?\d+\.\d+)|(-?\d+)')

    # matches = pattern.findall(input_string)
    # result = []
    matches = pattern.findall(input_string)
    flattened_matches = [item for sublist in matches for item in sublist if item]
    return flattened_matches


class DataLoader:
    def __init__(self, path: str) -> None:
        self.count = None
        self.dim = None
        self.path = path
        self.data = None
        self.labels = None
        self.is_labels = False
        self.labels_dict = None

    def loadData(self) -> None:
        data = []
        self.labels = []

        with open(self.path, 'r') as f:
            line = f.readline().split()
            if len(line) == 3:
                self.count, self.dim, number_of_clusters = map(int, line)
                self.labels_dict = dict()
            else:
                self.count, self.dim = map(int, line)
                number_of_clusters = None


            for i in range(self.count):
                line = f.readline().strip()
                data_in_line = get_re_matches(line)
                data.append(list(map(float, data_in_line[:self.dim])))

                if number_of_clusters:
                    label = int(float(data_in_line[int(self.dim)]))
                    self.labels.append(label)
                    if len(data_in_line) == self.dim + 2:
                        self.labels_dict[label] = data_in_line[-1]

            self.data = np.array(data)

        if self.labels:
            self.is_labels = True
            self.labels = np.array(self.labels)
        else:
            self.labels = None

    def getData(self):
        return self.data

    def getLabels(self):
        return self.labels

    def getLabelsDict(self):
        return self.labels_dict

    def getDim(self):
        return self.dim

    def isLabels(self):
        return self.is_labels
