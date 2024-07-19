import numpy as np
from numba import njit, prange

class DataLoader:
    def __init__(self, path: str) -> None:
        self.count = None
        self.dim = None
        self.path = path
        self.data = None
        self.labels = None
        self.is_labels = False

    def loadData(self) -> None:
        data = []
        self.labels = []

        with open(self.path, 'r') as f:
            self.count, self.dim = map(int, f.readline().split())
            for i in range(self.count):
                line = f.readline().split()
                data.append(list(map(float, line[:self.dim])))
                if len(line) == self.dim + 1:
                    label = int(float(line[int(self.dim)]))
                    self.labels.append(label)

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

    def getDim(self):
        return self.dim

    def isLabels(self):
        return self.is_labels
