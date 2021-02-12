from torchmeta.utils.data import Task

class TensorTask(Task):
    def __init__(self, inputs, labels, index, num_classes, 
                 transform=None, target_transform=None):
        super(TensorTask, self).__init__(index, num_classes, transform=transform, target_transform=target_transform)
        self.inputs, self.labels = inputs, labels

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)
