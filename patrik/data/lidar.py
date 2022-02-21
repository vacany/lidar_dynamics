import torch

from data.datasets.semantic_kitti import SemKittiDataset

class Lidar_Dataset(torch.utils.data.Dataset):
    def __init__(self, NAME='Semantic_Kitti', seq=0, frame=0):
        super().__init__()

        self.name = NAME
        self.sequence = seq
        self.frame = frame

        self._get_base_dataset()


    def _get_base_dataset(self):
        self.base_dataset = SemKittiDataset(sequence=self.sequence, frame=self.frame)

        print(f"Chosen Dataset: {self.name}")

    def __getitem__(self, item):
        data = self.base_dataset.get_frame()

        #TODO

        return data

    def __len__(self):
        # TODO
        return 10
