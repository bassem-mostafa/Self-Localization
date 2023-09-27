import cv2
import numpy

from torch.utils.data import Dataset
from pathlib          import Path

class HPatchesSquenceDataset(Dataset):
    def __init__(
            self,
            root = "../dataset(s)/hpatches/hpatches-sequences-release",
            ):
        root_path = Path(root)
        assert root_path.exists(), f"Dataset path `{str(root_path)}` does NOT exist!"
        self.items = []
        folders = [elem for elem in root_path.iterdir() if elem.is_dir()]
        for folder in folders:
            img_names = list(folder.glob('*.png')) + \
                        list(folder.glob('*.jpg')) + \
                        list(folder.glob('*.ppm'))
            self.items += img_names

        self.len = len(self.items)
        assert self.len > 0, f"Dataset path `{str(root_path)}` does NOT have supported formats"

    def __getitem__(self, index):
        image_path = Path(self.items[index])
        assert image_path.exists(), f"Image path `{str(image_path)}` does NOT exist!"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        assert image is not None, f"Couldn't load `{str(image_path)}`"

        # Convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32') / 255  # HxWxC

        # Load homographies
        homography = numpy.loadtxt(image_path.parent / ("H_1_" + image_path.stem)).astype('float32') if image_path.stem != "1" else None
        homography_inv = numpy.linalg.inv(homography) if homography is not None else None

        return {
            "image": image,  # HxWxC
            "path": str(image_path),
            "homography": homography,
            "homography_inv": homography_inv,
            }

    def __len__(self):
        return self.len

if __name__ == '__main__':
    print(f"Data-set Demo")
    from tqdm import tqdm
    from matplotlib import pyplot

    dataset = HPatchesSquenceDataset()
    # from torch.utils.data import DataLoader
    # loader = DataLoader(dataset, batch_size=32, shuffle=True) # FIXME dataloader expects dataset to be equal sized, which is not our case here
    pyplot.ion()
    pyplot.figure(1) # Create a figure with id 1
    for item in tqdm(dataset):
        if not pyplot.fignum_exists(1): # If figure has been closed, break
            print(f"Window has been closed!")
            break
        pyplot.figure(1) # re-select figure 1
        pyplot.imshow(item['image'])
        pyplot.show(block=False)
        pyplot.pause(0.001)