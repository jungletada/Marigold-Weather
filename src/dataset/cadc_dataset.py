import os
import torch
import numpy as np
from .base_depth_dataset import \
    BaseDepthDataset, DepthFileNameMode, DatasetMode


class CADCDataset(BaseDepthDataset):
    RAW_HEIGHT = 1024
    RAW_WIDTH = 1280
    IMGFOLDER = "left-image-full-size"
    DEPTHFOLDER = "depth-map-full-size"
    def __init__(
        self,
        **kwargs):
        super(CADCDataset, self).__init__(
            min_depth=1e-6,
            max_depth=70,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id_,
            **kwargs)

        self.gt_depth_path = os.path.join(self.dataset_dir, 'gt_depths.npy')
        self.gt_labels = np.load(self.gt_depth_path, allow_pickle=True).astype(np.float32)

    def _load_rgb_data(self, rgb_rel_path, dict_names=("rgb_int", "rgb_norm")):
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_data = {
            dict_names[0]: torch.from_numpy(rgb).int(),
            dict_names[1]: torch.from_numpy(rgb_norm).float(),
        }
        return rgb_data

    def benchmark_crop(self, rasters):
        top, bottom = 232, 776
        for key, value in rasters.items():
            rasters[key] = value[:, top:bottom, :]
        return rasters
        

    def _get_data_item(self, index):
        rgb_rel_path = self.filenames[index][0]
        rasters = {} # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))
        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            depth_data = {}
            depth_label = torch.from_numpy(self.gt_labels[index]).unsqueeze(dim=0)
            depth_data["depth_raw_linear"] = depth_label
            depth_data["depth_filled_linear"] = depth_label.clone()
            rasters.update(depth_data)
            # valid mask
            rasters["valid_mask_raw"] = self._get_valid_mask(
                rasters["depth_raw_linear"]
            ).clone()
            rasters["valid_mask_filled"] = self._get_valid_mask(
                rasters["depth_filled_linear"]
            ).clone()

        
        rasters = self.benchmark_crop(rasters)

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other

if __name__ == '__main__':
    ds = CADCDataset(
        filename_ls_path='data_split/cadc/test_files.txt',
        dataset_dir='data/cadcd', 
        mode=DatasetMode.EVAL,
        disp_name='CADCD',)
    
    print(len(ds))
    itemd = ds[152]
    print(itemd["index"])
    print(itemd["rgb_relative_path"])
    print(itemd["rgb_int"].shape)
    print(itemd["rgb_norm"].shape)
    print(itemd["depth_filled_linear"].shape)