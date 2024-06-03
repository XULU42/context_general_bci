# %%

from typing import List
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from scipy.interpolate import interp1d
import scipy.signal as signal
from einops import rearrange, reduce

import logging

logger = logging.getLogger(__name__)
try:
    import h5py
except:
    logger.info(
        "h5py not installed, please install with `conda install -c anaconda h5py`"
    )

from context_general_bci.config import DataKey, DatasetConfig
from context_general_bci.subjects import (
    SubjectInfo,
    SubjectArrayRegistry,
    create_spike_payload,
)
from context_general_bci.tasks import (
    ExperimentalTask,
    ExperimentalTaskLoader,
    ExperimentalTaskRegistry,
)


@ExperimentalTaskRegistry.register
class NeucyberPercerptionCOLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.neucyber_perception_co
    r"""
    Neucyber CIBR perception center out task loader
    """

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        task: ExperimentalTask,
    ):
        with h5py.File(datapath, "r") as h5file:
            # Todo
            # 1. source is float, NDT2 needs unit8 input
            # 2. data validation: after round, ntk data min is zero, max is 236 which do not make sense for 50 ms bin size.
            #    this is relate to padding value setting
            full_spikes = np.round(h5file["ntk"][:]).astype(np.uint8)

        full_spikes = torch.tensor(full_spikes)
        print(full_spikes.shape, full_spikes.max())
        meta_payload = {}
        meta_payload["path"] = []

        for t in range(1):
            single_payload = {
                DataKey.spikes: create_spike_payload(full_spikes[t], context_arrays),
                # DataKey.bhvr_vel: bhvr_vars[DataKey.bhvr_vel][t].clone(),  # for unupervised pretraining, bhvr data is not needed, ignore this currently
            }
            single_path = cache_root / f"{t}.pth"
            meta_payload["path"].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)


if __name__ == "__main__":
    preprocessed_path = Path(
        "/home/yuezhifeng_lab/aochuan/DATA/workspace/electrophysiological_data_process/code/ndt2/context_general_bci/data/preprocessed/neucyber_data"
    )
    context_name = "Bohr-main"
    NeucyberPercerptionCOLoader.load(
        "/home/yuezhifeng_lab/aochuan/DATA/workspace/electrophysiological_data_process/code/ndt2/context_general_bci/data/neucyber_data/perception/bohr/bohr_240402_01.mat",
        None,
        preprocessed_path,
        None,
        [context_name],
        None,
        None,
    )
    # currently only test for saved trial data structure and tensor shape
    trial_data = torch.load(preprocessed_path / "0.pth")
    assert (
        DataKey.spikes in trial_data and context_name in trial_data[DataKey.spikes]
    ), "Trial data structure not valid!"
    trial_data_shape = trial_data[DataKey.spikes][context_name].shape
    assert (
        len(trial_data_shape) == 3 and trial_data_shape[-1] == 1
    ), "Trial data shape not valid. It should have 3 dims(time bins, number of neurons, 1)"
    print("Saved trial data structure and shape is valid!")
