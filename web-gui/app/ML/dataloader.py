import pandas as pd
import torch
import os


class HypoxiaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, info):
        self.data_dir = data_dir
        self.info = info

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        folders, has_hypoxia = self.info[index]
        folders = str(folders).split(",")
        data_folder = "hypoxia" if has_hypoxia else "regular"
        bpm = []
        uterus = []
        bpm_ts = []
        uterus_ts = []
        for f in folders:
            data_loc = os.path.join(self.data_dir, data_folder, f)
            for _file in sorted(os.listdir(os.path.join(data_loc, "bpm"))):
                _file_uterus = _file.replace("1.csv", "2.csv").replace("3.csv", "4.csv")
                if os.path.isfile(os.path.join(data_loc, "uterus", _file_uterus)):
                    bpm_data = pd.read_csv(os.path.join(data_loc, "bpm", _file))
                    uterus_data = pd.read_csv(os.path.join(data_loc, "uterus", _file_uterus))
                    bpm.append(bpm_data["value"].to_list())
                    uterus.append(uterus_data["value"].to_list())
                    bpm_ts.append(bpm_data["time_sec"].to_list())
                    uterus_ts.append(uterus_data["time_sec"].to_list())
            #for _file in sorted(os.listdir(os.path.join(data_loc, "uterus"))):
                #uterus.append(pd.read_csv(os.path.join(data_loc, "uterus", _file)))
        #print(folders)
        assert len(bpm) == len(uterus)
        data = dict()
        data["bpm"] = []
        data["uterus"] = []
        data["bpm_ts"] = []
        data["uterus_ts"] = []
        for x in bpm:
            data["bpm"].extend(x)
        for x in uterus:
            data["uterus"].extend(x)
        for x in bpm_ts:
            data["bpm_ts"].extend(x)
        for x in uterus_ts:
            data["uterus_ts"].extend(x)
        #print(len(bpm), folders)
        #i = torch.randint(low=0, high=len(bpm), size=(1,))
        #return {"bpm": torch.tensor(data["bpm"][i], dtype=torch.float32).unsqueeze(0), "uterus": torch.tensor(data["uterus"][i], dtype=torch.float32).unsqueeze(0)}, torch.tensor([1.0 - float(has_hypoxia), float(has_hypoxia)], dtype=torch.float32)
        return {
                "bpm": torch.tensor(data["bpm"], dtype=torch.float32),
                "uterus": torch.tensor(data["uterus"], dtype=torch.float32),
                "bpm_ts": torch.tensor(data["bpm_ts"], dtype=torch.float32),
                "uterus_ts": torch.tensor(data["uterus_ts"], dtype=torch.float32),
                "folders": ",".join(folders)
                }, torch.tensor([1.0 - float(has_hypoxia), float(has_hypoxia)], dtype=torch.float32)