import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

from tqdm import tqdm


sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from VoxGENEO import Voxelization as Vox


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class CnnBaseline(nn.Module):

    def __init__(self, conv_num, kernel_size, plot=True) -> None:
        super().__init__()

        self.conv_layer = nn.Conv3d(in_channels=1, out_channels=conv_num, kernel_size=kernel_size, padding='same', device=device, dtype=torch.double)
        self.conv_layer2 = nn.Conv3d(in_channels=conv_num, out_channels=conv_num, kernel_size=kernel_size, padding='same', device=device, dtype=torch.double)
        self.conv_num = conv_num

        if plot:
            print(f"Total Number of train. params = {self.get_num_total_params()}")

    
    def get_conv_nums(self):
        return self.conv_num

    def get_cvx_coefficients(self):
        return {}

    def get_dict_parameters(self):
        # For consistency
        return self.get_cvx_coefficients()

    def get_geneo_params(self):
        # For consistency
        return {}

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x:torch.Tensor):

        conv = self.conv_layer2(self.conv_layer(x))

        conv_pred = torch.zeros_like(x)

        for i in range(self.conv_num):
            conv_pred += conv[:, [i]]

        conv_pred = torch.relu(torch.tanh(conv_pred))

        return conv_pred


class CnnBaseline2(nn.Module):

    def __init__(self, conv_num, kernel_size, plot=True) -> None:
        super().__init__()

        self.conv_layer = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3,2,2), padding='same', device=device, dtype=torch.double)

        self.conv_num = conv_num

        if plot:
            print(f"Total Number of train. params = {self.get_num_total_params()}")

    
    def get_conv_nums(self):
        return self.conv_num

    def get_cvx_coefficients(self):
        return {}

    def get_dict_parameters(self):
        # For consistency
        return self.get_cvx_coefficients()

    def get_geneo_params(self):
        # For consistency
        return {}

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x:torch.Tensor):

        conv = self.conv_layer(x)

        # print(x.shape)
        # print(conv.shape)

        conv_pred = torch.zeros_like(x)

        for i in range(self.conv_num):
            conv_pred += conv[:, [i]]

        conv_pred = torch.relu(torch.tanh(conv_pred))

        #Vox.plot_voxelgrid(conv_pred[0][0].cpu().detach().numpy())

        return conv_pred



if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    from datasets.ts40k import ToFullDense, torch_TS40K, ToTensor
    from torchvision.transforms import Compose
    from torch.utils.data import DataLoader
    from utils.observer_utils import init_metrics, visualize_batch, process_batch
    from scenenet_pipeline.torch_geneo.criterions.geneo_loss import GENEO_Loss
    from scenenet_pipeline.torch_geneo.criterions.w_mse import HIST_PATH
    from torch.profiler import profile, record_function, ProfilerActivity


    ROOT_PROJECT = str(Path(os.getcwd()).parent.absolute().parent.absolute())
    print(ROOT_PROJECT)
    SAVE_DIR = ROOT_PROJECT + "/dataset/torch_dataset"


    dummy = CnnBaseline(3, (9, 9, 9)).to(device)

    composed = Compose([ToTensor(), ToFullDense()])
    ts40k = torch_TS40K(dataset_path=SAVE_DIR, split='test', transform=composed)

    ts40k_loader = DataLoader(ts40k, batch_size=1, shuffle=True, num_workers=4)
    geneo_loss = GENEO_Loss(torch.tensor([]), hist_path=HIST_PATH, alpha=1, rho=3, epsilon=0.1)
    tau=0.65
    test_metrics = init_metrics(tau) 
    test_loss = 0

    
    
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            with record_function("model_inference"):
                vox, gt = ts40k[0]
                dummy(vox[None])

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

        input("\n\nContinue?\n\n")

        # --- Test Loop ---
        for batch in tqdm(ts40k_loader, desc=f"testing..."):
            loss, pred = process_batch(dummy, batch, geneo_loss, None, test_metrics, requires_grad=False)
            test_loss += loss

            test_res = test_metrics.compute()
            pre = test_res['Precision']
            rec = test_res['Recall']

            # print(f"Precision = {pre}")
            # print(f"Recall = {rec}")
            #if pre <= 0.1 or rec <= 0.1:
            if pre >= 0.3 and rec >= 0.20:
            #if True:
                print(f"Precision = {pre}")
                print(f"Recall = {rec}")
                vox, gt = batch
                visualize_batch(vox, gt, pred, tau=tau)
                input("\n\nPress Enter to continue...")

            test_metrics.reset()

        test_loss = test_loss /  len(ts40k_loader)
        test_res = test_metrics.compute()
        print(f"\ttest_loss = {test_loss:.3f};")
        for met in test_res:
            print(f"\t{met} = {test_res[met]:.3f};")




        