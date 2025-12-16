import os
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import *
from model import EncodeProcessDecode

def masked_mse(pred, y, to_interpolate_mask):
    mask = to_interpolate_mask.bool().squeeze(-1)  # Shape [num_points]
    
    pred_masked = pred[mask]  # Shape [num_masked_points, 4]
    y_masked = y[mask]        # Shape [num_masked_points, 4]
    
    mse = torch.mean((pred_masked - y_masked) ** 2)
    
    return mse

def train(args, model, train_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )

    loss_list = []
    runtime_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch in train_dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out, to_interpolate_mask = model(batch)

            loss = masked_mse(out, batch.y, to_interpolate_mask)
            # print(f"loss: {loss}")
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_runtime = time.time() - start_time

        avg_loss = epoch_loss / len(train_dataloader)
        loss_list.append(avg_loss)
        runtime_list.append(epoch_runtime)

        scheduler.step()

        if (epoch) % args.save_freq == 0:
            model_save_path = os.path.join(args.exp_name, f"model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch} to {model_save_path}")
    
        print(f"Epoch {epoch}/{args.num_epochs} - Loss: {avg_loss:.9f}, Runtime: {epoch_runtime:.2f}s")

    results = {"loss": loss_list, "runtime": runtime_list}
    txt_path = f"{args.exp_name}/results.txt"
    pkl_path = f"{args.exp_name}/results.pkl"

    with open(txt_path, "w") as f:
        for epoch, (loss, runtime) in enumerate(zip(loss_list, runtime_list)):
            f.write(f"Epoch {epoch+1}: Loss = {loss:.9f}, Runtime = {runtime:.2f}s\n")

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Training completed. Results saved to {txt_path} and {pkl_path}")

def plot_sst(lat, lon, temperature, count, ground_truth):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    contour = ax.contourf(lon_grid, lat_grid, temperature,
                        levels=np.arange(0, 32, 2),
                        cmap='nipy_spectral', extend='both')

    ax.coastlines()
    ax.gridlines(draw_labels=True)

    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05)
    cbar.set_label('degC')


    if ground_truth:
        plt.savefig(f"fig/ground_truth/sea_surface_temperature{count}.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"fig/pred/sea_surface_temperature{count}.png", dpi=300, bbox_inches='tight')



def eval(args, model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    scaler = MeanVarianceScaler(args.stats_path)

    total_loss = 0.0
    for batch in test_dataloader:
        batch = batch.to(device)

        out, to_interpolate_mask = model(batch)

        loss = masked_mse(out, batch.y, to_interpolate_mask)

        total_loss += loss.item()


    avg_loss = total_loss / len(test_dataloader)
    print(f"avg_loss: {avg_loss}")

def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_save_path", type=str)
    parser.add_argument("--split_path", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--stats_path", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--to_train", action="store_true")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_size", type=int)
    parser.add_argument("--latent_size", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--model_residual_connection", action="store_true")
    parser.add_argument("--dir_info", type=str)
    parser.add_argument("--sigma", type=float, required=False)
    parser.add_argument("--dir_message", type=str)
    parser.add_argument("--message_passing_steps", type=int)
    parser.add_argument("--latent_residual_connection", action="store_true")
    parser.add_argument("--layer_node_residual_connection", action="store_true")
    parser.add_argument("--layer_edge_residual_connection", action="store_true")
    parser.add_argument("--mask_latent_size", type=int)
    parser.add_argument("--node_input_size", type=int)
    parser.add_argument("--edge_input_size", type=int)
    parser.add_argument("--use_random", action="store_true")
    parser.add_argument("--ratio", type=float, required=False)
    parser.add_argument("--use_noise", action="store_true")
    parser.add_argument("--sigma2", type=float)
    parser.add_argument("--scale", action="store_true")


    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
        print(f"Folder '{args.exp_name}' created.")
    else:
        print(f"Folder '{args.exp_name}' already exists.")

    model = EncodeProcessDecode(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.to_train:
        if args.data_name == "sst":
            train_dataloader, _, _ = load_sst(args, "train")
        else:
            train_dataloader = load_train(args)
        train(args, model, train_dataloader)
    else:
        if args.data_name == "sst":
            test_dataloader, lat, lon = load_sst(args, "test")
        else:
            test_dataloader = load_test(args)
        with torch.no_grad():
            eval(args, model, test_dataloader)


if __name__ == "__main__":
    main()
