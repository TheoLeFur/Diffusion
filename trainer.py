import torch
import torch.nn
import torchvision
import torch.utils as utils
import os
from torchvision import transforms
from typing import Dict
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.networks import Unet
from diffusion.diffusion import DiffusionTrainer, DiffusionSampler


def train(model_config: Dict):

    device = "mps"

    dataset = torchvision.datasets.CIFAR10("data/cifar10", train=True, download=True, transform=torchvision.transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]))

    dataloader = utils.data.DataLoader(
        dataset, batch_size=model_config["batch_size"], shuffle=True)
    model = Unet(model_config["n_channels"],
                 model_config["t_length"], model_config["n_res_blocks"]).float().to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_config["learning_rate"])
    trainer = DiffusionTrainer(
        model, model_config["beta1"], model_config["betaT"], model_config["t_length"]).float().to(device)

    for epoch in range(model_config["n_epochs"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:

            for images, _ in tqdmDataLoader:

                optimizer.zero_grad()
                images = images.to(device)
                loss = trainer(images).mean()
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "loss": loss
                })

            torch.save(model.state_dict(), os.path.join(
                model_config["save_model_dir"] + str(epoch) + "_.pt"))


def evaluate(model_config: Dict):

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Unet(model_config["n_channels"],
                 model_config["t_length"], model_config["n_res_blocks"]).float().to(device)

    checkpoint = torch.load(os.path.join(
        model_config["save_model_dir"] + str(len(model_config["n_epochs"]) - 1)), map_location=device)
    model.load_state_dict(checkpoint)

    model.eval()

    sampler = DiffusionSampler(
        model, model_config["beta1"], model_config["betaT"], model_config["t_length"])
    noisy_img = torch.randn(
        [model_config["batch_size"], 3, 32, 32], device=device)
    torch.clamp(noisy_img * 0.5 + 0.5, 0, 1)
    sampled = sampler(noisy_img)
    plt.imshow(sampled)
    plt.show()
