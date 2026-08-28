"""Pretrain absorbing D3PM priors on FastMRI (fine image or coarse row profile)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from d3pm_runner import D3PM
from itw.configs import default_device, default_fastmri_root
from itw.data.fastmri import FastMRIDataset
from itw.discrete import kspace_to_row_disc, magnitude_to_fine_disc
from itw.row_d3pm import build_coarse_backbone, build_fine_backbone


class FastMRIDiscDataset(Dataset):
    """Yields discrete targets for D3PM pretraining: (x_disc, cond)."""

    def __init__(
        self,
        h5_root: str,
        mode: str = "fine",
        scout_size: int = 32,
        target_dim: int = 300,
        fine_size: int = 96,
        n_bins: int = 8,
    ):
        if mode not in {"fine", "coarse"}:
            raise ValueError(f"mode must be 'fine' or 'coarse', got {mode}")
        self.mode = mode
        self.fine_size = fine_size
        self.n_bins = n_bins
        self.base = FastMRIDataset(
            h5_root, scout_size=scout_size, target_dim=target_dim
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        _z, c, kspace = self.base[idx]
        if self.mode == "fine":
            x = magnitude_to_fine_disc(
                c.unsqueeze(0), size=self.fine_size, n_bins=self.n_bins
            )[0]
        else:
            x = kspace_to_row_disc(kspace.unsqueeze(0), n_bins=self.n_bins)[0]
        cond = torch.tensor(0, dtype=torch.long)
        return x, cond


def train_fastmri_d3pm(
    mode: str = "fine",
    data_root: str | None = None,
    save_dir: str | None = None,
    n_bins: int = 8,
    fine_size: int = 96,
    n_t: int = 1000,
    batch_size: int = 8,
    n_epochs: int = 50,
    lr: float = 1e-3,
    num_workers: int = 4,
    save_every: int = 5,
    device: str | None = None,
    coarse_hidden: int = 64,
) -> D3PM:
    data_root = data_root or default_fastmri_root()
    device = device or default_device()
    if save_dir is None:
        save_dir = (
            "models_d3pm_fastmri_fine"
            if mode == "fine"
            else "models_d3pm_fastmri_coarse_kspace"
        )
    os.makedirs(save_dir, exist_ok=True)

    if mode == "fine":
        backbone = build_fine_backbone(n_bins=n_bins)
    else:
        backbone = build_coarse_backbone(n_bins=n_bins, hidden=coarse_hidden)

    d3pm = D3PM(
        backbone,
        n_t,
        num_classes=n_bins,
        hybrid_loss_coeff=0.0,
        forward_type="absorb",
        schedule="cosine",
    ).to(device)

    dataset = FastMRIDiscDataset(
        data_root,
        mode=mode,
        fine_size=fine_size,
        n_bins=n_bins,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=lr)
    log_path = Path(save_dir) / "train.log"

    print(
        f"mode={mode} params={sum(p.numel() for p in d3pm.x0_model.parameters())} "
        f"batches/epoch={len(dataloader)}"
    )

    for epoch in range(n_epochs):
        d3pm.train()
        loss_ema = None
        pbar = tqdm(dataloader, desc=f"{mode} epoch {epoch}")
        for x, cond in pbar:
            x = x.to(device)
            cond = cond.to(device)
            optim.zero_grad()
            loss, info = d3pm(x, cond)
            loss.backward()
            optim.step()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()
            pbar.set_description(
                f"{mode} epoch {epoch} loss {loss_ema:.4f} ce {info['ce_loss']:.4f}"
            )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"epoch: {epoch}, loss: {loss_ema:.6f}, ce: {info['ce_loss']:.6f}\n"
            )

        if (epoch + 1) % save_every == 0:
            ckpt = Path(save_dir) / f"model_absorb_cosine_{epoch}.pth"
            torch.save(d3pm.state_dict(), ckpt)

    final = Path(save_dir) / "model_absorb_cosine_final.pth"
    torch.save(d3pm.state_dict(), final)
    print(f"saved {final}")
    return d3pm


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain FastMRI D3PM priors")
    parser.add_argument(
        "--mode",
        choices=("fine", "coarse", "both"),
        default="both",
        help="Which prior to train",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--fine-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--coarse-hidden", type=int, default=64)
    parser.add_argument("--fine-save-dir", type=str, default="models_d3pm_fastmri_fine")
    parser.add_argument("--coarse-save-dir", type=str, default="models_d3pm_fastmri_coarse_kspace")
    args = parser.parse_args()

    modes = ("fine", "coarse") if args.mode == "both" else (args.mode,)
    for mode in modes:
        train_fastmri_d3pm(
            mode=mode,
            data_root=args.data_root,
            save_dir=args.fine_save_dir if mode == "fine" else args.coarse_save_dir,
            n_bins=args.n_bins,
            fine_size=args.fine_size,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            save_every=args.save_every,
            device=args.device,
            coarse_hidden=args.coarse_hidden,
        )


if __name__ == "__main__":
    main()
