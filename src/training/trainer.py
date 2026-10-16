"""Entrenamiento con checkpoints reanudables.

last.pt  -> estado completo (modelo, optimizador, scheduler, scaler, RNG, historial, época)
best.pt  -> mejor accuracy de VALIDACIÓN en R_train; si empata, menor val_loss
            (nunca se usa el conjunto de prueba)
Pausar = Ctrl+C. Reanudar = misma orden con --resume.
"""
import csv
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.utils.seed import get_rng_state, set_rng_state


class Trainer:
    def __init__(self, model, cfg, run_dir, device, train_loader, val_loader):
        self.model, self.cfg, self.device = model.to(device), cfg, device
        self.tr, self.va = train_loader, val_loader
        self.run_dir = Path(run_dir)
        self.ck_dir = self.run_dir / "checkpoints"
        self.met_dir = self.run_dir / "metrics"
        t = cfg.training
        self.t = t
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            (decay if p.ndim >= 2 and n not in ("pos_embed", "cls_token") else no_decay).append(p)
        self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": t.weight_decay},
                                      {"params": no_decay, "weight_decay": 0.0}], lr=t.lr)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_factor)
        self.use_amp = bool(t.amp) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.crit = nn.CrossEntropyLoss(label_smoothing=t.label_smoothing)
        self.history, self.best_acc, self.best_loss = [], -1.0, float("inf")
        self.best_epoch, self.bad, self.finished = -1, 0, False

    def _lr_factor(self, e):
        w, E = self.t.warmup_epochs, self.t.epochs
        if e < w:
            return (e + 1) / w
        return max(0.01, 0.5 * (1 + math.cos(math.pi * (e - w) / max(1, E - w))))

    def _state(self, epoch):
        return {"epoch": epoch, "model": self.model.state_dict(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
                "rng": get_rng_state(), "history": self.history, "best_acc": self.best_acc,
                "best_loss": self.best_loss, "best_epoch": self.best_epoch, "bad": self.bad,
                "finished": self.finished}

    def _restore(self, ck):
        self.model.load_state_dict(ck["model"]); self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"]); self.scaler.load_state_dict(ck["scaler"])
        self.history, self.best_acc = ck["history"], ck["best_acc"]
        self.best_loss = ck.get("best_loss", float("inf"))
        self.best_epoch, self.bad, self.finished = ck["best_epoch"], ck["bad"], ck["finished"]
        set_rng_state(ck["rng"])

    def _write_log(self):
        self.met_dir.mkdir(parents=True, exist_ok=True)
        with open(self.met_dir / "train_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])
            w.writeheader(); w.writerows(self.history)

    def _epoch(self, loader, train):
        self.model.train(train)
        tot_loss = correct = n = 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                    out = self.model(x)
                    loss = self.crit(out, y)
                if train:
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.t.grad_clip:
                        self.scaler.unscale_(self.opt)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.t.grad_clip)
                    self.scaler.step(self.opt); self.scaler.update()
                tot_loss += loss.item() * len(y)
                correct += (out.argmax(1) == y).sum().item(); n += len(y)
        return tot_loss / n, correct / n

    def fit(self, resume=False, overwrite=False):
        last = self.ck_dir / "last.pt"
        start = 0
        if last.exists():
            if resume:
                ck = load_checkpoint(last, self.device)
                self._restore(ck)
                start = ck["epoch"] + 1
                print(f"[resume] continuando desde la época {start + 1}/{self.t.epochs}")
            elif overwrite:
                shutil.rmtree(self.run_dir)
            else:
                raise FileExistsError(f"{last} ya existe. Usa --resume para continuar o --overwrite para empezar de cero.")
        if self.finished or start >= self.t.epochs:
            print("[fit] el entrenamiento ya estaba terminado."); return
        try:
            for epoch in range(start, self.t.epochs):
                t0 = time.time()
                tl, ta = self._epoch(self.tr, True)
                vl, va = self._epoch(self.va, False)
                lr = self.opt.param_groups[0]["lr"]
                self.sched.step()
                self.history.append({"epoch": epoch + 1, "lr": lr, "train_loss": tl, "train_acc": ta,
                                     "val_loss": vl, "val_acc": va, "seconds": time.time() - t0})
                print(f"ep {epoch+1:03d}/{self.t.epochs} lr={lr:.2e} train {tl:.4f}/{ta:.4f} val {vl:.4f}/{va:.4f} ({time.time()-t0:.1f}s)")
                if va > self.best_acc or (va == self.best_acc and vl < self.best_loss):
                    self.best_acc, self.best_loss, self.best_epoch, self.bad = va, vl, epoch + 1, 0
                    save_checkpoint({"model": self.model.state_dict(), "epoch": epoch + 1, "val_acc": va},
                                    self.ck_dir / "best.pt")
                else:
                    self.bad += 1
                stop = self.t.patience and self.bad >= self.t.patience
                self.finished = bool(stop) or epoch + 1 == self.t.epochs
                if self.finished or (epoch + 1) % self.t.save_every == 0:
                    save_checkpoint(self._state(epoch), last)
                self._write_log()
                if stop:
                    print(f"[early stopping] sin mejora en {self.t.patience} épocas"); break
        except KeyboardInterrupt:
            print("\n[pausa] Interrumpido. Reanuda con --resume (se retoma desde el último last.pt).")
            raise SystemExit(130)
        print(f"[fit] mejor val_acc={self.best_acc:.4f} (val_loss={self.best_loss:.4f}) en la época {self.best_epoch}")