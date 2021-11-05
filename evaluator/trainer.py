"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import os

import torch
from torch import nn
from torch.optim import lr_scheduler

from .cutmix import cutmix


class EvalTrainer(object):
    def __init__(self, model_style, model_content, opt_style, opt_content,
                 logger, cfg, use_ddp=False):
        super().__init__()  # inherit nn.Module for easy cuda parallelizing

        self.s_model = model_style
        self.s_opt = opt_style
        self.s_scheduler = lr_scheduler.CosineAnnealingLR(self.s_opt,
                                                          T_max=cfg.max_epoch,
                                                          last_epoch=-1)

        self.c_model = model_content
        self.c_opt = opt_content
        self.c_scheduler = lr_scheduler.CosineAnnealingLR(self.c_opt,
                                                          T_max=cfg.max_epoch,
                                                          last_epoch=-1)

        self.logger = logger
        self.cfg = cfg
        self.use_ddp = use_ddp

        self.epoch = 0
        self.step = 0
        self.checkpoint_dir = self.cfg.work_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # LOGGING Information
        self.loss_dict = {}  # put losses to log

        if self.cfg.resume is not None:
            self.resume(self.cfg.resume)

    def save(self):
        model_name = f'net_{self.epoch}_{self.step}.pth'
        trainer_state_dict = {
            "epoch": self.epoch,
            "step": self.step,
            "opt_style": self.s_opt.state_dict(),
            "opt_content": self.c_opt.state_dict(),
            "model_style": self.s_model.state_dict(),
            "model_content": self.c_model.state_dict()
        }

        model_path = os.path.join(self.checkpoint_dir, model_name)
        torch.save(trainer_state_dict, model_path)

    def resume(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]

        self.s_model.load_state_dict(checkpoint["model_style"])
        self.s_opt.load_state_dict(checkpoint["opt_style"])
        self.s_scheduler = lr_scheduler.CosineAnnealingLR(self.s_opt,
                                                          self.cfg.max_epoch,
                                                          last_epoch=self.epoch)

        self.c_model.load_state_dict(checkpoint["model_content"])
        self.c_opt.load_state_dict(checkpoint["opt_content"])
        self.c_scheduler = lr_scheduler.CosineAnnealingLR(self.c_opt,
                                                          self.cfg.max_epoch,
                                                          last_epoch=self.epoch)

    def cutmix_step(self, img, sidx, cidx, crit):
        img, lam, sidx_, cidx_ = cutmix(img, sidx, cidx, self.cfg.beta)

        self.s_model.zero_grad()
        self.s_opt.zero_grad()
        s_logit = self.s_model(img)
        s_loss = crit(s_logit, sidx)*lam + crit(s_logit, sidx_)*(1.-lam)

        s_loss.backward()
        self.s_opt.step()

        self.c_model.zero_grad()
        self.c_opt.zero_grad()
        c_logit = self.c_model(img)
        c_loss = crit(c_logit, cidx)*lam + crit(c_logit, cidx_)*(1.-lam)

        c_loss.backward()
        self.c_opt.step()

        return s_loss, c_loss

    def plain_step(self, img, sidx, cidx, crit):
        self.s_model.zero_grad()
        self.s_opt.zero_grad()
        s_logit = self.s_model(img)
        s_loss = crit(s_logit, sidx)

        s_loss.backward()
        self.s_opt.step()

        self.c_model.zero_grad()
        self.c_opt.zero_grad()
        c_logit = self.c_model(img)
        c_loss = crit(c_logit, cidx)

        c_loss.backward()
        self.c_opt.step()

        return s_loss, c_loss

    def train(self, loader, val_loader):
        self.s_model.train()
        self.c_model.train()
        crit = nn.CrossEntropyLoss()
        start_epoch = self.epoch

        for epoch in range(start_epoch, self.cfg.max_epoch):
            self.epoch = epoch
            for dp in loader:
                img = dp["imgs"].cuda()
                sidx = dp["fids"].cuda()
                cidx = dp["cids"].cuda()

                r = torch.rand(1)
                if r < self.cfg.cutmix_prob:
                    s_loss, c_loss = self.cutmix_step(img, sidx, cidx, crit)

                else:
                    s_loss, c_loss = self.plain_step(img, sidx, cidx, crit)

                self.loss_dict['loss_style'] = s_loss
                self.loss_dict['loss_content'] = c_loss

                self.step += 1

                if not self.step % self.cfg.log_iter:
                    if r < self.cfg.cutmix_prob:
                        self.logger.info(
                            f'(Cutmix)\tStyle loss: {self.loss_dict["loss_style"]} Content loss: {self.loss_dict["loss_content"]}'
                        )
                    else:
                        self.logger.info(
                            f'(Plain)\tStyle loss: {self.loss_dict["loss_style"]} Content loss: {self.loss_dict["loss_content"]}'
                        )

            self.s_scheduler.step()
            self.c_scheduler.step()

            if self.cfg.rank == 0:
                s_acc, c_acc = self.validate(val_loader)
                self.logger.info(
                    '####################################################################\n' +
                    f'[Epoch {self.epoch}] Style Acc: {s_acc:.2f} Content Acc: {c_acc:.2f}' +
                    '####################################################################\n'
                )
                if not self.epoch % self.cfg.save_epoch:
                    self.save()

    def validate(self, loader):
        self.s_model.eval()
        self.c_model.eval()

        correct_s = 0
        correct_c = 0
        total_num = 0

        for dp in loader:
            img = dp["imgs"].cuda()
            sidx = dp["fids"]
            cidx = dp["cids"]

            logit_s = self.s_model(img)
            pred_s = logit_s.max(dim=1)[1].detach().cpu()
            correct_s += torch.sum(pred_s == sidx).item()

            logit_c = self.c_model(img)
            pred_c = logit_c.max(dim=1)[1].detach().cpu()
            correct_c += torch.sum(pred_c == cidx).item()

            total_num += img.size(0)

        s_acc = correct_s / total_num * 100
        c_acc = correct_c / total_num * 100

        self.s_model.train()
        self.c_model.train()

        return s_acc, c_acc
