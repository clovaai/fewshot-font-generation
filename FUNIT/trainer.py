"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from pathlib import Path

import torch
import torch.nn as nn
import base.utils as utils
from base.trainer import cyclize


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


class FUNITTrainer(nn.Module):
    def __init__(self, model, g_opt, d_opt, writer, logger, cfg, use_ddp):
        super(FUNITTrainer, self).__init__()
        self.model = model
        self.gen_opt = g_opt
        self.dis_opt = d_opt
        self.writer = writer
        self.logger = logger
        self.cfg = cfg
        self.use_ddp = use_ddp
        self.step = 0
        self.checkpoint_dir = self.cfg.work_dir / "checkpoints"
        if cfg.resume is not None:
            self.resume(cfg.resume)

    def gen_update(self, co_data, cl_data, tg_data):
        self.gen_opt.zero_grad()
        this_model = self.model.module if self.use_ddp else self.model
        al, ad, xr, cr, sr, ac = this_model.gen_update(co_data, cl_data, tg_data, self.cfg)
        self.loss_gen_total = torch.mean(al)
        self.loss_gen_recon_x = torch.mean(xr)
        self.loss_gen_recon_c = torch.mean(cr)
        self.loss_gen_recon_s = torch.mean(sr)
        self.loss_gen_adv = torch.mean(ad)
        self.accuracy_gen_adv = torch.mean(ac)
        self.gen_opt.step()
        update_average(this_model.gen_test, this_model.gen)
        return self.accuracy_gen_adv.item()

    def dis_update(self, co_data, cl_data, tg_data):
        self.dis_opt.zero_grad()
        this_model = self.model.module if self.use_ddp else self.model
        al, lfa, lre, reg, acc = this_model.dis_update(co_data, cl_data, tg_data, self.cfg)
        self.loss_dis_total = torch.mean(al)
        self.loss_dis_fake_adv = torch.mean(lfa)
        self.loss_dis_real_adv = torch.mean(lre)
        self.loss_dis_reg = torch.mean(reg)
        self.accuracy_dis_adv = torch.mean(acc)
        self.dis_opt.step()
        return self.accuracy_dis_adv.item()

    def test(self, co_data, cl_data):
        this_model = self.model.module if self.use_ddp else self.model
        return this_model.test(co_data, cl_data)

    def resume(self, checkpoint_path):
        this_model = self.model.module if self.use_ddp else self.model

        weights = torch.load(checkpoint_path)
        this_model.gen.load_state_dict(weights['generator'])
        this_model.gen_test.load_state_dict(weights['generator_ema'])
        self.step = weights["step"]

        this_model.dis.load_state_dict(weights['discriminator'])

        self.dis_opt.load_state_dict(weights['d_optimizer'])
        self.gen_opt.load_state_dict(weights['optimizer'])

        print('Resume from iteration %d' % self.step)

    def save(self, snapshot_dir):
        this_model = self.model.module if self.use_ddp else self.model
        # Save generators, discriminators, and optimizers
        save_dic = {
            "generator": this_model.gen.state_dict(),
            "generator_ema": this_model.gen_test.state_dict(),
            "discriminator": this_model.dis.state_dict(),
            "optimizer": self.gen_opt.state_dict(),
            "d_optimizer": self.dis_opt.state_dict(),
            "step": self.step

        }
        save_name = Path(snapshot_dir) / f"{self.step:06d}.pth"

        torch.save(save_dic, save_name)

    def load_gen(self, ckpt_name):
        state_dict = torch.load(ckpt_name)
        self.model.gen.load_state_dict(state_dict['generator'])
        self.model.gen_test.load_state_dict(state_dict['generator_ema'])

    def translate(self, co_data, cl_data):
        return self.model.translate(co_data, cl_data)

    def translate_k_shot(self, co_data, cl_data, k, mode):
        return self.model.translate_k_shot(co_data, cl_data, k, mode)

    def train(self, loader, val_loaders, max_iter):
        for data in cyclize(loader):
            epoch = self.step // len(loader)
            if self.use_ddp and (self.step % len(loader)) == 0:
                loader.sampler.set_epoch(epoch)
                
            keys = ('trg_imgs', 'src_imgs', 'style_imgs', 'trg_cids',
                    'trg_fids')
            target_img, content_img, style_stack, target_uidx, target_sidx = \
                [data[key].cuda() for key in keys]

            co_data = [content_img]
            cl_data = [style_stack]
            tg_data = [target_img, target_sidx, target_uidx]

            d_acc = self.dis_update(co_data, cl_data, tg_data)
            g_acc = self.gen_update(co_data, cl_data, tg_data)
            torch.cuda.synchronize()

            if self.cfg.rank == 0:
                if self.step % self.cfg['print_freq'] == 0:
                    self.logger.info('D acc: %.4f\t G acc: %.4f' % (d_acc, g_acc))
                    self.logger.info("Iteration: %08d/%08d" % (self.step, max_iter))

                if (self.step % self.cfg['save_freq'] == 0 or self.step % self.cfg['val_freq'] == 0):
                    with torch.no_grad():
                        for tag, val_loader in val_loaders.items():
                            outputs = []
                            trg_imgs = []
                            for t, val_data in enumerate(val_loader):
                                val_co_data = val_data["src_imgs"].cuda()
                                val_cl_data = val_data["style_imgs"].cuda()
                                trg = val_data["trg_imgs"]

                                out = self.test(val_co_data, val_cl_data)
                                outputs.append(out)
                                trg_imgs.append(trg)

                            outputs = torch.cat(outputs)
                            trg_imgs = torch.cat(trg_imgs)

                            nrow = val_loader.dataset.n_gen
                            grid = utils.make_comparable_grid(trg_imgs,
                                                              outputs,
                                                              nrow=nrow)

                            self.writer.add_image(tag, grid, global_step=self.step)

                if self.step % self.cfg['save_freq'] == 0:
                    self.save(self.checkpoint_dir)
                    self.logger.info('Saved model at iteration %d' % (self.step))

            if self.step >= max_iter:
                break

            self.step += 1
