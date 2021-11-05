"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
import torch.nn.functional as F

from base.trainer import BaseTrainer, cyclize
import base.utils as utils


def to_batch(batch):
    in_batch = {
        "ref_imgs": batch["ref_imgs"].cuda(),
        "ref_fids": batch["ref_fids"].cuda(),
        "ref_decs": batch["ref_decs"].cuda(),
        "trg_fids": batch["trg_fids"].cuda(),
        "trg_decs": batch["trg_decs"].cuda(),
    }
    return in_batch


class DMTrainer(BaseTrainer):
    def __init__(self, gen, disc, g_optim, d_optim, aux_clf, ac_optim,
                 writer, logger, cfg, use_ddp):
        super().__init__(gen, disc, g_optim, d_optim, aux_clf, ac_optim,
                         writer, logger, cfg, use_ddp)

        self.to_batch = to_batch

    def train(self, loader, val_loaders, max_step=100000):

        self.gen.train()
        if self.disc is not None:
            self.disc.train()

        # loss stats
        losses = utils.AverageMeters("g_total", "pixel", "disc", "gen", "fm",
                                     "ac", "ac_gen")
        # discriminator stats
        discs = utils.AverageMeters("real_font", "real_uni", "fake_font", "fake_uni",
                                    "real_font_acc", "real_uni_acc",
                                    "fake_font_acc", "fake_uni_acc")
        # etc stats
        stats = utils.AverageMeters("B_style", "B_target", "ac_acc", "ac_gen_acc")

        self.clear_losses()

        self.logger.info("Start training ...")

        for batch in cyclize(loader):
            epoch = self.step // len(loader)
            if self.use_ddp and (self.step % len(loader)) == 0:
                loader.sampler.set_epoch(epoch)

            ref_imgs = batch["ref_imgs"].cuda()
            ref_fids = batch["ref_fids"].cuda()
            ref_decs = batch["ref_decs"].cuda()

            trg_imgs = batch["trg_imgs"].cuda()
            trg_fids = batch["trg_fids"].cuda()
            trg_cids = batch["trg_cids"].cuda()
            trg_decs = batch["trg_decs"].cuda()

            B = len(trg_imgs)
            stats.updates({
                "B_style": ref_imgs.size(0),
                "B_target": B
            })

            sc_feats = self.gen.encode_write(ref_fids, ref_decs, ref_imgs)
            gen_imgs = self.gen.read_decode(trg_fids, trg_decs)

            real_font, real_uni, *real_feats = self.disc(
                trg_imgs, trg_fids, trg_cids, out_feats=self.cfg['fm_layers']
            )

            fake_font, fake_uni = self.disc(gen_imgs.detach(), trg_fids, trg_cids)

            self.add_gan_d_loss([real_font, real_uni], [fake_font, fake_uni])
            self.d_optim.zero_grad()
            self.d_backward()
            self.d_optim.step()

            fake_font, fake_uni, *fake_feats = self.disc(
                gen_imgs, trg_fids, trg_cids, out_feats=self.cfg['fm_layers']
            )
            self.add_gan_g_loss(fake_font, fake_uni)

            self.add_fm_loss(real_feats, fake_feats)

            def racc(x):
                return (x > 0.).float().mean().item()

            def facc(x):
                return (x < 0.).float().mean().item()

            discs.updates({
                "real_font": real_font.mean().item(),
                "real_uni": real_uni.mean().item(),
                "fake_font": fake_font.mean().item(),
                "fake_uni": fake_uni.mean().item(),

                'real_font_acc': racc(real_font),
                'real_uni_acc': racc(real_uni),
                'fake_font_acc': facc(fake_font),
                'fake_uni_acc': facc(fake_uni)
            }, B)

            self.add_pixel_loss(gen_imgs, trg_imgs)

            self.g_optim.zero_grad()

            self.add_ac_losses_and_update_stats(
                sc_feats, ref_decs, gen_imgs, trg_decs, stats
            )
            self.ac_optim.zero_grad()
            self.ac_backward()
            self.ac_optim.step()

            self.g_backward()
            self.g_optim.step()

            loss_dic = self.clear_losses()
            losses.updates(loss_dic, B)  # accum loss stats

            # EMA g
            self.accum_g()
            if self.is_bn_gen:
                self.sync_g_ema(batch)

            torch.cuda.synchronize()

            if self.cfg.rank == 0:
                if self.step % self.cfg.tb_freq == 0:
                    self.plot(losses, discs, stats)

                if self.step % self.cfg.print_freq == 0:
                    self.log(losses, discs, stats)
                    self.logger.debug("GPU Memory usage: max mem_alloc = %.1fM / %.1fM",
                                      torch.cuda.max_memory_allocated() / 1000 / 1000,
                                      torch.cuda.max_memory_cached() / 1000 / 1000)
                    losses.resets()
                    discs.resets()
                    stats.resets()

                    nrow = len(trg_imgs)
                    grid = utils.make_comparable_grid(trg_imgs.detach().cpu(),
                                                      gen_imgs.detach().cpu(),
                                                      nrow=nrow)
                    self.writer.add_image("last", grid)

                if self.step > 0 and self.step % self.cfg.val_freq == 0:
                    epoch = self.step / len(loader)
                    self.logger.info("Validation at Epoch = {:.3f}".format(epoch))

                    if not self.is_bn_gen:
                        self.sync_g_ema(batch)

                    for _key, _loader in val_loaders.items():
                        n_row = _loader.dataset.n_gen
                        self.infer_save_img(_loader, tag=_key, n_row=n_row)

                    self.save(self.cfg.save, self.cfg.get('save_freq', self.cfg.val_freq))
            else:
                pass

            if self.step >= max_step:
                break

            self.step += 1

        self.logger.info("Iteration finished.")

    def infer_ac(self, sc_feats, comp_ids):
        aux_out = self.aux_clf(sc_feats)
        loss = F.cross_entropy(aux_out, comp_ids)
        acc = utils.accuracy(aux_out, comp_ids)
        return loss, acc

    def add_ac_losses_and_update_stats(self, in_sc_feats, in_decs, gen_imgs, trg_decs, stats):
        in_sc_feats = in_sc_feats.flatten(0, 1)
        in_decs = in_decs.flatten(0, 1)
        ac_loss, ac_acc = self.infer_ac(in_sc_feats, in_decs)
        self.ac_losses['ac'] = ac_loss * self.cfg['ac_w']
        stats.ac_acc.update(ac_acc, in_decs.numel())

        trg_decs = trg_decs.flatten(0, 1)
        feats = self.gen_ema.comp_enc(gen_imgs)
        gen_comp_feats = feats["last"].flatten(0, 1)

        ac_gen_loss, ac_gen_acc = self.infer_ac(gen_comp_feats, trg_decs)
        stats.ac_gen_acc.update(ac_gen_acc, trg_decs.numel())
        self.frozen_ac_losses['ac_gen'] = ac_gen_loss * self.cfg['ac_gen_w']

    def log(self, L, D, S):
        self.logger.info(
            f"Step {self.step:7d}\n"
            f"{'|D':<12} {L.disc.avg:7.3f} {'|G':<12} {L.gen.avg:7.3f} {'|FM':<12} {L.fm.avg:7.3f} {'|R_font':<12} {D.real_font_acc.avg:7.3f} {'|F_font':<12} {D.fake_font_acc.avg:7.3f} {'|R_uni':<12} {D.real_uni_acc.avg:7.3f} {'|F_uni':<12} {D.fake_uni_acc.avg:7.3f}\n"
        )
