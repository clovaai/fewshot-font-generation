"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.trainer import BaseTrainer, cyclize, binarize_labels, expert_assign
import base.utils as utils

from .hsic import RbfHSIC


def to_batch(batch):
    in_batch = {
        "style_imgs": batch["style_imgs"].cuda(),
        "char_imgs": batch["char_imgs"].cuda(),
    }
    return in_batch


class MXTrainer(BaseTrainer):
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
        losses = utils.AverageMeters("g_total", "pixel", "disc", "gen", "fm", "indp_exp", "indp_fact",
                                     "ac_s", "ac_c", "cross_ac_s", "cross_ac_c",
                                     "ac_gen_s", "ac_gen_c", "cross_ac_gen_s", "cross_ac_gen_c")
        # discriminator stats
        discs = utils.AverageMeters("real_font", "real_uni", "fake_font", "fake_uni",
                                    "real_font_acc", "real_uni_acc",
                                    "fake_font_acc", "fake_uni_acc")
        # etc stats
        stats = utils.AverageMeters("B", "ac_acc_s", "ac_acc_c", "ac_gen_acc_s", "ac_gen_acc_c")

        self.clear_losses()

        self.logger.info("Start training ...")

        for batch in cyclize(loader):
            epoch = self.step // len(loader)
            if self.use_ddp and (self.step % len(loader)) == 0:
                loader.sampler.set_epoch(epoch)

            style_imgs = batch["style_imgs"].cuda()
            style_fids = batch["style_fids"].cuda()
            style_decs = batch["style_decs"]
            char_imgs = batch["char_imgs"].cuda()
            char_fids = batch["char_fids"].cuda()
            char_decs = batch["char_decs"]

            trg_imgs = batch["trg_imgs"].cuda()
            trg_fids = batch["trg_fids"].cuda()
            trg_cids = batch["trg_cids"].cuda()
            trg_decs = batch["trg_decs"]

            ##############################################################
            # infer
            ##############################################################

            B = len(trg_imgs)
            n_s = style_imgs.shape[1]
            n_c = char_imgs.shape[1]

            style_feats = self.gen.encode(style_imgs.flatten(0, 1))  # (B*n_s, n_exp, *feat_shape)
            char_feats = self.gen.encode(char_imgs.flatten(0, 1))

            self.add_indp_exp_loss(torch.cat([style_feats["last"], char_feats["last"]]))

            style_facts_s = self.gen.factorize(style_feats, 0)  # (B*n_s, n_exp, *feat_shape)
            style_facts_c = self.gen.factorize(style_feats, 1)
            char_facts_s = self.gen.factorize(char_feats, 0)
            char_facts_c = self.gen.factorize(char_feats, 1)

            self.add_indp_fact_loss(
                [style_facts_s["last"], style_facts_c["last"]],
                [style_facts_s["skip"], style_facts_c["skip"]],
                [char_facts_s["last"], char_facts_c["last"]],
                [char_facts_s["skip"], char_facts_c["skip"]],
                                  )

            mean_style_facts = {k: utils.add_dim_and_reshape(v, 0, (-1, n_s)).mean(1) for k, v in style_facts_s.items()}
            mean_char_facts = {k: utils.add_dim_and_reshape(v, 0, (-1, n_c)).mean(1) for k, v in char_facts_c.items()}
            gen_feats = self.gen.defactorize(mean_style_facts, mean_char_facts)
            gen_imgs = self.gen.decode(gen_feats)

            stats.updates({
                "B": B,
            })

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
                torch.cat([style_facts_s["last"], char_facts_s["last"]]),
                torch.cat([style_fids.flatten(), char_fids.flatten()]),
                torch.cat([style_facts_c["last"], char_facts_c["last"]]),
                style_decs + char_decs,
                gen_imgs,
                trg_fids,
                trg_decs,
                stats
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
                    grid = utils.make_comparable_grid(*style_imgs.transpose(0, 1).detach().cpu(),
                                                      *char_imgs.transpose(0, 1).detach().cpu(),
                                                      trg_imgs.detach().cpu(),
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

    def add_indp_exp_loss(self, exps):
        exps = [F.adaptive_avg_pool2d(exps[:, i], 1).squeeze() for i in range(exps.shape[1])]
        exp_pairs = [*combinations(exps, 2)]

        crit = RbfHSIC(1)
        for pair in exp_pairs:
            self.add_loss(pair, self.g_losses, "indp_exp", self.cfg["indp_exp_w"], crit)

    def add_indp_fact_loss(self, *exp_pairs):
        pairs = []
        for _exp1, _exp2 in exp_pairs:
            _pairs = [(F.adaptive_avg_pool2d(_exp1[:, i], 1).squeeze(),
                       F.adaptive_avg_pool2d(_exp2[:, i], 1).squeeze())
                      for i in range(_exp1.shape[1])]
            pairs += _pairs

        crit = RbfHSIC(1)
        for pair in pairs:
            self.add_loss(pair, self.g_losses, "indp_fact", self.cfg["indp_fact_w"], crit)

    def infer_comp_ac(self, fact_experts, comp_ids):
        B, n_experts = fact_experts.shape[:2]

        ac_logit_s_flat, ac_logit_c_flat = self.aux_clf(fact_experts.flatten(0, 1))

        n_s = ac_logit_s_flat.shape[-1]
        ac_prob_s_flat = nn.Softmax(dim=-1)(ac_logit_s_flat)
        uniform_dist_s = torch.zeros_like(ac_prob_s_flat).fill_((1./n_s)).cuda()
        uniform_loss_s = F.kl_div(ac_prob_s_flat, uniform_dist_s, reduction="batchmean")  # causes increasing weight norm ; to be modified

        ac_logit_c = ac_logit_c_flat.reshape((B, n_experts, -1))  # (bs, n_exp, n_comps)
        n_comps = ac_logit_c.shape[-1]
        binary_comp_ids = binarize_labels(comp_ids, n_comps).cuda()
        ac_loss_c = torch.as_tensor(0.).cuda()
        accs = 0.

        for _b_comp_id, _logit in zip(binary_comp_ids, ac_logit_c):
            _prob = nn.Softmax(dim=-1)(_logit)  # (n_exp, n_comp)
            T_probs = _prob.T[_b_comp_id].detach().cpu()  # (n_T, n_exp)
            cids, eids = expert_assign(T_probs)
            _max_ids = torch.where(_b_comp_id)[0][cids]
            ac_loss_c += F.cross_entropy(_logit[eids], _max_ids)
            acc = T_probs[cids, eids].sum() / n_experts
            accs += acc

        ac_loss_c /= B
        accs /= B

        return ac_loss_c, uniform_loss_s, accs.item()

    def infer_style_ac(self, fact_experts, style_ids):
        B, n_experts = fact_experts.shape[:2]
        ac_in_flat = fact_experts.flatten(0, 1)
        style_ids_flat = style_ids.repeat_interleave(n_experts, dim=0)

        ac_logit_s_flat, ac_logit_c_flat = self.aux_clf(ac_in_flat)
        ac_loss_s = F.cross_entropy(ac_logit_s_flat, style_ids_flat)

        n_c = ac_logit_c_flat.shape[-1]
        ac_prob_c_flat = nn.Softmax(dim=-1)(ac_logit_c_flat)
        uniform_dist_c = torch.zeros_like(ac_prob_c_flat).fill_((1./n_c)).cuda()
        uniform_loss_c = F.kl_div(ac_prob_c_flat, uniform_dist_c, reduction="batchmean")  # causes increasing weight norm ; to be modified

        _, est_ids = ac_logit_s_flat.max(dim=-1)
        acc = (style_ids_flat == est_ids).float().mean().item()

        return ac_loss_s, uniform_loss_c, acc

    def add_ac_losses_and_update_stats(self, style_facts, style_ids, char_facts, comp_ids,
                                       gen_imgs, gen_style_ids, gen_comp_ids, stats):

        ac_loss_s, cross_ac_loss_s, acc_s = self.infer_style_ac(style_facts, style_ids)
        ac_loss_c, cross_ac_loss_c, acc_c = self.infer_comp_ac(char_facts, comp_ids)

        self.ac_losses["ac_s"] = ac_loss_s * self.cfg["ac_w"]
        self.ac_losses["ac_c"] = ac_loss_c * self.cfg["ac_w"]
        self.ac_losses["cross_ac_s"] = cross_ac_loss_s * self.cfg["ac_w"] * self.cfg["ac_cross_w"]
        self.ac_losses["cross_ac_c"] = cross_ac_loss_c * self.cfg["ac_w"] * self.cfg["ac_cross_w"]
        stats.ac_acc_s.update(acc_s, len(style_ids))
        stats.ac_acc_c.update(acc_c, sum([*map(len, comp_ids)]))

        gen_feats = self.gen_ema.encode(gen_imgs)
        gen_style_facts = self.gen_ema.factorize(gen_feats, 0)["last"]
        gen_char_facts = self.gen_ema.factorize(gen_feats, 1)["last"]

        gen_ac_loss_s, gen_cross_ac_loss_s, gen_acc_s = self.infer_style_ac(gen_style_facts, gen_style_ids)
        gen_ac_loss_c, gen_cross_ac_loss_c, gen_acc_c = self.infer_comp_ac(gen_char_facts, gen_comp_ids)
        stats.ac_gen_acc_s.update(gen_acc_s, len(gen_style_ids))
        stats.ac_gen_acc_c.update(gen_acc_c, sum([*map(len, gen_comp_ids)]))

        self.frozen_ac_losses['ac_gen_s'] = gen_ac_loss_s * self.cfg['ac_gen_w']
        self.frozen_ac_losses['ac_gen_c'] = gen_ac_loss_c * self.cfg['ac_gen_w']
        self.frozen_ac_losses['cross_ac_gen_s'] = gen_cross_ac_loss_s * self.cfg['ac_gen_w'] * self.cfg["ac_cross_w"]
        self.frozen_ac_losses['cross_ac_gen_c'] = gen_cross_ac_loss_c * self.cfg['ac_gen_w'] * self.cfg["ac_cross_w"]

    def plot(self, losses, discs, stats):
        tag_scalar_dic = {
            'train/g_total_loss': losses.g_total.val,
            'train/pixel_loss': losses.pixel.val,
            'train/indp_exp_loss': losses.indp_exp.val,
            'train/indp_fact_loss': losses.indp_fact.val,
        }

        if self.disc is not None:
            tag_scalar_dic.update({
                'train/d_real_font': discs.real_font.val,
                'train/d_real_uni': discs.real_uni.val,
                'train/d_fake_font': discs.fake_font.val,
                'train/d_fake_uni': discs.fake_uni.val,

                'train/d_real_font_acc': discs.real_font_acc.val,
                'train/d_real_uni_acc': discs.real_uni_acc.val,
                'train/d_fake_font_acc': discs.fake_font_acc.val,
                'train/d_fake_uni_acc': discs.fake_uni_acc.val
            })

            if self.cfg['fm_w'] > 0.:
                tag_scalar_dic['train/feature_matching'] = losses.fm.val

        if self.aux_clf is not None:
            tag_scalar_dic.update({
                'train/ac_loss_s': losses.ac_s.val,
                'train/ac_loss_c': losses.ac_c.val,
                'train/cross_ac_loss_s': losses.cross_ac_s.val,
                'train/cross_ac_loss_c': losses.cross_ac_c.val,
                'train/ac_acc_s': stats.ac_acc_s.val,
                'train/ac_acc_c': stats.ac_acc_c.val
            })

            if self.cfg['ac_gen_w'] > 0.:
                tag_scalar_dic.update({
                    'train/ac_gen_loss_s': losses.ac_gen_s.val,
                    'train/ac_gen_loss_c': losses.ac_gen_c.val,
                    'train/cross_ac_gen_loss_s': losses.cross_ac_gen_s.val,
                    'train/cross_ac_gen_loss_c': losses.cross_ac_gen_c.val,
                    'train/ac_gen_acc_s': stats.ac_gen_acc_s.val,
                    'train/ac_gen_acc_c': stats.ac_gen_acc_c.val
                })

        self.writer.add_scalars(tag_scalar_dic, self.step)

    def log(self, L, D, S):
        self.logger.info(
            f"Step {self.step:7d}\n"
            f"{'|D':<12} {L.disc.avg:7.3f} {'|G':<12} {L.gen.avg:7.3f} {'|FM':<12} {L.fm.avg:7.3f} {'|R_font':<12} {D.real_font_acc.avg:7.3f} {'|F_font':<12} {D.fake_font_acc.avg:7.3f} {'|R_uni':<12} {D.real_uni_acc.avg:7.3f} {'|F_uni':<12} {D.fake_uni_acc.avg:7.3f}\n"
            f"{'|AC_s':<12} {L.ac_s.avg:7.3f} {'|AC_c':<12} {L.ac_c.avg:7.3f} {'|cr_AC_s':<12} {L.cross_ac_s.avg:7.3f} {'|cr_AC_c':<12} {L.cross_ac_c.avg:7.3f} {'|AC_acc_s':<12} {S.ac_acc_s.avg:7.1%} {'|AC_acc_c':<12} {S.ac_acc_c.avg:7.1%}\n"
            f"{'|AC_g_s':<12} {L.ac_gen_s.avg:7.3f} {'|AC_g_c':<12} {L.ac_gen_c.avg:7.3f} {'|cr_AC_g_s':<12} {L.cross_ac_gen_s.avg:7.3f} {'|cr_AC_g_c':<12} {L.cross_ac_gen_c.avg:7.3f} {'|AC_g_acc_s':<12} {S.ac_gen_acc_s.avg:7.1%} {'|AC_g_acc_c':<12} {S.ac_gen_acc_c.avg:7.1%}\n"
            f"{'|L1':<12} {L.pixel.avg:7.3f} {'|INDP_EXP':<12} {L.indp_exp.avg:7.4f} {'|INDP_FACT':<12} {L.indp_fact.avg:7.4f}"
        )
