"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy

import torch
import torch.nn as nn


def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


class FUNITModel(nn.Module):
    def __init__(self, gen, dis):
        super(FUNITModel, self).__init__()
        self.gen = gen
        self.dis = dis
        self.gen_test = copy.deepcopy(self.gen)
        
    def gen_update(self, co_data, cl_data, tg_data, hp):
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        tg = tg_data[0].cuda()
        lb = tg_data[1].cuda()
        unicode_lb = tg_data[2].cuda()
        
        c_xa = self.gen.enc_content(xa)
        s_xb = self.gen.enc_class_model(xb)
        xt = self.gen.decode(c_xa, s_xb)  # translation
        l_adv_t, gacc_t, xt_gan_feat = self.dis.calc_gen_loss(xt, lb, unicode_lb)
        _, _, tg_gan_feat = self.dis(tg, lb, unicode_lb)
        l_c_rec = recon_criterion(xt_gan_feat.mean(3).mean(2),
                                  tg_gan_feat.mean(3).mean(2))
        l_x_rec = recon_criterion(xt, tg)
        l_adv = l_adv_t
        acc = gacc_t
        l_total = (hp['gan_w'] * l_adv + hp['r_w'] * l_x_rec + hp[
            'fm_w'] * l_c_rec)
        l_total.backward()
        
        return l_total, l_adv, l_x_rec, l_c_rec, l_c_rec, acc
    
    def dis_update(self, co_data, cl_data, tg_data, hp):
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        tg = tg_data[0].cuda()
        lb = tg_data[1].cuda()
        unicode_lb = tg_data[2].cuda()
        
        tg.requires_grad_()
        l_real_pre, acc_r, resp_r, unicode_resp_r = self.dis.calc_dis_real_loss(tg, lb, unicode_lb)
        l_real = hp['gan_w'] * l_real_pre
        l_real.backward(retain_graph=True)

        l_reg_pre = self.dis.calc_grad2(resp_r, tg)
        l_reg_pre_uni = self.dis.calc_grad2(unicode_resp_r, tg)
        l_reg = 10 * (l_reg_pre + l_reg_pre_uni)
        l_reg.backward()
        with torch.no_grad():
            c_xa = self.gen.enc_content(xa)
            s_xb = self.gen.enc_class_model(xb)
            xt = self.gen.decode(c_xa, s_xb)
        l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach(),
                                                              lb,
                                                              unicode_lb)
        l_fake = hp['gan_w'] * l_fake_p
        l_fake.backward()
        l_total = l_fake + l_real + l_reg
        acc = 0.5 * (acc_f + acc_r)
        
        return l_total, l_fake_p, l_real_pre, l_reg_pre, acc
    
#     def forward(self, co_data, cl_data, tg_data, hp, mode):
#         xa = co_data[0].cuda()
#         xb = cl_data[0].cuda()
#         tg = tg_data[0].cuda()
#         lb = tg_data[1].cuda()
#         unicode_lb = tg_data[2].cuda()
#         if mode == 'gen_update':
#             c_xa = self.gen.enc_content(xa)
#             s_xb = self.gen.enc_class_model(xb)
#             xt = self.gen.decode(c_xa, s_xb)  # translation
#             l_adv_t, gacc_t, xt_gan_feat = self.dis.calc_gen_loss(xt, lb, unicode_lb)
#             _, _, tg_gan_feat = self.dis(tg, lb, unicode_lb)
#             l_c_rec = recon_criterion(xt_gan_feat.mean(3).mean(2),
#                                       tg_gan_feat.mean(3).mean(2))
#             l_x_rec = recon_criterion(xt, tg)
#             l_adv = l_adv_t
#             acc = gacc_t
#             l_total = (hp['gan_w'] * l_adv + hp['r_w'] * l_x_rec + hp[
#                 'fm_w'] * l_c_rec)
#             l_total.backward()
#             return l_total, l_adv, l_x_rec, l_c_rec, l_c_rec, acc
#         elif mode == 'dis_update':
#             tg.requires_grad_()
#             l_real_pre, acc_r, resp_r, unicode_resp_r = self.dis.calc_dis_real_loss(tg, lb, unicode_lb)
#             l_real = hp['gan_w'] * l_real_pre
#             l_real.backward(retain_graph=True)

#             l_reg_pre = self.dis.calc_grad2(resp_r, tg)
#             l_reg_pre_uni = self.dis.calc_grad2(unicode_resp_r, tg)
#             l_reg = 10 * (l_reg_pre + l_reg_pre_uni)
#             l_reg.backward()
#             with torch.no_grad():
#                 c_xa = self.gen.enc_content(xa)
#                 s_xb = self.gen.enc_class_model(xb)
#                 xt = self.gen.decode(c_xa, s_xb)
#             l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach(),
#                                                                   lb,
#                                                                   unicode_lb)
#             l_fake = hp['gan_w'] * l_fake_p
#             l_fake.backward()
#             l_total = l_fake + l_real + l_reg
#             acc = 0.5 * (acc_f + acc_r)
#             return l_total, l_fake_p, l_real_pre, l_reg_pre, acc
#         else:
#             assert 0, 'Not support operation'

    def test(self, co_data, cl_data):
        self.eval()
        self.gen.eval()
        self.gen_test.eval()
        c_xa = self.gen_test.enc_content(co_data)
        s_xb = self.gen_test.enc_class_model(cl_data)
        xt = self.gen_test.decode(c_xa, s_xb)
        self.train()

        if cl_data.size(1) > 3:
            return xt
        else:
            return xt

    def translate_k_shot(self, co_data, cl_data):
        self.eval()
        xa = co_data.cuda()
        xb = cl_data.cuda()
        c_xa = self.gen_test.enc_content(xa)
        s_xb = self.gen_test.enc_class_model(xb)
        xt = self.gen_test.decode(c_xa, s_xb)
        return xt

    def compute_k_style(self, style_batch, k):
        self.eval()
        style_batch = style_batch.cuda()
        s_xb_before = self.gen_test.enc_class_model(style_batch)
        s_xb_after = s_xb_before.squeeze(-1).permute(1, 2, 0)
        s_xb_pool = torch.nn.functional.avg_pool1d(s_xb_after, k)
        s_xb = s_xb_pool.permute(2, 0, 1).unsqueeze(-1)
        return s_xb

    def translate_simple(self, content_image, class_code):
        self.eval()
        xa = content_image.cuda()
        s_xb_current = class_code.cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current
