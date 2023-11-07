import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os
from simplex import Simplex_CLASS
import random
import matplotlib.pyplot as plt
def generate_simplex_noise(
        Simplex_instance, x, t, random_param=False, octave=6, persistence=0.8, frequency=64,
        in_channels=1
        ):
    noise = torch.empty(x.shape).to(x.device)
    for i in range(in_channels):
        Simplex_instance.newSeed()
        if random_param:
            param = random.choice(
                    [(2, 0.6, 16), (6, 0.6, 32), (7, 0.7, 32), (10, 0.8, 64), (5, 0.8, 16), (4, 0.6, 16), (1, 0.6, 64),
                     (7, 0.8, 128), (6, 0.9, 64), (2, 0.85, 128), (2, 0.85, 64), (2, 0.85, 32), (2, 0.85, 16),
                     (2, 0.85, 8),
                     (2, 0.85, 4), (2, 0.85, 2), (1, 0.85, 128), (1, 0.85, 64), (1, 0.85, 32), (1, 0.85, 16),
                     (1, 0.85, 8),
                     (1, 0.85, 4), (1, 0.85, 2), ]
                    )
            # 2D octaves seem to introduce directional artifacts in the top left
            noise[:, i, ...] = torch.unsqueeze(
                    torch.from_numpy(
                            # Simplex_instance.rand_2d_octaves(
                            #         x.shape[-2:], param[0], param[1],
                            #         param[2]
                            #         )
                            Simplex_instance.rand_3d_fixed_T_octaves(
                                    x.shape[-2:], t.detach().cpu().numpy(), param[0], param[1],
                                    param[2]
                                    )
                            ).to(x.device), 0
                    ).repeat(x.shape[0], 1, 1, 1)
        noise[:, i, ...] = torch.unsqueeze(
                torch.from_numpy(
                        # Simplex_instance.rand_2d_octaves(
                        #         x.shape[-2:], octave,
                        #         persistence, frequency
                        #         )
                        Simplex_instance.rand_3d_fixed_T_octaves(
                                x.shape[-2:], t.detach().cpu().numpy(), octave,
                                persistence, frequency
                                )
                        ).to(x.device), 0
                ).repeat(x.shape[0], 1, 1, 1)
    return noise
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
    
   
    with torch.no_grad():
        
        #setup vectors used in the algorithm
        singulars = H_funcs.singulars()
        Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
        Sigma[:singulars.shape[0]] = singulars
        U_t_y = H_funcs.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

        #initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
        inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
        inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
        inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)     

        # implement p(x_T | x_0, y) as given in the paper
        # if eigenvalue is too small, we just treat it as zero (only for init) 
        init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
        init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
        init_y = init_y.view(*x.size())
        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
        remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas
        if True:    
            #setup iteration variables
            x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs = [x]
        
            #iterate over the timesteps
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
                
                if et.size(1) == 6:
                    et = et[:, :3]
                
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                #variational inference conditioned on y
                sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
                sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
                xt_mod = xt / at.sqrt()[0, 0, 0, 0]
                V_t_x = H_funcs.Vt(xt_mod)
                SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
                V_t_x0 = H_funcs.Vt(x0_t)
                SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

                falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
                cond_before_lite = singulars * sigma_next > sigma_0
                cond_after_lite = singulars * sigma_next < sigma_0
                cond_before = torch.hstack((cond_before_lite, falses))
                cond_after = torch.hstack((cond_after_lite, falses))

                std_nextC = sigma_next * etaC
                sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

                std_nextA = sigma_next * etaA
                sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
                
                diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

                #missing pixels
                Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

                #less noisy than y (after)                  
                Vt_xt_mod_next[:, cond_after] = \
                    V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite] + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
                #noisier than y (before)
                Vt_xt_mod_next[:, cond_before] = \
                    (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite])

                #aggregate all 3 cases and give next prediction
                xt_mod_next = H_funcs.V(Vt_xt_mod_next)
                xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
        else:
        
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [x]
            x0_preds = []
            betas = b
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(betas, t.long())
                atm1 = compute_alpha(betas, next_t.long())
                beta_t = 1 - at / atm1
                x = xs[-1].to('cuda')

                output = model(x, t.float())
                e = output

                x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
                x0_from_e = torch.clamp(x0_from_e, -1, 1)
                x0_preds.append(x0_from_e.to('cpu'))
                mean_eps = (
                    (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
                ) / (1.0 - at)

                mean = mean_eps
                noise = torch.randn_like(x)
                mask = 1 - (t == 0).float()
                mask = mask.view(-1, 1, 1, 1)
                logvar = beta_t.log()
                sample = mean + mask * torch.exp(0.5 * logvar) * noise
                xs.append(sample.to('cpu'))
            print("over over")
    

    return xs, x0_preds