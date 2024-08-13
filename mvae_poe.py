import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import m_elbo
from vae_vision import Encoder_vision, Decoder_vision
from vae_audio import Encoder_audio, Decoder_audio
from vae_interoception import Encoder_interoception, Decoder_interoception
eta = 1e-6
eps = 1e-8


class MultiVAE(nn.Module):
    def __init__(self, latent_dim, device):
        super(MultiVAE, self).__init__()
        self.encoder_vision = Encoder_vision(latent_dim)
        self.decoder_vision = Decoder_vision(latent_dim)
        self.encoder_audio = Encoder_audio(latent_dim)
        self.decoder_audio = Decoder_audio(latent_dim)
        self.encoder_interoception = Encoder_interoception(latent_dim)
        self.decoder_interoception = Decoder_interoception(latent_dim)
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, vision=None, audio=None, interoception=None):
        mu, logvar = self.encoder(vision=vision, audio=audio, interoception=interoception)
        latent = self.reparameterize(mu, logvar)
        if vision is not None:
            vision_recon = self.decoder_vision(latent)
        else:
            vision_recon = None
        if audio is not None:
            audio_recon = self.decoder_audio(latent)
        else:
            audio_recon = None
        if interoception is not None:
            interoception_recon = self.decoder_interoception(latent)
        else:
            interoception_recon = None
        return vision_recon, audio_recon, interoception_recon, mu, logvar

    def encoder(self, vision=None, audio=None, interoception=None, mu_prior=None, var_prior=None):
        batch_size_current = vision.size(0) if vision is not None else audio.size(
            0) if audio is not None else interoception.size(0)
        mu, logvar = self.prior_distribution(batch_size_current, mu_prior=mu_prior, var_prior=var_prior)
        mu = mu.to(self.device)
        logvar = logvar.to(self.device)
        if vision is not None:
            mu_vision, logvar_vision = self.encoder_vision(vision)
            mu = torch.cat((mu, mu_vision.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, logvar_vision.unsqueeze(0)), dim=0)
        if audio is not None:
            mu_audio, logvar_audio = self.encoder_audio(audio)
            mu = torch.cat((mu, mu_audio.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, logvar_audio.unsqueeze(0)), dim=0)
        if interoception is not None:
            mu_interoception, logvar_interoception = self.encoder_interoception(interoception)
            mu = torch.cat((mu, mu_interoception.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, logvar_interoception.unsqueeze(0)), dim=0)

        mu, logvar = product_experts(mu, logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps_ = Variable(std.data.new(std.size()).normal_())
            return eps_.mul(std).add_(mu)
        else:
            return mu

    def prior_distribution(self, _batch_size, mu_prior=None, var_prior=None):
        if mu_prior is None:
            mu = Variable(torch.zeros(1, _batch_size, self.latent_dim))
            logvar = Variable(torch.zeros(1, _batch_size, self.latent_dim))
        else:
            mu = Variable(torch.from_numpy(mu_prior[None, :]).to(torch.float32))
            logvar = Variable(torch.from_numpy(var_prior[None, :]).to(torch.float32))
        return mu, logvar


def product_experts(mu, logvar):
    var = torch.exp(logvar) + eps
    t = 1. / (var + eps)
    pd_mu = torch.sum(mu * t, dim=0) / torch.sum(t, dim=0)
    pd_var = 1. / torch.sum(t, dim=0)
    pd_logvar = torch.log(pd_var + eps)
    return pd_mu, pd_logvar


def train(args, model, data_loader, optimizer, device='cuda', mu_prior=None, var_prior=None, save_epoch=0, desc=None):
    model.train()
    loss_average = 0
    loss_vision = 0
    loss_audio = 0
    loss_interoception = 0
    means = torch.zeros(1, args.latent_dim)
    logvars = torch.zeros(1, args.latent_dim)
    for batch_idx, data in enumerate(data_loader):
        if desc == 'vision_interoception':
            vision = data[0].to(device)
            audio = None
            interoception = data[1].to(device)
        elif desc == 'audio_interoception':
            vision = None
            audio = data[0].to(device)
            interoception = data[1].to(device)
        elif desc == 'vision_audio':
            vision = data[0].to(device)
            audio = data[1].to(device)
            interoception = None
        else:
            vision = data[0].to(device)
            audio = data[1].to(device)
            interoception = data[2].to(device)
        vision_recon, audio_recon, interoception_recon, mu, logvar = model(vision=vision, audio=audio, interoception=interoception)
        loss_vision = 0
        loss_audio = 0
        loss_interoception = 0
        no_average = 1

        if mu_prior is not None:
            loss_joint, la, lb, lc = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                                recon_audio=audio_recon, audio=audio, recon_interoception=interoception_recon, interoception=interoception,
                                mu=mu, logvar=logvar, mu_prior=mu_prior, var_prior=var_prior,
                                variational_beta=args.variational_beta, dim=args.latent_dim)
            if vision is not None:
                vision_recon1, _, _, mu_vision, logvar_vision = model(vision=vision, audio=None, interoception=None)
                loss_vision, _, _, _ = m_elbo(args=args, recon_vision=vision_recon1, vision=vision,
                                     recon_audio=None, audio=None, recon_interoception=None, interoception=None,
                                     mu=mu_vision, logvar=logvar_vision, mu_prior=mu_prior, var_prior=var_prior,
                                     variational_beta=args.variational_beta, dim=args.latent_dim)
                no_average += 1
            if audio is not None:
                _, audio_recon1, _, mu_audio, logvar_audio = model(vision=None, audio=audio, interoception=None)
                loss_audio, _, _, _ = m_elbo(args=args, recon_vision=None, vision=None,
                                    recon_audio=audio_recon1, audio=audio, recon_interoception=None, interoception=None,
                                    mu=mu_audio, logvar=logvar_audio, mu_prior=mu_prior, var_prior=var_prior,
                                    variational_beta=args.variational_beta, dim=args.latent_dim)
                no_average += 1
            if interoception is not None:
                _, _, interoception_recon1, mu_interoception, logvar_interoception = model(vision=None, audio=None, interoception=interoception)
                loss_interoception, _, _, _ = m_elbo(args=args, recon_vision=None, vision=None,
                                      recon_audio=None, audio=None, recon_interoception=interoception_recon1, interoception=interoception,
                                      mu=mu_interoception, logvar=logvar_interoception, mu_prior=mu_prior, var_prior=var_prior,
                                      variational_beta=args.variational_beta, dim=args.latent_dim)
                no_average += 1
        else:
            loss_joint, la, lb, lc = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                                recon_audio=audio_recon, audio=audio, recon_interoception=interoception_recon, interoception=interoception,
                                mu=mu, logvar=logvar, variational_beta=args.variational_beta)
            if vision is not None:
                vision_recon1, _, _, mu_vision, logvar_vision = model(vision=vision, audio=None, interoception=None)
                loss_vision, _, _, _  = m_elbo(args=args, recon_vision=vision_recon1, vision=vision,
                                     recon_audio=None, audio=None, recon_interoception=None, interoception=None,
                                     mu=mu_vision, logvar=logvar_vision, variational_beta=args.variational_beta)
                no_average += 1
            if audio is not None:
                _, audio_recon1, _, mu_audio, logvar_audio = model(vision=None, audio=audio, interoception=None)
                loss_audio, _, _, _ = m_elbo(args=args, recon_vision=None, vision=None,
                                    recon_audio=audio_recon1, audio=audio, recon_interoception=None, interoception=None,
                                    mu=mu_audio, logvar=logvar_audio, variational_beta=args.variational_beta)
                no_average += 1
            if interoception is not None:
                _, _, interoception_recon1, mu_interoception, logvar_interoception = model(vision=None, audio=None, interoception=interoception)
                loss_interoception, _, _, _ = m_elbo(args=args, recon_vision=None, vision=None,
                                      recon_audio=None, audio=None, recon_interoception=interoception_recon1, interoception=interoception,
                                      mu=mu_interoception, logvar=logvar_interoception, variational_beta=args.variational_beta)
                no_average += 1

        loss = (loss_joint + loss_vision + loss_audio + loss_interoception) / no_average
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_average += loss.item()
        loss_vision += la.item()
        loss_audio += lb.item()
        loss_interoception += lc.item()
        if save_epoch != 0:
            for mean in mu:
                mean = mean.cpu()
                means = torch.cat((means, mean.unsqueeze(0)), dim=0)
            for lv in logvar:
                lv = lv.cpu()
                logvars = torch.cat((logvars, lv.unsqueeze(0)), dim=0)
    loss_average /= len(data_loader.dataset)
    return loss_average, means[1:], logvars[1:], loss_vision, loss_audio, loss_interoception

