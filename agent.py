import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wishart, multivariate_normal
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import dataloader
import mvae_moe
import mvae_mopoe
import mvae_poe
import vae_audio
import vae_interoception
import vae_vision
from utils import save_toFile, save_checkpoint

const = 1e-7
beta = 1.0


class Agent:
    def __init__(self, name, args, channel):
        super(Agent, self).__init__()
        self.name = name  # Agent name
        self.args = args
        self.channel = channel  # communication channel
        self.seed = self.args.seed  # Store seed if provided

        if self.seed is not None:
            np.random.seed(self.seed)  # Set NumPy seed
            torch.manual_seed(self.seed)  # Set PyTorch seed
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                
        self.device = args.device
        self.dim = self.args.latent_dim
        self.K = self.args.K  # number of categories
        self.D = self.args.D  # number of data points
        self.mh_epochs = self.args.mh_epochs
        self.runPath = self.args.run_path + self.name
        self.dataloader = None
        self.model = None  # model definition
        self.model_train = None  # model train function
        self.z_means = None  # latent space
        self.z_logvars = None
        self.true_label = np.load('emotion_labels.npy')

        self.mu_prior = None
        self.var_prior = None
        self.ARI = []
        self.self_label = []  
        self.agent_label = [] 
        self.infer_label = [] 
        self.pred_label = []  # predicted labels
        self.acceptedCount = []
        self.lossList = []

        # Hyper-parameters, \mu, \Lambda, and sign w
        self.hyper_mu = np.repeat(0.0, self.dim)
        self.hyper_lamb = np.identity(self.dim)
        self.mu = np.empty((self.K, self.dim))
        self.lamb = np.empty((self.K, self.dim, self.dim))
        self.w = np.random.multinomial(1, [1 / self.K] * self.K, size=self.D)
        self.initialize()

    def initialize(self):
        self.dataloader = dataloader.get_modalities(batch_size=self.args.batch_size, shuffle=False, agent=self.name, _vision=True, _audio=True, _interoception=True)
        if self.args.expert == 'MoE':
            self.model = mvae_moe.MultiVAE(latent_dim=self.dim, device=self.device)
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model, device_ids=[0, 1, 2, 3])
            self.model.to(self.device)
            self.model_train = mvae_moe.train
        elif self.args.expert == 'PoE':
            self.model = mvae_poe.MultiVAE(latent_dim=self.dim, device=self.device)
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model, device_ids=[0, 1, 2, 3])
            self.model.to(self.device)
            self.model_train = mvae_poe.train
        elif self.args.expert == 'MoPoE':
            self.model = mvae_mopoe.MultiVAE(latent_dim=self.dim, device=self.device)
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model, device_ids=[0, 1, 2, 3])
            self.model.to(self.device)
            self.model_train = mvae_mopoe.train
        else:
            print('Error in Expert Description!')

    def initial_mu_lamb(self):
        for k in range(self.K):
            self.lamb[k] = wishart.rvs(df=self.dim, scale=self.hyper_lamb, size=1)
            self.mu[k] = np.random.multivariate_normal(mean=self.hyper_mu,
                                                       cov=np.linalg.inv(beta * self.lamb[k])).flatten()

    def initial_hyperparameter(self):
        centroid = np.mean(self.z_means, axis=0)
        covariance = np.cov(self.z_means.T)
        con_inv = np.linalg.inv(covariance)
        self.hyper_mu = centroid
        self.hyper_lamb = con_inv

    def train(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.model.to(self.device)

        for epoch in range(1, self.args.mvae_epochs + 1):
            save_epoch = 0
            if epoch == self.args.mvae_epochs:
                save_epoch = 1
            loss_current, means, logvars, loss_vision, loss_audio, loss_interoception = self.model_train(args=self.args, model=self.model, optimizer=optimizer,
                mu_prior=self.mu_prior, var_prior=self.var_prior, save_epoch=save_epoch, data_loader=self.dataloader, device=self.device)
            print('Epoch', epoch, ':', loss_current, loss_vision, loss_audio, loss_interoception)
            self.lossList.append(loss_current)

        _means = means.detach().numpy()
        #_logvars = logvars.detach().numpy()
        self.z_means = _means[:self.D]
        #self.z_logvars = _logvars[:self.D]


    def speakTo(self, agent, mode):
        tmp_eta_n = np.zeros((self.K, self.D))
        eta_d = np.zeros((self.D, self.K))
        denominator = np.zeros(self.D)
        numerator = np.zeros(self.D)
        accepted = 0

        # Calculate eta for w
        for k in range(self.K):
            tmp_eta_n[k] = np.diag(
                -0.5 * (self.z_means - self.mu[k]).dot(self.lamb[k]).dot((self.z_means - self.mu[k]).T)).copy()
            tmp_eta_n[k] += 0.5 * np.log(np.linalg.det(self.lamb[k]) + const)
            eta_d[:, k] = np.exp(tmp_eta_n[k])
        eta_d /= (np.sum(eta_d, axis=1, keepdims=True))

        for d in range(self.D):
            # Sampling w
            pvals = eta_d[d]
            if True in np.isnan(np.array(pvals)):
                pvals = [1 / self.K] * self.K
            self.w[d] = np.random.multinomial(n=1, pvals=pvals, size=1).flatten()

            if mode == 0:  # No communication
                self.channel[d] = self.w[d]
                self.pred_label.append(np.argmax(self.channel[d]))
                self.infer_label.append(np.argmax(self.channel[d]))
            elif mode == 1:  # All accepted
                self.channel[d] = self.w[d]
                agent.pred_label.append(np.argmax(self.channel[d]))
                agent.infer_label.append(np.argmax(self.channel[d]))
                
                agent.agent_label.append(np.argmax(agent.w[d]))
                accepted += 1
            else:  # mode == -1: MH sampling
                denominator[d] = multivariate_normal.pdf(agent.z_means[d], mean=agent.mu[np.argmax(self.w[d])],
                                                         cov=np.linalg.inv(agent.lamb[np.argmax(self.w[d])]), allow_singular=True)
                #denominator[d] = multivariate_normal.pdf(agent.z_means[d], mean=agent.mu[np.argmax(self.w[d])],
                                                         #cov=np.linalg.inv(agent.lamb[np.argmax(self.w[d])]))
                numerator[d] = multivariate_normal.pdf(agent.z_means[d], mean=agent.mu[np.argmax(agent.w[d])],
                                                       cov=np.linalg.inv(agent.lamb[np.argmax(agent.w[d])]))
                ratio = min(1, denominator[d] / numerator[d])
                u = np.random.rand()
                if ratio >= u:
                    self.channel[d] = self.w[d]
                    accepted += 1
                else:
                    self.channel[d] = agent.w[d]
                agent.pred_label.append(np.argmax(self.channel[d]))
                agent.infer_label.append(np.argmax(self.channel[d]))
                agent.self_label.append(np.argmax(self.w[d]))
                agent.agent_label.append(np.argmax(agent.w[d]))
        self.acceptedCount.append(accepted)
        
    def update(self):
        beta_hat_k = np.zeros(self.K)
        m_hat_kd = np.zeros((self.K, self.dim))
        w_hat_kdd = np.zeros((self.K, self.dim, self.dim))
        nu_hat_k = np.zeros(self.K)

        for k in range(self.K):
            beta_hat_k[k] = np.sum(self.channel[:, k]) + beta
            m_hat_kd[k] = np.sum(self.channel[:, k] * self.z_means.T, axis=1)
            m_hat_kd[k] += beta * self.hyper_mu
            m_hat_kd[k] /= beta_hat_k[k]
            tmp_w_dd = np.dot((self.channel[:, k] * self.z_means.T), self.z_means)
            tmp_w_dd += beta * np.dot(self.hyper_mu.reshape(self.dim, 1), self.hyper_mu.reshape(1, self.dim))
            tmp_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape(self.dim, 1), m_hat_kd[k].reshape(1, self.dim))
            tmp_w_dd += np.linalg.inv(self.hyper_lamb)
            w_hat_kdd[k] = np.linalg.inv(tmp_w_dd)
            nu_hat_k[k] = np.sum(self.channel[:, k]) + self.dim

            # sampling \lambda and \mu
            self.lamb[k] = wishart.rvs(size=1, df=nu_hat_k[k], scale=w_hat_kdd[k]) + const
            self.mu[k] = np.random.multivariate_normal(size=1, mean=m_hat_kd[k],
                                                       cov=np.linalg.inv(beta_hat_k[k] * self.lamb[k])).flatten()

    def evaluate(self):
        if len(self.infer_label) > self.D*self.mh_epochs:
            self.infer_label = self.infer_label[self.D*self.mh_epochs:]
        if len(self.self_label) > self.D*self.mh_epochs:
            self.self_label = self.self_label[self.D*self.mh_epochs:]
        if len(self.agent_label) > self.D*self.mh_epochs:
            self.agent_label = self.agent_label[self.D*self.mh_epochs:]
            
        if len(self.pred_label) > self.D:
            self.pred_label = self.pred_label[self.D:]
        self.ARI.append(adjusted_rand_score(self.true_label, self.pred_label))


    def prior_update(self):
        self.mu_prior = np.zeros((self.D, self.dim))
        self.var_prior = np.zeros((self.D, self.dim))
        for d in range(self.D):
            self.mu_prior[d] = self.mu[np.argmax(self.channel[d])]
            self.var_prior[d] = np.diag(np.linalg.inv(self.lamb[np.argmax(self.channel[d])]))

    def reconstruction(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        vision_recon = []
        audio_recon = []
        interoception_recon = []
        for batch_idx, data in enumerate(self.dataloader):
            vision = data[0].to(self.device)
            audio = data[1].to(self.device)
            interoception = data[2].to(self.device)
            _recon_vis, _recon_aud, _recon_tac, _, _ = self.model(vision=vision, audio=audio, interoception=interoception)
            _recon_vis = _recon_vis.cpu()
            _recon_aud = _recon_aud.cpu()
            _recon_tac = _recon_tac.cpu()
            _recon_vis = _recon_vis.detach().numpy()
            _recon_aud = _recon_aud.detach().numpy()
            _recon_tac = _recon_tac.detach().numpy()
            for _recon in _recon_vis:
                vision_recon.append(_recon)
            for _recon in _recon_aud:
                audio_recon.append(_recon)
            for _recon in _recon_tac:
                interoception_recon.append(_recon)
        return vision_recon, audio_recon, interoception_recon
