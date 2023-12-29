import os
import time
import math
import copy
import numpy as np
import torch
from metrics import averageMeter
from utils.utils import roc_btw_arr

class BaseTrainer:
    """Trainer for a conventional iterative training of model for classification"""
    def __init__(self, optimizer, optimizer_pre, optimizer_e, training_cfg, device):
        self.training_cfg = training_cfg
        self.device = device
        self.optimizer = optimizer
        self.optimizer_pre = optimizer_pre
        self.optimizer_e = optimizer_e

    def train(self, model, d_dataloaders, logger=None, logdir=""):
        cfg = self.training_cfg
    
        time_meter = averageMeter()
        train_loader, val_loader, test_loader = (d_dataloaders["training"], d_dataloaders["validation"], d_dataloaders["test"])
        OOD_val_loader, OOD_test_loader = (d_dataloaders["OOD_validation"], d_dataloaders["OOD_test"])
        kwargs = {'dataset_size': len(train_loader.dataset)}
        i_iter = 0
        best_val_loss = np.inf
        bs = train_loader.batch_size
        z_dim = model.encoder.z_dim
        z_shape = (bs, z_dim)
        model.encoder.load_state_dict(torch.load("pretrained/encoder_ho_1_z_dim_15.pth"))
        model.decoder.load_state_dict(torch.load("pretrained/decoder_ho_1_z_dim_15.pth"))
        # for i_epoch in range(1, cfg['n_epoch_pre'] + 1):
        #     for x, _ in train_loader:
        #         model.train()
        #         d_train = model.pretrain_step(x.to(self.device), optimizer_pre=self.optimizer_pre, **kwargs)
        #         logger.process_iter_train(d_train)
        #         if i_iter % cfg.print_interval == 0:
        #             print(f'Pretraining_AE: {i_iter}', d_train['loss'])
        #             logger.add_val(i_iter, d_train)
        #         if i_iter % cfg.val_interval == 0:
        #             in_pred = self.predict(model, val_loader, self.device)
        #             ood1_pred = self.predict(model, OOD_val_loader, self.device)
        #             for key, val in in_pred.items():
        #                 auc_val = roc_btw_arr(ood1_pred[key], val)
        #                 print(f'AUC_val({key}): ', auc_val)
        #                 print(f'mean_in: {val.mean()}, mean_ood: {ood1_pred[key].mean()}')
        #                 logger.add_val(i_iter, {f'validation/auc/{key}_': auc_val})
        #                 logger.add_val(i_iter, {f'validation/mean_in/{key}_': val.mean()})
        #                 logger.add_val(i_iter, {f'validation/mean_ood/{key}_': ood1_pred[key].mean()})

        #             in_pred = self.predict(model, test_loader, self.device)
        #             ood1_pred = self.predict(model, OOD_test_loader, self.device)
        #             for key, val in in_pred.items():
        #                 auc_val = roc_btw_arr(ood1_pred[key], val)
        #                 print(f'AUC_test({key}): ', auc_val)
        #                 print(f'mean_in: {val.mean()}, mean_ood: {ood1_pred[key].mean()}')
        #                 logger.add_val(i_iter, {f'test/auc/{key}_': auc_val})
        #                 logger.add_val(i_iter, {f'test/mean_in/{key}_': val.mean()})
        #                 logger.add_val(i_iter, {f'test/mean_ood/{key}_': ood1_pred[key].mean()})
        #         if i_iter % cfg.visualize_interval == 0:
        #             d_val = model.visualization_step(train_loader, procedure = "pretrain", device=self.device)
        #             logger.add_val(i_iter, d_val)                   
        #         i_iter += 1 # added
        
        # torch.save(model.encoder.state_dict(), "pretrained/encoder_ho_1_z_dim_15.pth")
        # torch.save(model.decoder.state_dict(), "pretrained/decoder_ho_1_z_dim_15.pth")
        for i_epoch in range(1, cfg['n_epoch_ebm'] + 1):
            for x, _ in train_loader:
                model.train()
                d_train = model.train_energy_step(x.to(self.device), optimizer_e=self.optimizer_e, **kwargs)
                logger.process_iter_train(d_train)  
                if i_iter % cfg.print_interval == 0:
                    print(f'Pretraining_ebm: {i_iter}', d_train['loss'])
                    logger.add_val(i_iter, d_train)
                   
                if i_iter % cfg.visualize_interval == 0:
                    d_val = model.visualization_step(train_loader, procedure = "train_energy", device=self.device)
                    logger.add_val(i_iter, d_val)
                i_iter += 1
        torch.save(model.ebm.net.fc_nets.state_dict(), "pretrained/ebm_ho_1_z_dim_15.pth")
        # model.ebm.net.fc_nets.load_state_dict(torch.load("pretrained/ebm_conv_layer.pth"))
        import torch.optim as optim
        if not cfg['fix_decoder']:
            self.optimizer_pre = optim.Adam([{'params': model.encoder.parameters(), 'lr': cfg.optimizer['lr_encoder']},
                                           # {'params': model.decoder.parameters(), 'lr':cfg.optimizer['lr_decoder']}
                            ])
            
            if model.train_sigma:
                optimizer = optim.Adam([# 'params': model.encoder.parameters(), 'lr': cfg.optimizer['lr_encoder']},
                                        {'params': model.decoder.parameters(), 'lr':cfg.optimizer['lr_decoder']},
                                        {'params': model.sigma.fc_nets.parameters(), 'lr': cfg.optimizer['lr_sigma']},
                                        {'params': model.ebm.net.fc_nets.parameters(), 'lr': cfg.optimizer['lr_energy']}
                                        ])
            else:
                optimizer = optim.Adam([#{'params': model.encoder.parameters(), 'lr': cfg.optimizer['lr_encoder']},
                                        {'params': model.decoder.parameters(), 'lr':cfg.optimizer['lr_decoder']},
                                        # {'params': model.ebm.parameters(), 'lr': cfg.training.optimizer['lr_energy']}
                                        ])
        else:
            for param in model.decoder.parameters():
                param.requires_grad = False
            for param in model.ebm.net.fc_nets.parameters():
                param.requires_grad = False
            self.optimizer_pre = optim.Adam([{'params': model.encoder.parameters(), 'lr': cfg.optimizer['lr_encoder']}])
            if model.train_sigma:
                optimizer = optim.Adam([# {'params': model.encoder.parameters(), 'lr': cfg.optimizer['lr_encoder']},
                                        {'params': model.sigma.fc_nets.parameters(), 'lr': cfg.optimizer['lr_sigma']}
                                        # {'params': model.ebm.net.fc_nets.parameters(), 'lr': cfg.training.optimizer['lr_energy']}
                                        ])
            else:
                # optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': cfg.optimizer['lr_encoder']},
                #                         # {'params':model.ebm.net.fc_nets.parameters(), 'lr': cfg.training.optimizer['lr_energy']}
                #                         ])
                optimizer = None
            
        self.optimizer = optimizer
        for i_epoch in range(1, cfg['n_epoch'] + 1):
            for x, _ in train_loader:
                model.train()
                start_ts = time.time()

                if model.train_sigma:
                    d_train_t, _ = model.train_step(x.to(self.device), optimizer=self.optimizer, **kwargs)
                    for _ in range(5):
                        neg_x = model.sample(shape = z_shape, sample_step = model.ebm.sample_step,
                                                    device = self.device, replay = model.ebm.replay, apply_noise = True)
                    #d_train_p = model.pretrain_step(x.to(self.device), optimizer_pre=self.optimizer_pre, pretrain =False, **kwargs)
                        d_train_p_neg = model.pretrain_step(neg_x.to(self.device), optimizer_pre=self.optimizer_pre, pretrain =False,neg_sample = True, **kwargs)
                else:
                    d_train_p = model.pretrain_step(x.to(self.device), optimizer_pre=self.optimizer_pre, pretrain =False, **kwargs)
                # d_train_e = model.train_energy_step(x.to(self.device), optimizer_e=self.optimizer_e, pretrain = False, **kwargs)    

                time_meter.update(time.time() - start_ts)
                if model.train_sigma:
                    logger.process_iter_train(d_train_t)
                if i_iter % cfg.print_interval == 0:
                    d_train = logger.summary_train(i_iter)
                    print(
                        f"Epoch [{i_epoch:d}] \nIter [{i_iter:d}]\tAvg Loss: {d_train['loss/train_loss_']:.6f}\tElapsed time: {time_meter.sum:.4f}"
                    )
                    time_meter.reset()
                    logger.add_val(i_iter, d_train)
                    if model.train_sigma:
                        logger.add_val(i_iter, d_train_t)
                        # logger.add_val(i_iter, d_train_p)
                        logger.add_val(i_iter, d_train_p_neg)
                    else:
                        logger.add_val(i_iter,d_train_p)



                model.eval()
                if i_iter % cfg.val_interval == 0:
                    in_pred = self.predict(model, val_loader, self.device)
                    ood1_pred = self.predict(model, OOD_val_loader, self.device)
                    for key, val in in_pred.items():
                        auc_val = roc_btw_arr(ood1_pred[key], val)
                        print(f'AUC_val({key}): ', auc_val)
                        print(f'mean_in: {val.mean()}, mean_ood: {ood1_pred[key].mean()}')
                        logger.add_val(i_iter, {f'validation/auc/{key}_': auc_val})
                        logger.add_val(i_iter, {f'validation/mean_in/{key}_': val.mean()})
                        logger.add_val(i_iter, {f'validation/mean_ood/{key}_': ood1_pred[key].mean()})

                    in_pred = self.predict(model, test_loader, self.device)
                    ood1_pred = self.predict(model, OOD_test_loader, self.device)
                    for key, val in in_pred.items():
                        auc_val = roc_btw_arr(ood1_pred[key], val)
                        print(f'AUC_test({key}): ', auc_val)
                        print(f'mean_in: {val.mean()}, mean_ood: {ood1_pred[key].mean()}')
                        logger.add_val(i_iter, {f'test/auc/{key}_': auc_val})
                        logger.add_val(i_iter, {f'test/mean_in/{key}_': val.mean()})
                        logger.add_val(i_iter, {f'test/mean_ood/{key}_': ood1_pred[key].mean()})
                
                if i_iter % cfg.visualize_interval == 0:
                    d_val = model.visualization_step(train_loader, procedure = "train", device=self.device)
                    logger.add_val(i_iter, d_val)
                i_iter += 1

        self.save_model(model, logdir, i_iter="last")
        return model, best_val_loss

    def save_model(self, model, logdir, best=False, i_iter=None, i_epoch=None):
        if best:
            pkl_name = "model_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = f"model_iter_{i_iter}.pkl"
            else:
                pkl_name = f"model_epoch_{i_epoch}.pkl"
        state = {"epoch": i_epoch, "iter": i_iter, "model_state": model.state_dict()}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f"Model saved: {pkl_name}")


    def predict(self, m, dl, device, flatten=False, pretrain = False):
        """run prediction for the whole dataset"""
        l_result = {"neg_log_prob": [], "recon_error": [], "energy":[], "log_det_jacobian":[]}
        for x, _ in dl:
            with torch.no_grad():
                if flatten:
                    x = x.view(len(x), -1)
                pred = m.neg_log_prob(x.cuda(device), pretrain=pretrain)

            for key, val in pred.items():
                l_result[key].append(val.detach().cpu())
        for key, val in l_result.items():
            l_result[key] = torch.cat(val)
        return l_result