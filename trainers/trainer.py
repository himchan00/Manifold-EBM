import os
import time
import torch.optim as optim
import numpy as np
import torch
from metrics import averageMeter
from utils.utils import roc_btw_arr

class BaseTrainer:
    """Trainer for a conventional iterative training of model for classification"""
    def __init__(self, optimizer, training_cfg, device):
        self.training_cfg = training_cfg
        self.device = device
        self.optimizer = optimizer

    def train(self, model, d_dataloaders, logger=None, logdir=""):
        cfg = self.training_cfg
        time_meter = averageMeter()
        train_loader, val_loader, test_loader = (d_dataloaders["training"], d_dataloaders["validation"], d_dataloaders["test"])
        OOD_val_loader, OOD_test_loader = (d_dataloaders["OOD_validation"], d_dataloaders["OOD_test"])
        kwargs = {'dataset_size': len(train_loader.dataset)}
        i_iter = 0
        best_val_loss = np.inf

        # model.encoder.load_state_dict(torch.load(f"pretrained/encoder_ho_1_unnorm.pth"))
        # model.decoder.load_state_dict(torch.load(f"pretrained/decoder_ho_1_unnorm.pth"))
        # model.minimizer.load_state_dict(torch.load(f"pretrained/minimizer_ho_1_lr_1e-4_original.pth"))
        sigma_params = []
        # sigma_name_list = ["fc4_s.weight", "fc5_s.weight", "fc6_s.weight", "fc4_s.bias", "fc5_s.bias", "fc6_s.bias"]
        sigma_name_list = ["conv4_s.weight", "conv5_s.weight", "conv6_s.weight", "conv4_s.bias", "conv5_s.bias", "conv6_s.bias"]
        for name, param in model.encoder_target.named_parameters():
            if name in sigma_name_list:
                print(f"{name} added to sigma_params")
                sigma_params.append(param)

        optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': cfg.optimizer['lr_encoder']},
                        {'params': model.decoder.parameters(), 'lr':cfg.optimizer['lr_decoder']},
                        {'params': sigma_params, 'lr':cfg.optimizer['lr_sigma']},
                        ])

        for i_epoch in range(1, cfg['n_epoch'] + 1):
            for x, _ in train_loader:
                model.train()
                start_ts = time.time()

                d_train_t = model.train_step(x.to(self.device), optimizer=optimizer, **kwargs)

                # update target encoder
                tau = 0.005
                for (name, param), param_target in zip(model.encoder.named_parameters(), model.encoder_target.parameters()):
                    if name in sigma_name_list:
                        pass
                    else:
                        param_target.data = param_target.data * (1.0 - tau) + param.data * tau

                    

                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train_t)
                if i_iter % cfg.print_interval == 0:
                    d_train = logger.summary_train(i_iter)
                    print(
                        f"Epoch [{i_epoch:d}] \nIter [{i_iter:d}]\tAvg Loss: {d_train['loss/train_loss_']:.6f}\tElapsed time: {time_meter.sum:.4f}"
                    )
                    time_meter.reset()
                    logger.add_val(i_iter, d_train)
                    logger.add_val(i_iter, d_train_t)


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

                    # in_pred = self.predict(model, test_loader, self.device)
                    # ood1_pred = self.predict(model, OOD_test_loader, self.device)
                    # for key, val in in_pred.items():
                    #     auc_val = roc_btw_arr(ood1_pred[key], val)
                    #     print(f'AUC_test({key}): ', auc_val)
                    #     print(f'mean_in: {val.mean()}, mean_ood: {ood1_pred[key].mean()}')
                    #     logger.add_val(i_iter, {f'test/auc/{key}_': auc_val})
                    #     logger.add_val(i_iter, {f'test/mean_in/{key}_': val.mean()})
                    #     logger.add_val(i_iter, {f'test/mean_ood/{key}_': ood1_pred[key].mean()})
                
                if i_iter % cfg.visualize_interval == 0:
                    d_val = model.visualization_step(train_loader, procedure = "train", device=self.device)
                    logger.add_val(i_iter, d_val)
                i_iter += 1

        # torch.save(model.encoder.state_dict(), f"pretrained/encoder_ho_1_unnorm.pth")
        # torch.save(model.decoder.state_dict(), f"pretrained/decoder_ho_1_unnorm.pth")

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
        l_result = {}
        for x, _ in dl:
            with torch.no_grad():
                if flatten:
                    x = x.view(len(x), -1)
                pred = m.neg_log_prob(x.cuda(device), n_eval = 10, train = False)

            for key, val in pred.items():
                if key in l_result:
                    l_result[key].append(val.detach().cpu())
                else:
                    l_result[key] = [val.detach().cpu()]
        for key, val in l_result.items():
            l_result[key] = torch.cat(val)
        return l_result