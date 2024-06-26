from core.Trainer import Trainer
from time import time
import wandb
import logging
from optim.losses.image_losses import *
from optim.losses.NCC2 import *
import matplotlib.pyplot as plt
import copy
from optim.losses.monai_perceptual_loss import PerceptualLoss2
from optim.losses import PerceptualLoss,MedicalNetPerceptualSimilarity
from optim.metrics.jacobian import jacobian_determinant,MidpointNormalize
from optim.losses.adversarial_loss import PatchAdversarialLoss
from monai.networks.layers import Act
from model_zoo.patchgan_discriminator import PatchDiscriminator
from dl_utils.visualization import plot_warped_grid
import torchvision.transforms as transforms
import nibabel as nib

class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)
        self.ncc_loss = NCC2()
        self.deform_R = DisplacementRegularizer('gradient-l2')
        self.beta_max = training_params['beta'] if 'beta' in training_params.keys() else 3
        self.delta = training_params['delta'] if 'delta' in training_params.keys() else 1
        self.gamma = training_params['gamma'] if 'gamma' in training_params.keys() else 1

        self.max_iter = training_params['max_iter'] if 'max_iter' in training_params.keys() else 500
        self.adv_loss=PatchAdversarialLoss(criterion="least_squares")

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Train local client
        :param model_state: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :param start_epoch: int
            start epoch
        :return:
            self.model.state_dict():
        """
        discriminator = PatchDiscriminator(
            spatial_dims=3,
            num_layers_d=3,
            num_channels=32,
            in_channels=1,
            out_channels=1,
            kernel_size=4,
            activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
            norm="BATCH",
            bias=False,
            padding=1,
        )
        discriminator.to(self.device)
        optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4) #################
        self.autoencoder_warm_up_n_epochs = 0
        self.disc_tolerance = 1.5

        if model_state is not None:
            self.model.load_state_dict(model_state)
            # to keep training from a saved checkpoint
            checkpoint = torch.load('./weights/last/discriminator.pt', map_location=torch.device(self.device))
            discriminator.load_state_dict(checkpoint['model_weights'])
            optimizer_d.load_state_dict(checkpoint['optimizer_weights'])
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer
        epoch_jacobian= []
        epoch_losses = []
        epoch_losses_after_deformation = []
        epoch_losses_reg = []
        epoch_losses_deformation = []
        epoch_disc_losses = []
        epoch_adv_losses = []

        self.early_stop = False
        
        for epoch in range(self.training_params['nr_epochs']):
            if self.early_stop:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_perc_neg_jac_det,batch_adv_loss,batch_disc_loss,batch_loss,batch_loss_after_deformation,batch_loss_reg, batch_loss_deform, count_images = 0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0
            self.beta = np.clip(self.beta_max * (epoch / self.max_iter), 1e-3, self.beta_max)
            discriminator.train()
            for data in self.train_ds:            

                images = data[0].to(self.device)
                transformed_images = self.transform(images) if self.transform is not None else images
                b, c, w, h,d= images.shape

                count_images += b
                # Forward Pass
                reconstruction, result_dict = self.model(transformed_images, registration=False)

                global_prior = result_dict['x_prior']
                reversed_img = result_dict['x_reversed']
                deformation = result_dict['deformation']
                logits_fake = discriminator(global_prior.contiguous().float())[-1]

                # Losses
                loss_rec = self.criterion_rec(global_prior, images, result_dict)
                loss_rec_after_deformation = self.criterion_rec(reconstruction, images, result_dict)
                reg_deform = self.deform_R(deformation)
                loss_deform = self.ncc_loss(images, reconstruction) if torch.equal(reconstruction, reversed_img) \
                    else (self.ncc_loss(images, reconstruction) + self.ncc_loss(global_prior, reversed_img))/2
                loss_adv=self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

                loss = loss_rec  + self.gamma * loss_adv
                if epoch > 10:
                   loss = loss_rec + self.delta * loss_deform + self.beta * reg_deform + self.gamma * loss_adv

                self.optimizer.zero_grad()
                # Backward Pass
                loss.backward()
                self.optimizer.step()
                
              #  Discriminator part
                

                logits_fake = discriminator(global_prior.contiguous().detach())[-1]
                loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d =  self.gamma * discriminator_loss ##############
                if epoch > self.autoencoder_warm_up_n_epochs and epoch_adv_loss < self.disc_tolerance:
                    optimizer_d.zero_grad(set_to_none=True)
                    loss_d.backward()
                    optimizer_d.step()

                _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation.cpu().detach().numpy(),reconstruction.cpu().detach().numpy())

                batch_perc_neg_jac_det+=perc_neg_jac_det* images.size(0)

                batch_loss += loss_rec.item() * images.size(0)
                batch_loss_after_deformation += loss_rec_after_deformation.item() * images.size(0)
                batch_loss_reg += reg_deform.item() * images.size(0)
                batch_loss_deform += loss_deform.item() * images.size(0)
                batch_adv_loss += loss_adv.item() * images.size(0)
                batch_disc_loss += loss_d.item() * images.size(0)

            torch.cuda.empty_cache()
            epoch_perc_neg_jac_det = (batch_perc_neg_jac_det) / count_images if count_images > 0 else batch_perc_neg_jac_det
            epoch_loss = (batch_loss) / count_images if count_images > 0 else batch_loss
            epoch_loss_after_deformation =( batch_loss_after_deformation) / count_images if count_images > 0 else batch_loss
            epoch_loss_reg = (batch_loss_reg) / count_images if count_images > 0 else batch_loss_reg
            epoch_loss_deformation = (batch_loss_deform) / count_images if count_images > 0 else batch_loss_deform
            epoch_adv_loss = (batch_adv_loss) / count_images if count_images > 0 else batch_adv_loss
            epoch_disc_loss = (batch_disc_loss) / count_images if count_images > 0 else batch_disc_loss

            epoch_jacobian.append(epoch_perc_neg_jac_det)
            epoch_losses.append(epoch_loss)
            epoch_losses_after_deformation.append(epoch_loss_after_deformation)
            epoch_losses_reg.append(epoch_loss_reg)
            epoch_losses_deformation.append(epoch_loss_deformation)           
            epoch_disc_losses.append(epoch_disc_loss)
            epoch_adv_losses.append(epoch_adv_loss)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})
            wandb.log({"Train/Loss_After_Deformation": epoch_loss_after_deformation, '_step_': epoch})
            wandb.log({"Train/Loss_Reg_": epoch_loss_reg, '_step_': epoch})
            wandb.log({"Train/Loss_Deformation_": epoch_loss_deformation, '_step_': epoch})
            wandb.log({"Train/Loss_Adv_": epoch_adv_loss, '_step_': epoch})
            wandb.log({"Train/Loss_Discriminator_": epoch_disc_loss, '_step_': epoch})
            wandb.log({"Train/perc_neg_jac_det_": epoch_perc_neg_jac_det, '_step_': epoch})
            # Save latest model
            torch.save({'model_weights': self.model.state_dict(),
                        'optimizer_weights': self.optimizer.state_dict(),
                        'epoch': epoch}, self.client_path + '/latest_model.pt')
            img = transformed_images[0].cpu().detach().numpy()
            rec_ = reconstruction[0].cpu().detach().numpy()
            gl_prior = global_prior[0].cpu().detach().numpy()
            deff = deformation[0,:,:,:,:].detach().cpu().numpy()
            elements = [img,gl_prior, rec_,np.abs(gl_prior-img), np.abs(rec_ - img),jacdet,deff]
            v_maxs = [1, 1, 1,0.5,0.5,0.5,0.5]
            diffp, axarr = plt.subplots(3, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
            diffp.set_size_inches(len(elements) * 4, 3 * 4)
            for i in range(len(elements)):
                for axis in range(3):
                    if i<=len(elements)-2:
                        axarr[axis, i].axis('off')
                        v_max = v_maxs[i]
                        c_map = 'gray' if v_max == 1 else 'plasma'
                        if axis == 0:
                            el = np.squeeze(elements[i])[int(w / 2), :, :]
                        elif axis == 1:
                            el = np.squeeze(elements[i])[:, int(w / 2), :]
                        else:
                            el = np.squeeze(elements[i])[:, :, int(w / 2)]

                        axarr[axis, i].imshow(np.squeeze(el).T, vmin=0, vmax=v_max, cmap=c_map, origin='lower')
                        
                    elif i==len(elements)-1:
                        
                        if axis == 0:
                            temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, int(w / 2),:,:],axes=(1,2)), np.rot90(elements[i][np.newaxis,1, int(w / 2),:,:],axes=(1,2))), 0)
                            plot_warped_grid(ax=axarr[axis, i],disp=temp) # .rot90(axes=(2,3)
                        elif axis == 1:
                            temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, :, int(w / 2),:],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :, int(w / 2),:],axes=(1,2))), 0)
                            plot_warped_grid(ax=axarr[axis, i],disp=temp)
                        else:
                            temp=np.concatenate((np.rot90(elements[i][np.newaxis,1, :,:, int(w / 2)],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :,:, int(w / 2)],axes=(1,2))), 0)
                            plot_warped_grid(ax=axarr[axis, i],disp=temp)
                    elif i==len(elements)-2:    
                        axarr[axis, i].axis('off')
                        v_max = v_maxs[i]
                        c_map = 'bwr'
                        if axis == 0:
                            el = np.squeeze(elements[i])[int(w / 2), :, :]
                        elif axis == 1:
                            el = np.squeeze(elements[i])[:, int(w / 2), :]
                        else:
                            el = np.squeeze(elements[i])[:, :, int(w / 2)] 
                        axarr[axis, i].imshow(np.squeeze(el).T, cmap=c_map, norm = MidpointNormalize(midpoint=0))
     
            wandb.log({'Train/Example_': [
                wandb.Image(diffp, caption="Iteration_" + str(epoch))]})
            torch.save({'model_weights': discriminator.state_dict(),
            'optimizer_weights': optimizer_d.state_dict(),
            'epoch': epoch}, self.client_path + '/discriminator.pt')
            # Run validation
            self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        return self.model.state_dict(), self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """
        torch.cuda.empty_cache()
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_after_deformation': 0,
            task + '_loss_pl': 0,
        }
        loss_mse_array=[]
        loss_mse_array=np.array(loss_mse_array)
        test_total = 0
        with torch.no_grad():
            for data in test_data:
                x = data[0]
                b, c, h, w,d = x.shape
                test_total += b
                x = x.to(self.device)

                # Forward pass
                x_rec, rec_dict = self.test_model(x)
                x_ = rec_dict['x_prior']
                deformation = rec_dict['deformation']
                loss_rec = self.criterion_rec(x_, x, rec_dict)

                loss_mse = self.criterion_rec(x_rec, x)
          
                loss_mse_array= np.append(loss_mse_array,loss_mse.detach().cpu().numpy())
                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_after_deformation'] += loss_mse.item() * x.size(0)

                if test_total<=20:
                    img = x.detach().cpu()[0].numpy()
                    rec_ = x_rec.detach().cpu()[0].numpy()
                    gl_prior = x_.detach().cpu()[0].numpy()
                    deff = deformation[0,:,:,:,:].detach().cpu().numpy()
                    _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation.cpu().detach().numpy(),x_rec.detach().cpu().numpy())        
                    elements = [img,gl_prior, rec_,np.abs(gl_prior-img), np.abs(rec_ - img),jacdet,deff]
                    v_maxs = [1, 1, 1,0.5,0.5,0.5,0.5]
                    diffp, axarr = plt.subplots(3, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                    diffp.set_size_inches(len(elements) * 4, 3 * 4)
                    for i in range(len(elements)):
                        for axis in range(3):
                            if i<=len(elements)-2:
                                axarr[axis, i].axis('off')
                                v_max = v_maxs[i]
                                c_map = 'gray' if v_max == 1 else 'plasma'
                                if axis == 0:
                                    el = np.squeeze(elements[i])[int(w / 2), :, :]
                                elif axis == 1:
                                    el = np.squeeze(elements[i])[:, int(w / 2), :]
                                else:
                                    el = np.squeeze(elements[i])[:, :, int(w / 2)]

                                axarr[axis, i].imshow(np.squeeze(el).T, vmin=0, vmax=v_max, cmap=c_map, origin='lower')
                                
                            elif i==len(elements)-1:
                                
                                if axis == 0:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, int(w / 2),:,:],axes=(1,2)), np.rot90(elements[i][np.newaxis,1, int(w / 2),:,:],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp) # .rot90(axes=(2,3)
                                elif axis == 1:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, :, int(w / 2),:],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :, int(w / 2),:],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp)
                                else:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,1, :,:, int(w / 2)],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :,:, int(w / 2)],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp)
                            elif i==len(elements)-2:    
                                axarr[axis, i].axis('off')
                                v_max = v_maxs[i]
                                c_map = 'bwr'
                                if axis == 0:
                                    el = np.squeeze(elements[i])[int(w / 2), :, :]
                                elif axis == 1:
                                    el = np.squeeze(elements[i])[:, int(w / 2), :]
                                else:
                                    el = np.squeeze(elements[i])[:, :, int(w / 2)] 
                                axarr[axis, i].imshow(np.squeeze(el).T, cmap=c_map, norm = MidpointNormalize(midpoint=0))
                    wandb.log({task + '/Example_': [
                            wandb.Image(diffp, caption="Iteration_" + str(epoch)+"_"+str(test_total))]})

        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        wandb.log({'beta': self.beta, '_step_': epoch})

        epoch_val_loss = (metrics[task + '_loss_rec'])/ test_total
        if task == 'Val':
            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                self.best_weights = copy.deepcopy(model_weights)
                self.best_opt_weights = copy.deepcopy(opt_weights)
                torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights,
                            'epoch': epoch}, self.client_path + '/best_model.pt')
            self.early_stop = self.early_stopping(epoch_val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_val_loss)

