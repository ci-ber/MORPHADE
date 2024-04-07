import logging
#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
#
from optim.losses import MedicalNetPerceptualSimilarity

from torch.nn import  MSELoss
#
from dl_utils.visualization import plot_warped_grid,save_warped_grid
from optim.losses import PerceptualLoss

from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as ssim2

import nibabel as nib
#
from optim.metrics.jacobian import jacobian_determinant,MidpointNormalize

#
from dl_utils import *
from optim.metrics import *
from optim.losses.image_losses import NCC
from core.DownstreamEvaluator import DownstreamEvaluator

from transforms.synthetic import *


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_= True):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = MSELoss().to(self.device)
        self.criterion_MSE = MSELoss().to(self.device)
        self.compute_scores = True
        self.l_ncc = NCC(win=[9, 9])
        self.experiment="b001_from_scratch"
        self.criterion_PL = PerceptualLoss(device=device)

        self.global_= True

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """
        self.test_alzheimer(global_model)


    def test_alzheimer(self, global_model):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.model.load_state_dict(global_model)
        self.model.eval()
        task='Test_Alzheimer'
        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
        }
        loss_mse_array=[]
        loss_mse_array=np.array(loss_mse_array)
        loss_rec_array=[]
        loss_rec_array=np.array(loss_rec_array)
        loss_l1_array=[]
        loss_l1_array=np.array(loss_l1_array)
        loss_ssim_array=[]
        loss_ssim_array=np.array(loss_ssim_array) 

        kk=0
        for dataset_key in self.test_data_dict.keys():

            dataset = self.test_data_dict[dataset_key]
            test_total=0
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                x_rec, x_rec_dict = self.model(x)
                if len(x.shape)==4:
                    b, c, h, w = x.shape
                else:
                    b, c, h, w,d = x.shape

                test_total += b
                x_ = x_rec_dict['x_prior']
                deformation = x_rec_dict['deformation']
                loss_rec = self.criterion_rec(x_, x)
                loss_mse = self.criterion_MSE(x_rec, x)
                self.criterion_PL = MedicalNetPerceptualSimilarity(device=device)
                loss_pl = self.criterion_PL(x_, x)
                loss_mse_array= np.append(loss_mse_array,loss_mse.detach().cpu().numpy())
                loss_rec_array= np.append(loss_rec_array,loss_rec.detach().cpu().numpy())
                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)

                _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation.cpu().detach().numpy(),x_rec.cpu().detach().numpy())

                ssimm=ssim2(x_rec, x, data_range=1.)
                loss=np.abs(x_rec.detach().cpu()[0].numpy()-x.detach().cpu()[0].numpy())
                loss_l1_array= np.append(loss_l1_array,loss.mean())
                loss_ssim_array= np.append(loss_ssim_array,ssimm.detach().cpu().numpy())
                loss_pl = self.criterion_PL(x_rec, x).mean().detach().cpu().numpy()

                np.save("./results/"+self.experiment+"/alzheimer_loss"+str(idx)+".npy", loss, allow_pickle=True, fix_imports=True)
                np.save("./results/"+self.experiment+"/alzheimer_jacdet"+str(idx)+".npy", jacdet, allow_pickle=True, fix_imports=True)

                w=128
                #change manually
                self.experiment="b001_from_scratch"
                if idx%1==0:
                    rec_nifti = nib.Nifti1Image(np.squeeze(x_rec.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(rec_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_rec.nii.gz')
                    gl_prior_nifti = nib.Nifti1Image(np.squeeze(x_.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(gl_prior_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_gl_prior.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(x.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(x_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_img.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(deformation.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(x_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_deff.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(jacdet) , np.eye(4))
                    nib.save(x_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_jacdet.nii.gz')

                if idx<=20 or idx%260==0:
                    if len(x.shape)==4:
                        global_prior = x_.detach().cpu()[0].numpy()
                        rec = x_rec.detach().cpu()[0].numpy()
                        img = x.detach().cpu()[0].numpy()
                        deff = deformation.detach().cpu()[0].numpy()

                        elements = [img, global_prior, rec, np.abs(global_prior-img), np.abs(rec-img),deff]
                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for i in range(len(axarr)):
                            axarr[i].axis('off')
                            if i!=len(axarr)-1:
                                v_max = 1 if i < np.floor(((len(elements)-1) / 2)) + 1 else 0.5
                                c_map = 'gray' if i < np.floor(((len(elements)-1) / 2)) + 1 else 'inferno'               
                                axarr[i].imshow(np.squeeze(elements[i].transpose(1, 2, 0)), vmin=0, vmax=v_max, cmap=c_map)
                            else:
                                plot_warped_grid(ax=axarr[i],disp=deff)
                    else:
                        global_prior = x_.detach().cpu()[0].numpy()
                        rec = x_rec.detach().cpu()[0].numpy()
                        img = x.detach().cpu()[0].numpy()
                        deff = deformation[0,:,:,:,:].detach().cpu().numpy()
            
                    
                    elements = [img,global_prior, rec,np.abs(global_prior-img), np.abs(rec - img),loss*np.abs(jacdet),jacdet,deff]
                    v_maxs = [1, 1, 1,0.5,0.5,0.5,0.5,0.5]
                    diffp, axarr = plt.subplots(3, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                    diffp.set_size_inches(len(elements) * 4, 3 * 4)
                    for i in range(len(elements)):
                        for axis in range(3):
                            if i<=len(elements)-2:
                                axarr[axis, i].axis('off')
                                v_max = v_maxs[i]
                                c_map = 'gray' if v_max == 1 else 'plasma'
                                # print(elements[i].shape)
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
                                    if idx%260==0:
                                        save_warped_grid(disp=temp,save= self.experiment,idx=idx)
                                        save_warped_grid(disp=temp,save= self.experiment,idx=idx,interval=2)
                                elif axis == 1:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, :, int(w / 2),:],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :, int(w / 2),:],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp)
                                    if idx%260==0:
                                        save_warped_grid(disp=temp,save= self.experiment,idx=idx)
                                        save_warped_grid(disp=temp,save= self.experiment,idx=idx,interval=2)
                                else:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,1, :,:, int(w / 2)],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :,:, int(w / 2)],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp)
                                    if idx%260==0:
                                        save_warped_grid(disp=temp,save= self.experiment,idx=idx,interval=1)
                                        save_warped_grid(disp=temp,save= self.experiment,idx=idx,interval=2)
                            elif i==len(elements)-2:    
                                axarr[axis, i].axis('off')
                                v_max = v_maxs[i]
                                c_map = 'bwr'
                                # print(elements[i].shape)
                                if axis == 0:
                                    el = np.squeeze(elements[i])[int(w / 2), :, :]
                                elif axis == 1:
                                    el = np.squeeze(elements[i])[:, int(w / 2), :]
                                else:
                                    el = np.squeeze(elements[i])[:, :, int(w / 2)] 
                                axarr[axis, i].imshow(np.squeeze(el).T, cmap=c_map, norm = MidpointNormalize(midpoint=0))
                    wandb.log({task + '/Example_': [
                            wandb.Image(diffp, caption="Test_Alzheimer_" + str(idx))]})
                    

        fig, ax = plt.subplots()
        ax.hist(loss_mse_array, bins=len(loss_mse_array)) 
        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': idx})
        print("Alzheimer MAE ALL:",loss_l1_array.mean())   
        print("Alzheimer SSIM ALL:",loss_ssim_array.mean()) 


    def test_healthy(self, global_model):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.model.load_state_dict(global_model)
        self.model.eval()
        task='Test_Healthy'
        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
        }
        loss_mse_array=[]
        loss_mse_array=np.array(loss_mse_array)
        loss_rec_array=[]
        loss_rec_array=np.array(loss_rec_array)
        loss_l1_array=[]
        loss_l1_array=np.array(loss_l1_array)
        loss_ssim_array=[]
        loss_ssim_array=np.array(loss_ssim_array) 
        kk=0
        for dataset_key in self.test_data_dict.keys():

            dataset = self.test_data_dict[dataset_key]
            test_total=0
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                x_rec, x_rec_dict = self.model(x)
                if len(x.shape)==4:
                    b, c, h, w = x.shape
                else:
                    b, c, h, w,d = x.shape

                test_total += b
                x_ = x_rec_dict['x_prior']
                deformation = x_rec_dict['deformation']
                loss_rec = self.criterion_rec(x_, x)
                loss_mse = self.criterion_MSE(x_rec, x)
                self.criterion_PL = MedicalNetPerceptualSimilarity(device=device)
                loss_pl = self.criterion_PL(x_, x)
                if loss_mse.detach().cpu().numpy()<0:
                    print("loss_mse is smalelr than 0")
                    loss_mse=0
                loss_mse_array= np.append(loss_mse_array,loss_mse.detach().cpu().numpy())
                loss_rec_array= np.append(loss_rec_array,loss_rec.detach().cpu().numpy())
                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)
                _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation.cpu().detach().numpy(),x_rec.cpu().detach().numpy())

                ssimm=ssim2(x_rec, x, data_range=1.)
                loss=np.abs(x_rec.detach().cpu()[0].numpy()-x.detach().cpu()[0].numpy())
                loss_l1_array= np.append(loss_l1_array,loss.mean())
                loss_2=np.abs(x_.detach().cpu()[0].numpy()-x.detach().cpu()[0].numpy())
                loss_l1_prior_array= np.append(loss_l1_prior_array,loss_2.mean())
                loss_ssim_array= np.append(loss_ssim_array,ssimm.detach().cpu().numpy())
                ssimm2=ssim2(x_, x, data_range=1.)

                loss_ssim_prior_array= np.append(loss_ssim_prior_array,ssimm2.detach().cpu().numpy())
                loss_pl = self.criterion_PL(x_rec, x).mean().detach().cpu().numpy()

                np.save("./results/"+self.experiment+"/healthy_loss"+str(idx)+".npy", loss, allow_pickle=True, fix_imports=True)
                np.save("./results/"+self.experiment+"/healthy_jacdet"+str(idx)+".npy", jacdet, allow_pickle=True, fix_imports=True)

                #change manually
                experiment="b001_from_scratch"
                if idx%1==0:
                    rec_nifti = nib.Nifti1Image(np.squeeze(x_rec.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(rec_nifti, './results/'+experiment+'/test_healthy_'+str(idx)+'_rec.nii.gz')
                    gl_prior_nifti = nib.Nifti1Image(np.squeeze( x_.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(gl_prior_nifti, './results/'+experiment+'/test_healthy_'+str(idx)+'_gl_prior.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(x.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(x_nifti, './results/'+experiment+'/test_healthy_'+str(idx)+'_img.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(deformation.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(x_nifti, './results/'+experiment+'/test_healthy_'+str(idx)+'_deff.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(jacdet) , np.eye(4))
                    nib.save(x_nifti, './results/'+experiment+'/test_healthy_'+str(idx)+'_jacdet.nii.gz')

                if idx<=20:
                    if len(x.shape)==4:
                        global_prior = x_.detach().cpu()[0].numpy()
                        rec = x_rec.detach().cpu()[0].numpy()
                        img = x.detach().cpu()[0].numpy()

                        deff = deformation.detach().cpu()[0].numpy()

                        elements = [img, global_prior, rec, np.abs(global_prior-img), np.abs(rec-img),deff]
                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for i in range(len(axarr)):
                            axarr[i].axis('off')
                            if i!=len(axarr)-1:
                                v_max = 1 if i < np.floor(((len(elements)-1) / 2)) + 1 else 0.5
                                c_map = 'gray' if i < np.floor(((len(elements)-1) / 2)) + 1 else 'inferno'               
                                axarr[i].imshow(np.squeeze(elements[i].transpose(1, 2, 0)), vmin=0, vmax=v_max, cmap=c_map)
                            else:
                                plot_warped_grid(ax=axarr[i],disp=deff)
                    else:
                        global_prior = x_.detach().cpu()[0].numpy()
                        rec = x_rec.detach().cpu()[0].numpy()
                        img = x.detach().cpu()[0].numpy()
                        deff = deformation[0,:,:,:,:].detach().cpu().numpy()            
            
                    elements = [img,global_prior, rec,np.abs(global_prior-img), np.abs(rec - img),loss*np.abs(jacdet),jacdet,deff]
                    v_maxs = [1, 1, 1,0.5,0.5,0.5,0.5,0.5]
                    diffp, axarr = plt.subplots(3, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                    diffp.set_size_inches(len(elements) * 4, 3 * 4)
                    for i in range(len(elements)):
                        for axis in range(3):
                            if i<=len(elements)-2:
                                axarr[axis, i].axis('off')
                                v_max = v_maxs[i]
                                c_map = 'gray' if v_max == 1 else 'plasma'
                                # print(elements[i].shape)
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
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp,save= './results/'+self.experiment+'/test_healthy_'+str(idx)+'deff3.png') # .rot90(axes=(2,3)
                                elif axis == 1:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, :, int(w / 2),:],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :, int(w / 2),:],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp,save= './results/'+self.experiment+'/test_healthy_'+str(idx)+'deff2.png')
                                else:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,1, :,:, int(w / 2)],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :,:, int(w / 2)],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp,save= './results/'+self.experiment+'/test_healthy_'+str(idx)+'deff1.png')
                            elif i==len(elements)-2:    
                                axarr[axis, i].axis('off')
                                v_max = v_maxs[i]
                                c_map = 'bwr'
                                # print(elements[i].shape)
                                if axis == 0:
                                    el = np.squeeze(elements[i])[int(w / 2), :, :]
                                elif axis == 1:
                                    el = np.squeeze(elements[i])[:, int(w / 2), :]
                                else:
                                    el = np.squeeze(elements[i])[:, :, int(w / 2)] 
                                axarr[axis, i].imshow(np.squeeze(el).T, cmap=c_map, norm = MidpointNormalize(midpoint=0))
                    wandb.log({task + '/Example_': [
                            wandb.Image(diffp, caption="Test_" + str(idx))]})
                    

        fig, ax = plt.subplots()
        ax.hist(loss_mse_array, bins=len(loss_mse_array)) 
#        wandb.log({"Test_healthy_histogram": fig})
        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': idx})
        print("Healthy MAE ALL:",loss_l1_array.mean())  
        print("Healthy MAE prior ALL:",loss_l1_prior_array.mean())   
        print("Healthy SSIM ALL:",loss_ssim_array.mean()) 
        print("Healthy SSIM prior ALL:",loss_ssim_prior_array.mean()) 


        plot_roc_curve("final",False)
  