from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay,RocCurveDisplay
import numpy as np
from scipy.ndimage import gaussian_filter   

 

def plot_roc_curve(experiment:None,mean_std:None):

#Alzheimer
    alzheimer_jacdet_array=[]
    alzheimer_jacdet_array=np.array(alzheimer_jacdet_array)
    alzheimer_loss_array=[]
    alzheimer_loss_array=np.array(alzheimer_loss_array)       
    alzheimer_loss_jacdet_array=[]
    alzheimer_loss_jacdet_array=np.array(alzheimer_loss_jacdet_array)
    alzheimer_loss_logjacdet_array=[]
    alzheimer_loss_logjacdet_array=np.array(alzheimer_loss_logjacdet_array)


    healthy_jacdet_array=[]
    healthy_jacdet_array=np.array(healthy_jacdet_array)
    healthy_loss_array=[]
    healthy_loss_array=np.array(healthy_loss_array)  
    healthy_loss_jacdet_array=[]
    healthy_loss_jacdet_array=np.array(healthy_loss_jacdet_array)
    healthy_loss_logjacdet_array=[]
    healthy_loss_logjacdet_array=np.array(healthy_loss_logjacdet_array)

    for index in range(0,200):
                loss= np.load("./results/"+"constrained_deformer_b1"+"/"+"alzheimer_loss"+str(index)+".npy")
                jacdet= np.load("./results/"+"b001_from_scratch"+"/"+"alzheimer_jacdet"+str(index)+".npy")

                jacdet=jacdet
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)
                
                jacdet2=jacdet
                jacdet2[jacdet2>0]=0
                jacdet2=np.abs(jacdet2)
                jacdet2=gaussian_filter(jacdet2, sigma=1)

                data1=(loss*np.abs(jacdet2)).mean()
                data2=(loss*np.abs(log_jac_det)).mean()

                alzheimer_loss_array=np.append(alzheimer_loss_array,loss.mean())               
                alzheimer_jacdet_array=np.append(alzheimer_jacdet_array,log_jac_det.std())

                alzheimer_loss_jacdet_array=np.append(alzheimer_loss_jacdet_array,data1)
                alzheimer_loss_logjacdet_array=np.append(alzheimer_loss_logjacdet_array,data2)


    for index in range(0,200):
                loss= np.load("./results/"+"constrained_deformer_b1"+"/"+"healthy_loss"+str(index)+".npy")
                jacdet= np.load("./results/"+"b001_from_scratch"+"/"+"healthy_jacdet"+str(index)+".npy")
                
                jacdet=jacdet
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)

                jacdet2=jacdet
                jacdet2[jacdet2>0]=0
                jacdet2=np.abs(jacdet2)
                jacdet2=gaussian_filter(jacdet2, sigma=1)

                data1=(loss*np.abs(jacdet2)).mean()
                data2=(loss*np.abs(log_jac_det)).mean()
                
                healthy_loss_array=np.append(healthy_loss_array,loss.mean())               
                healthy_jacdet_array=np.append(healthy_jacdet_array,log_jac_det.std())               
                healthy_loss_jacdet_array=np.append(healthy_loss_jacdet_array,data1)
                healthy_loss_logjacdet_array=np.append(healthy_loss_logjacdet_array,data2)

    y_morphed_healthy=np.zeros(len(healthy_loss_jacdet_array))
    y_morphed_alzheimer=np.ones(len(alzheimer_loss_jacdet_array))

    y_morphed=np.concatenate((y_morphed_healthy,y_morphed_alzheimer))

    x_loss=np.concatenate((healthy_loss_array,alzheimer_loss_array))
    x_logjacdet=np.concatenate((healthy_jacdet_array,alzheimer_jacdet_array))
    x_loss_jacdet=np.concatenate((healthy_loss_jacdet_array,alzheimer_loss_jacdet_array))
    x_loss_logjacdet=np.concatenate((healthy_loss_logjacdet_array,alzheimer_loss_logjacdet_array))

    fig, ax = plt.subplots(figsize=(6, 6))

    colors = ["aqua", "darkorange","black","yellow","red","gray"]
    print(x_logjacdet)
    
    RocCurveDisplay.from_predictions(
            y_morphed,
            x_loss_jacdet,
            name=f"ROC curve for Loss*Jacdet",
             color=colors[3],
            ax=ax,
        )

    RocCurveDisplay.from_predictions(
            y_morphed,
            x_loss_logjacdet,
            name=f"ROC curve for Loss*LogJacdet",
             color=colors[5],
            ax=ax,
        ),
    
    RocCurveDisplay.from_predictions(
            y_morphed,
            x_loss,
            name=f"ROC curve for Loss",
             color=colors[3],
            ax=ax,
        )

    RocCurveDisplay.from_predictions(
            y_morphed,
            x_logjacdet,
            name=f"ROC curve for LogJacdet",
             color=colors[5],
            ax=ax,
        ),

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Alzheimer vs Healthy")
    plt.legend()
    plt.show()
    plt.savefig("./results/"+"final"+"/roc_curve_ad_vs_healthy.png")

    bins = np.linspace(0, 0.001, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.hist(healthy_loss_jacdet_array, bins, alpha=0.5, label='healty')
    plt.hist(alzheimer_loss_jacdet_array, bins, alpha=0.5, label='alzheimer')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig("./results/"+"final"+"/distributions_b001.png")
    np.save("./results/"+"final"+"/alzheimer_loss_jacdet_array_for_correlation.npy", alzheimer_loss_jacdet_array, allow_pickle=True, fix_imports=True)


