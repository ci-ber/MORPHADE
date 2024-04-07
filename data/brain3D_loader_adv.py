from core.DataLoader import DefaultDataset
import torchvision.transforms as transforms
from transforms.preprocessing import *
import torchio

class Flip:
    """
    Flip brain

    """

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        return torch.tensor((img.astype(np.float32)).copy())


class Brain3DLoader(DefaultDataset):
    def __init__(self, data_dir, file_type='', label_dir=None, mask_dir=None, target_size=(128, 128, 128), test=False):
        self.target_size = target_size
        self.RES = torchio.transforms.Resize(self.target_size)
        super(Brain3DLoader, self).__init__(data_dir, file_type, label_dir, mask_dir, target_size, test)

    def get_image_transform(self):
        default_t = transforms.Compose([ReadImage2(), Norm98()#, Norm98(),
                                       # ,Pad3D((1, 1, 1), type='end')  # Flip(), #  Slice(),
                                        ,AddChannelIfNeeded(dim=3),
                                       #,Resize3D(self.target_size)
                                        # ,AdjustIntensity()
                                        #
                                     #   torchio.RandomElasticDeformation(
                                     #       num_control_points=(7, 7, 7),  # or just 7
                                     #       locked_borders=2,
                                     #   ),
                                        torchio.RandomGamma(log_gamma=(-0.3, 0.3)),
                                         torchio.RandomFlip(axes=('LR')),
                                        # transforms.RandomAffine(15, (0.1, 0.1), (0.9, 1.1)),
                                       #  torchio.transforms.RandomNoise(),
                                        # transforms.RandomHorizontalFlip(0.5),
                                       #  transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8,1.2)),
                                        # ,transforms.ToTensor()
                                        ])
        return default_t

    def get_image_transform_test(self):
        default_t_test = transforms.Compose([ReadImage2(), Norm98()#, Norm98()
                                      # ,Pad3D((1, 1, 1), type='end')
                                        # Flip(), #  Slice(),
                                        ,AddChannelIfNeeded(dim=3)
                                       #,Resize3D(self.target_size)
                                        ])
        return default_t_test

    def get_label_transform(self):
        default_t_label = transforms.Compose([ReadImage2(),  To01()
                                            # ,Pad3D((1, 1))
                                             ,AddChannelIfNeeded()
                                             ,AssertChannelFirst()
                                            # ,self.RES
                                            ])#, Binarize()])
        return default_t_label