from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image



import torch, cv2
import numpy as np
import torchvision
from PIL import Image

from lib.datasets.kitti import KITTI
from lib.models.backbone4heatmap import Extractor, DID

def load_ckpt(model, ckpt_path):
    model_state_dict = model.state_dict()

    ckpt = torch.load(ckpt_path, map_location='cpu')
    pretrained_state_dict = ckpt['model_state']

    for k, v in pretrained_state_dict.items():
        if k in model_state_dict:
            model_state_dict[k] = v
    # model_state_dict.update(pretrained_state_dict)

    model.load_state_dict(model_state_dict)
    return model

class DifferenceFromConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return 1 - cos(model_output, self.features)

# class ResnetFeatureExtractor(torch.nn.Module):
#     def __init__(self, model):
#         super(ResnetFeatureExtractor, self).__init__()
#         self.model = model
#         self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
#
#     def __call__(self, x):
#         return self.feature_extractor(x)[:, :, 0, 0]


def inference(model, inputs, img_path=r'/home/pxr/pxrProject/3Ddetection/MonoSKD/data/KITTI3D/kitti/training/image_2/000694.png'):
    model = model.eval()
    img = np.array(Image.open(img_path))
    img = cv2.resize(img, (1280, 384))
    rgb_img_float = np.float32(img) / 255
    # input_tensor = preprocess_image(rgb_img_float,
    #                                 mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    concept = model(inputs['rgb'])[0, :]
    not_concept = [DifferenceFromConceptTarget(concept)]

    target_layers = [model.feat_up.ida_2.node_3.conv]
    # cam = GradCAM(model, target_layers)
    cam = EigenCAM(model, target_layers)
    targets = []
    grayscale_cam = cam(input_tensor=inputs['rgb'], targets=not_concept)
    grayscale_cam = grayscale_cam[0, :, :]
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    pl_image = Image.fromarray(visualization)
    pl_image.save('./heatmap_d.png')

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import yaml

    levels = [1, 1, 1, 3, 4, 1]
    channels = [16, 32, 128, 256, 512, 1024]

    # ckpt = r'/home/pxr/pxrProject/3Ddetection/MonoSKD/kitti_models/logs/rgb_baseline/checkpoints/student.pth'
    ckpt = r'/home/pxr/pxrProject/3Ddetection/MonoSKD/distilled.pth'
    # ckpt = 'teacher.pth'
    config = '/home/pxr/pxrProject/3Ddetection/MonoSKD/configs/monoskd.yaml'
    model_cfg = yaml.load(open(config, 'r'), Loader=yaml.Loader)

    data_cfg = {'random_flip': 0.0,
                'random_crop': 1.0,
                'scale': 0.4,
                'shift': 0.1,
                'use_dontcare': False,
                'class_merging': False,
                'writelist': ['Pedestrian', 'Car', 'Cyclist'],
                'use_3d_center': False,
                'data_dir': 'data/KITTI3D',
                'dense_depth_dir': '/home/pxr/pxrProject/3Ddetection/MonoSKD/data/KITTI3D/kitti/training/depth_dense',
                'label_dir': 'data/KITTI3D/kitti/training/label_2'}

    dataset = KITTI('/home/pxr/pxrProject/3Ddetection/MonoSKD', 'train', data_cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    mean_size = dataloader.dataset.cls_mean_size

    model = DID(mean_size=mean_size)
    # print(model)

    model = load_ckpt(model, ckpt)
    # print(model)

    inputs, calib, coord_range, targets, info = next(iter(dataloader))
    input_image = inputs['rgb']
    print(f'input image shape: {input_image.shape}')

    inference(model, inputs)
    # print(f'coord_range: {coord_range}, shape: {coord_range.shape}, calib: {calib}, shape: {calib.shape}')

