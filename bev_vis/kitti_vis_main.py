 
from __future__ import print_function
 
import os
import sys
import cv2
import os.path
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
from kitti_object import *
 
 
def visualization(image_path, label_path, calib_path, lidar_path):
    dataset = kitti_object(image_path, label_path, calib_path, lidar_path)   # linux 路径
 
    # 1-加载标签数据
    objects = dataset.get_label_objects()
    print("There are %d objects.", len(objects))
 
    # 2-加载图像
    img = dataset.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
 
    # 3-加载点云数据
    pc_velo = dataset.get_lidar()[:,0:3] # (x, y, z)
 
    # 4-加载标定参数
    calib = dataset.get_calibration()
 
    # 5-可视化原始图像
    print(' ------------ show raw image -------- ')
    Image.fromarray(img).show()
    
    # 6-在图像中画2D框
    print(' ------------ show image with 2D bounding box -------- ')
    show_image_with_boxes(img, objects, calib, False)
 
    # 7-在图像中画3D框
    print(' ------------ show image with 3D bounding box ------- ')
    show_image_with_boxes(img, objects, calib, True)
    
    # 8-将点云数据投影到图像
    print(' ----------- LiDAR points projected to image plane -- ')
    show_lidar_on_image(pc_velo, img, calib, img_width, img_height)
 
    # 9-画BEV图
    print('------------------ BEV of LiDAR points -----------------------------')
    show_lidar_topview(pc_velo, objects, calib)
 
     # 10-在BEV图中画2D框
    print('--------------- BEV of LiDAR points with bobes ---------------------')
    img1 = cv2.imread('save_output/BEV.png')     
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    show_lidar_topview_with_boxes(img1, objects, calib)
    
    
if __name__=='__main__':
    image_path = r'/home/pxr/pxrProject/3Ddetection/MonoSKD/data/KITTI3D/kitti/testing/image_2/000008.png'
    calib_path = r'/home/pxr/pxrProject/3Ddetection/MonoSKD/data/KITTI3D/kitti/testing/calib/000008.txt'
    label_path = r'/home/pxr/pxrProject/3Ddetection/MonoSKD/test_result/data/000008.txt'
    lidar_path = r'/home/pxr/pxrProject/3Ddetection/MonoSKD/data/KITTI3D/kitti/testing/velodyne/000008.bin'
    
    visualization(image_path, label_path, calib_path, lidar_path)