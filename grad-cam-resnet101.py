import os
import cv2
import glob
import torch
import traceback
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import numpy as np
import argparse


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if name == "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input



class GradCam:
    def __init__(self, model, target_layer_names, device):
        self.model = model
        self.model.eval()
        self.model = model.to(device)
        
        self.extractor = ModelOutputs(self.model, target_layer_names)
        
    def forward(self, input):
        return self.model(input)
    
    def __call__(self, input, topk=1):
    
        bboxes_top = list()
        
        
        for k in range(topk):
            features, output = self.extractor(input.to(device))
        
            index = np.argsort(output.cpu().data.numpy())[-1][-(k+1)]  # top probs to low
            
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
            
            one_hot = torch.sum(one_hot.to(device) * output)
            
            #self.model.features.zero_grad()
            #self.model.classifier.zero_grad()
            self.model.zero_grad()
            
            one_hot.backward(retain_graph=True)
            
            cam_list = []
            for j in range(len(features)):
                grads_val = self.extractor.get_gradients()[j].cpu().data.numpy()
                target = features[len(features) - j - 1]
                target = target.cpu().data.numpy()[0, :]
                
                weights = np.mean(grads_val, axis=(2, 3))[0, :]
                cam = np.zeros(target.shape[1:], dtype=np.float32)
                
                for i, w in enumerate(weights):
                    cam += w * target[i, :, :]
                
                cam = np.maximum(cam, 0)
                cam = cv2.resize(cam, (224, 224))
                cam = cam - np.min(cam)
                cam = cam / np.max(cam)
                
                cam_list.append(cam)
        
            for i in range(len(cam_list) - 1):
                cam_list[i] = cv2.resize(cam_list[i], (cam_list[i + 1].shape[0], cam_list[i + 1].shape[1]))
                #cam_list[i] = cv2.resize(cam_list[i], (224, 224))
                
                #cam_list[i + 1] = cam_list[i + 1] + cam_list[i]
                cam_list[i + 1] = np.array((cam_list[i + 1], cam_list[i])).max(axis=0)
            
            mask = cam_list[-1] 
            
            mask_copy = mask.copy()
            
            shreld = mask.sum() / (mask.shape[0] * mask.shape[1]) * 1.7
            
            mask = np.array(mask >= shreld, dtype='uint8')
            
            ret, binary = cv2.threshold(mask, shreld, 255, cv2.THRESH_BINARY)
            
            _, contours, hierarcy = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            cont_ = sorted(contours, key=cv2.contourArea, reverse=True)
            
            if len(cont_) == 0:
                box = np.zeros((4,2), dtype=int)
            else:
                c = cont_[0]
                # compute the rotated bounding box of the largest contour
                rect = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(rect))
            bboxes_top.append({"classify":index, "bbox":box, "mask":mask_copy})
        return bboxes_top


def visualize_label(img, boxes, path, color=(0, 255, 0)):
    """
    img: HWC
    boxes: array of num * 4 * 2
    """
    boxes = np.array(boxes).reshape(-1, 4, 2)
    img = np.ascontiguousarray(img)
    
    cv2.drawContours(img, boxes, -1, color, thickness=2)
    
    cv2.imwrite(path, img * 255)


def show_cam_on_image(img, mask, path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(path, np.uint8(255 * cam))

def checkpoint(box, origin_ratio):
    
    h_ratio, w_ratio = origin_ratio
    box = np.array(box).reshape(-1, 4, 2)
    xmin, ymin = box.min(1)[0]
    xmax, ymax = box.max(1)[0]
    xmin,ymin,xmax,ymax = np.max((xmin,0)),np.max((ymin,0)),np.min((xmax,224)),np.min((ymax,224))
    xmin,ymin,xmax,ymax = int(xmin * w_ratio), int(ymin * h_ratio), int(xmax * w_ratio), int(ymax * h_ratio)
    return str(xmin), str(ymin), str(xmax), str(ymax)
    
def visualize(img, boxes, path, color=(0, 255, 0)):
    
    xmin, ymin, xmax, ymax =  boxes
    img = np.ascontiguousarray(img)
    
    x,y,w,h = int(xmin), int(ymin), int(xmax)-int(xmin), int(ymax)-int(ymin) 
    #cv2.drawContours(img, boxes, -1, color, thickness=2)
    cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
    # return img
    cv2.imwrite(path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type= bool, default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--gpus', type=str, default="2", help="default GPU devices (0,1)")
    parser.add_argument('--target_layer', type=list, default=["layer1", "layer2", "layer3", "layer4"], help="default GPU devices (0,1)")#"8", "17", "26", "35", 
    parser.add_argument('--result-path', type=str, default='./Resnet101_result', help='Input image path')
    parser.add_argument('--image-path', type=str, default='./dataset', help='Input image path')
    
    args = parser.parse_args()
    
    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
        device = 'cuda'
        print("Using GPU for acceleration")
    else:
        device = 'cpu'
        print("Using CPU for computation")
    
    image_list = glob.glob(os.path.join(args.image_path, "*.JPEG"))
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    writer = open(os.path.join(args.result_path, "submission.txt"), "a")
    
    for num, image_path in enumerate(sorted(image_list)):
        
        try:
            # get the gradient model
            grad_cam = GradCam(model=models.resnet101(pretrained=True), target_layer_names=args.target_layer, device=device)
            
            image_name = image_path.replace(args.image_path+"/", "").replace(".JPEG", "")
            
            result_img_dir = args.result_path
            
            if not os.path.exists(result_img_dir):
                os.makedirs(result_img_dir)
            
            img = cv2.imread(image_path, 1)
            
            img_ = img.copy()
            
            origin_ratio = (img.shape[0] / 224, img.shape[1] / 224)
            
            img = np.float32(cv2.resize(img, (224, 224))) / 255
            
            cv2.imwrite(os.path.join(result_img_dir, image_name+"_source.jpg"), img * 255)
            
            input = preprocess_image(img)
            
            # If None, returns the map for the highest scoring category(top one).
            # Otherwise, returns the top5 category. the bbox is gotten by top to low.
            bboxs = grad_cam(input, topk=5)
            
            cam = np.zeros(img.shape[:-1], dtype=np.float32)
            
            ROIs = list()
            
            for i, bbox in enumerate(bboxs):
                if i == 0:
                    show_cam_on_image(img.copy(), bbox["mask"], os.path.join(result_img_dir, "top_" + str(i) + "_" + image_name + "_cam.jpg"))
                    visualize_label(img.copy(), bbox["bbox"], os.path.join(result_img_dir, "top_" + str(i) + "_" + image_name + "_bbox.jpg"))
                    ROIs.append(str(bbox["classify"]))
                    ROIs = ROIs + list(checkpoint(bbox["bbox"], origin_ratio))
                    cam = np.array((cam, bbox["mask"])).max(axis=0)
                    #visualize_label(img_.copy(), bbox["bbox"], os.path.join(result_img_dir, "top_" + str(i) + "_" + image_name + "_source_bbox.jpg"))
                else:
                    show_cam_on_image(img.copy(), bbox["mask"], os.path.join(result_img_dir, "top_" + str(i) + "_" + image_name + "_cam.jpg"))
                    visualize_label(img.copy(), bbox["bbox"], os.path.join(result_img_dir, "top_" + str(i) + "_" + image_name + "_bbox.jpg"))
                    ROIs.append(str(bbox["classify"]))
                    ROIs = ROIs + list(checkpoint(bbox["bbox"], origin_ratio))
                    cam = np.array((cam, bbox["mask"])).max(axis=0)
            
            writer.write(" ".join(ROIs)+"\n")
            show_cam_on_image(img.copy(), cam, os.path.join(result_img_dir, image_name + "_all_cam.jpg"))
            print(num, image_name + ".JPEG is finished!")
            exit()
        except Exception as e:
            traceback.print_exc()