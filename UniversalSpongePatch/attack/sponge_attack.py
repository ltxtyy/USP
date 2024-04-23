import os.path
import pickle
import random
from pathlib import Path

import cv2
import torch
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm

from local_yolos.yolov5.utils.general import non_max_suppression, xyxy2xywh
from attacks_tools.early_stopping_patch import EarlyStopping
from util.tool import get_model, forward, repeat_fill


transt = transforms.ToTensor()
transp = transforms.ToPILImage()



class IoU(nn.Module):
    def __init__(self, conf_threshold, iou_threshold, img_size, device) -> None:
        super(IoU, self).__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device

    def forward(self, output_clean, output_patch):
        batch_loss = []

        gn = torch.tensor(self.img_size)[[1, 0, 1, 0]]
        gn = gn.to(self.device)
        pred_clean_bboxes = non_max_suppression(output_clean, self.conf_threshold, self.iou_threshold, classes=None,
                                                max_det=1000)
        patch_conf = 0.001
        pred_patch_bboxes = non_max_suppression(output_patch, patch_conf, self.iou_threshold, classes=None,
                                                max_det=30000)


        for (img_clean_preds, img_patch_preds) in zip(pred_clean_bboxes, pred_patch_bboxes):  # per image

            for clean_det in img_clean_preds:

                clean_clss = clean_det[5]

                clean_xyxy = torch.stack([clean_det])  # .clone()
                clean_xyxy_out = (clean_xyxy[..., :4] / gn).to(
                    self.device)

                img_patch_preds_out = img_patch_preds[img_patch_preds[:, 5].view(-1) == clean_clss]

                patch_xyxy_out = (img_patch_preds_out[..., :4] / gn).to(self.device)

                if len(clean_xyxy_out) != 0:
                    target = self.get_iou(patch_xyxy_out, clean_xyxy_out)
                    if len(target) != 0:
                        target_m, _ = target.max(dim=0)
                    else:
                        target_m = torch.zeros(1).to(self.device)

                    batch_loss.append(target_m)

        one = torch.tensor(1.0).to(self.device)
        if len(batch_loss) == 0:
            return one

        return (one - torch.stack(batch_loss).mean())

    
    def get_iou(self, bbox1, bbox2):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            bbox1: (tensor) Ground truth bounding boxes, Shape: [num_objects, 4]
            bbox2: (tensor) Prior boxes from priorbox layers, Shape: [num_priors, 4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """

        inter = self.intersect(bbox1, bbox2)
        area_a = ((bbox1[:, 2] - bbox1[:, 0]) *
                  (bbox1[:, 3] - bbox1[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((bbox2[:, 2] - bbox2[:, 0]) *
                  (bbox2[:, 3] - bbox2[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union

    def intersect(self, box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

class UniversalSpongePatch:
    def __init__(self, patch_folder, train_loader, val_loader, epsilon, lambda_1, lambda_2, iter_eps=0.05, penalty_regularizer=0,
                  use_cuda=True, epochs=70, img_size=[640, 640], patch_size=[640, 640], models_vers=[5], use_patch=False):

        self.use_cuda = use_cuda and torch.cuda.is_available()
        print("CUDA Available: ", self.use_cuda)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.train_loader = train_loader
        self.val_loader = val_loader

        # load wanted models
        self.models = []
        self.model_names = []
        if 3 in models_vers:
            self.models.append(get_model('yolov3'))
            self.model_names.append('yolov3')
        if 5 in models_vers:
            self.models.append(get_model('yolov5'))
            self.model_names.append('yolov5')
        if 8 in models_vers:
            self.models.append(get_model('yolov8'))
            self.model_names.append('yolov8')
        

        self.iter_eps = iter_eps
        self.penalty_regularizer = penalty_regularizer
        self.use_patch = use_patch

        self.epsilon = epsilon
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.momentum = 0
        self.decay = 0.9

        self.epochs = epochs
        self.patch_size = patch_size
        self.image_size = img_size
        self.iou = IoU(conf_threshold=0.25, iou_threshold=0.45, img_size=self.image_size, device=self.device)

        self.full_patch_folder = "uap_train/" + patch_folder + "/"
        Path(self.full_patch_folder).mkdir(parents=True, exist_ok=True)

        self.current_dir = "experiments/" + patch_folder
        self.create_folders()

        self.current_train_loss = 0.0
        self.current_max_objects_loss = 0.0
        self.current_orig_classification_loss = 0.0
        self.min_bboxes_added_preds_loss = 0.0

        self.train_losses = []
        self.max_objects_loss = []
        self.orig_classification_loss = []
        self.val_losses = []
        self.val_max_objects_loss = []
        self.val_orig_classification_loss = []

        self.writer = None

    def create_folders(self):
        Path('/'.join(self.current_dir.split('/')[:2])).mkdir(parents=True, exist_ok=True)
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/final_results').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/saved_patches').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/losses').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/testing').mkdir(parents=True, exist_ok=True)


    def last_batch_calc(self, adv_patch, epoch_length,
                        epoch, i_batch):
        # calculate epoch losses
        self.current_train_loss /= epoch_length
        self.current_max_objects_loss /= epoch_length
        self.current_orig_classification_loss /= epoch_length

        self.train_losses.append(self.current_train_loss)
        self.max_objects_loss.append(self.current_max_objects_loss)
        self.orig_classification_loss.append(self.current_orig_classification_loss)

        # check on validation
        val_loss, sep_val_loss = self.evaluate_loss(self.val_loader, adv_patch)
        self.val_max_objects_loss.append(sep_val_loss[0])
        self.val_orig_classification_loss.append(sep_val_loss[1])
        self.val_losses.append(val_loss)

        if self.writer is not None:
            self.writer.add_scalar('loss/val_loss', val_loss, epoch_length * epoch + i_batch)

    def save_final_objects(self, adv_patch):
        # save patch
        transforms.ToPILImage()(adv_patch).save(
            self.current_dir + '/final_results/final_patch.png', 'PNG')

        # save losses
        with open(self.current_dir + '/losses/train_losses', 'wb') as fp:
            pickle.dump(self.train_losses, fp)
        with open(self.current_dir + '/losses/val_losses', 'wb') as fp:
            pickle.dump(self.val_losses, fp)
        with open(self.current_dir + '/losses/max_objects_losses', 'wb') as fp:
            pickle.dump(self.max_objects_loss, fp)
        with open(self.current_dir + '/losses/orig_classification_losses', 'wb') as fp:
            pickle.dump(self.orig_classification_loss, fp)

    def evaluate_loss(self, loader, adv_patch):
        val_loss = []
        max_objects_loss = []
        orig_classification_loss = []
        min_bboxes_added_preds_loss = []

        if self.use_patch:
            adv_patch = repeat_fill(adv_patch, self.image_size[0], self.image_size[1])

        adv_patch = adv_patch.to(self.device)

        for (img_batch, _) in loader:
            r = random.randint(0, len(self.models) - 1)

            with torch.no_grad():
                img_batch = torch.stack(img_batch)
                img_batch = img_batch.to(self.device)

                applied_batch = torch.clamp(img_batch[:] + adv_patch, 0, 1)

                output_clean = forward(self.models[r], img_batch, self.model_names[r])
                output_patch = forward(self.models[r], applied_batch, self.model_names[r])

                max_objects = self.max_objects(output_patch)

                bboxes_area = self.bboxes_area(output_clean, output_patch)

                iou = self.iou(output_clean, output_patch)

                batch_loss = max_objects.item() * self.lambda_1

                max_objects_loss.append(max_objects.item() * self.lambda_1)

                if not torch.isnan(iou):
                    batch_loss += (iou.item() * (1 - self.lambda_1))
                    orig_classification_loss.append(iou.item() * (1 - self.lambda_1))

                if not torch.isnan(bboxes_area):
                    batch_loss += (bboxes_area * self.lambda_2)

                val_loss.append(batch_loss)

                del img_batch, applied_batch, output_patch, batch_loss

                torch.cuda.empty_cache()

        loss = sum(val_loss) / len(val_loss)
        max_objects_loss = sum(max_objects_loss) / len(max_objects_loss)

        orig_classification_loss = sum(orig_classification_loss) / len(orig_classification_loss)

        print(f"total loss: {loss}")
        return loss, [max_objects_loss, min_bboxes_added_preds_loss, orig_classification_loss]


    def max_objects(self, output_patch, conf_thres=0.25, target_class=2):

        x2 = output_patch[:, :, 5:] * output_patch[:, :, 4:5]

        conf, j = x2.max(2, keepdim=False)
        all_target_conf = x2[:, :, target_class]
        under_thr_target_conf = all_target_conf[conf < conf_thres]

        conf_avg = len(conf.view(-1)[conf.view(-1) > conf_thres]) / len(output_patch)
        # print(f"pass to NMS: {conf_avg}")

        zeros = torch.zeros(under_thr_target_conf.size()).to(output_patch.device)
        zeros.requires_grad = True
        x3 = torch.maximum(-under_thr_target_conf + conf_thres, zeros)
        mean_conf = torch.sum(x3, dim=0) / (output_patch.size()[0] * output_patch.size()[1])

        return mean_conf

    def bboxes_area(self, output_clean, output_patch, conf_thres=0.25):

        t_loss = 0.0

        xc_patch = output_patch[..., 4] > conf_thres
        not_nan_count = 0

        # For each img in the batch
        for (xi, x), (li, l) in (zip(enumerate(output_patch), enumerate(output_clean))):  # image index, image inference

            x1 = x[xc_patch[xi]]  # box who > conf_thres
            x2 = x1[:, 5:] * x1[:, 4:5]  # # Compute conf for each class

            box_x1 = x1[:, :4]

            conf_x1, j_x1 = x2.max(1, keepdim=True)
            x1_full = torch.cat((box_x1, conf_x1, j_x1.float()), 1)
            x1_full = x1_full[conf_x1.view(-1) > conf_thres]

            # calculate bboxes' area avg
            bboxes_x1_wh = x1_full[:, 2:4]
            bboxes_x1_area = bboxes_x1_wh[:, 0] * bboxes_x1_wh[:, 1]
            img_loss = bboxes_x1_area.mean() / (self.patch_size[0] * self.patch_size[1])
            if not torch.isnan(img_loss):
                t_loss += img_loss
                not_nan_count += 1

        if not_nan_count == 0:
            t_loss_f = torch.tensor(torch.nan)
        else:
            t_loss_f = t_loss / not_nan_count

        return t_loss_f

    def loss_function_gradient(self, applied_patch, init_images, adv_patch):

        if self.use_cuda:
            init_images = init_images.cuda()
            applied_patch = applied_patch.cuda()
        r = random.randint(0, len(self.models)-1) # choose a random model

        with torch.no_grad():
            output_clean = forward(self.models[r], init_images, self.model_names[r]).detach()
        output_patch = forward(self.models[r], applied_patch, self.model_names[r])


        max_objects_loss = self.max_objects(output_patch)
        bboxes_area_loss = self.bboxes_area(output_clean, output_patch)
        iou_loss = self.iou(output_clean, output_patch)

        loss = max_objects_loss * self.lambda_1

        if not torch.isnan(iou_loss):
            loss += (iou_loss * (1 - self.lambda_1))
            self.current_orig_classification_loss += ((1 - self.lambda_1) * iou_loss.item())

        if not torch.isnan(bboxes_area_loss):
            loss += (bboxes_area_loss * self.lambda_2)

        self.current_train_loss += loss.item()
        self.current_max_objects_loss += (self.lambda_1 * max_objects_loss.item())

        if self.use_cuda:
            loss = loss.cuda()

        self.models[r].zero_grad()
        data_grad = torch.autograd.grad(loss, adv_patch)[0]
        return data_grad

    def fastGradientSignMethod(self, adv_patch, images, epsilon):
        
        if self.use_patch:
            adv_patch = repeat_fill(adv_patch, self.image_size[0], self.image_size[1])

        applied_patch = torch.clamp(images[:] + adv_patch, 0, 1)

        data_grad = self.loss_function_gradient(applied_patch, images, adv_patch)  

        # 利用动量
        # data_grad = data_grad + self.momentum * self.decay
        # self.momentum = data_grad

        # 更新梯度
        perturbed_patch = adv_patch - epsilon * data_grad.sign()
        # Adding clipping to maintain [0,1] range
        perturbed_patch_c = torch.clamp(perturbed_patch, 0, 1).detach()
        # Return the perturbed image
        return perturbed_patch_c

    def pgd_L2(self, epsilon=0.1, iter_eps=0.05, min_x=0.0, max_x=1.0):
        early_stop = EarlyStopping(delta=1e-4, current_dir=self.current_dir, patience=7)

        # 初始化
        start = 0
        patch_size = self.patch_size if self.use_patch else self.image_size
        patch = torch.zeros([3, patch_size[0], patch_size[1]])

        # 继续上次的训练
        if os.path.exists(self.full_patch_folder):
            files = os.listdir(self.full_patch_folder)
            if len(files) != 0:
                files = sorted(files, key=lambda x: int(x.split('_')[2]))
                start = int(files[-1].split('_')[2])
                path = os.path.join(self.full_patch_folder, f'uap_epoch_{start}_batch_0.png')
                adv = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                patch = transforms.ToTensor()(adv)

        patch.requires_grad = True

        adv_patch = patch
        for epoch in range(start, self.epochs):
            epoch_length = len(self.train_loader)
            print('Epoch:', epoch)
            if epoch == 0:
                val_loss = self.evaluate_loss(self.val_loader, adv_patch)[0]
                early_stop(val_loss, adv_patch.cpu(), epoch)

            # Perturb the input
            self.current_train_loss = 0.0
            self.current_max_objects_loss = 0.0
            self.current_orig_classification_loss = 0.0

            i = 0
            for (imgs, _) in tqdm(self.train_loader):  # for imgs in self.train_loader: #self.coco_train:
                if i % 25 == 0:
                    # print(f"batch {i}:")
                    patch_n = self.full_patch_folder + f"uap_epoch_{epoch}_batch_{i}.png"
                    transp(adv_patch).save(patch_n)

                x = torch.stack(imgs)

                adv_patch = self.fastGradientSignMethod(adv_patch, x, epsilon=iter_eps)

                # Project the perturbation to the epsilon ball (L2 projection)
                perturbation = adv_patch - patch

                norm = torch.sum(torch.square(perturbation))
                norm = torch.sqrt(norm)
                factor = min(1, epsilon / norm.item())  # torch.divide(epsilon, norm.numpy()[0]))

                adv_patch = (torch.clip(patch + perturbation * factor, min_x, max_x))  # .detach()

                i += 1
                if i == epoch_length:
                    self.last_batch_calc(adv_patch, epoch_length, epoch, i)

            # check if loss has decreased
            if early_stop(self.val_losses[-1], adv_patch.cpu(), epoch):
                self.final_epoch_count = epoch
                break

        print("Training finished")
        return early_stop.best_patch

    def run_attack(self):
        tensor_adv_patch = self.pgd_L2(epsilon=self.epsilon, iter_eps=0.0005)  # 05

        patch = tensor_adv_patch

        self.save_final_objects(tensor_adv_patch)
        adv_image = transp(patch if self.use_patch else patch[0])

        return adv_image

