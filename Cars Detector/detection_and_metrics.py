import torch
from torch import nn
from copy import deepcopy
import numpy as np

from scipy.signal import correlate2d, gaussian
from cv2 import resize

def rescale(img, shape):
    a = np.zeros(shape, dtype = img.dtype)
    a[0:img.shape[0], 0:img.shape[1]] = img
    return a

# ============================== 1 Classifier model ============================

class Classifier(nn.Module):
    def __init__(self, input_shape):
        super(Classifier, self).__init__()
        n_rows, n_cols, n_channels = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
#             nn.MaxPool2d(2)
        )
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16128, 2),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.clf(self.features(x))

def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    return Classifier(input_shape)

def fit_cls_model(X, y):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    from torch.utils.data import TensorDataset, DataLoader
    from torch import optim
    from torch.nn import BCELoss
    from torch.nn.functional import one_hot
    import torch
    
    model = get_cls_model((40, 100, 1))
    
    traindataset = TensorDataset(X, one_hot(y).type(torch.float))
    
    trainloader = DataLoader(traindataset, batch_size = 32)
    
    criterion = BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1.5e-3)
    
    log_n = 25
    
    for epoch in range(40):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % log_n == log_n - 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / log_n))
                running_loss = 0.0    
    
    print('Finished Training')
    
#     torch.save(model.state_dict(), 'classifier_model.pth')
    
    return model

# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    def linear_to_conv(linear, conv):
        linear_sd = linear.state_dict().copy()
        linear_sd['weight'] = linear_sd['weight'].view(conv.state_dict()['weight'].shape)
        linear_sd['bias'] = linear_sd['bias'].view(conv.state_dict()['bias'].shape)
        conv.load_state_dict(linear_sd)
        return conv
    
    model_sd = cls_model
    cls_model = get_cls_model((40, 100, 1))
    cls_model.load_state_dict(model_sd)
    
    linear1 = list(cls_model.clf.modules())[2]
    conv1 = nn.Conv2d(128, 2, (6, 21), padding = 'same')
    conv1 = linear_to_conv(linear1, conv1)
    
#     linear2 = list(cls_model.clf.modules())[4]
#     conv2 = nn.Conv2d(64, 32, 1, padding = 'same')
#     conv2 = linear_to_conv(linear2, conv2)
    
#     linear3 = list(cls_model.clf.modules())[6]
#     conv3 = nn.Conv2d(32, 2, 1, padding = 'same')
#     conv3 = linear_to_conv(linear3, conv3)
    
    detection_model = deepcopy(cls_model)
    
    detection_model.clf = nn.Sequential(
        conv1,
#         conv2,
#         conv3,
    )
    
    return detection_model

# ============================ 3 Simple detector ===============================

def gaussian_kernel(kernel_size = (40, 100), std = 3):
    std_x = std
    std_y = std / kernel_size[0] * kernel_size[1]
    g_x = gaussian(kernel_size[0], std = std).reshape(-1)
    g_y = gaussian(kernel_size[1], std = std_y).reshape(-1)
    kernel = np.outer(g_x, g_y)
    return kernel

def normalize(img):
    shifted = img - img.min()
    return shifted / shifted.max()

def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    CAR_SIZE = (40, 100)
    
    def bbox_at(img, center):
        max_idx = center
        return img[max(0, max_idx[0] - CAR_SIZE[0] // 2):min(max_idx[0] + CAR_SIZE[0] // 2, img.shape[0]), max(0, max_idx[1] - CAR_SIZE[1] // 2):min(max_idx[1] + CAR_SIZE[1] // 2, img.shape[1])].copy()
    
    def detections_by_heatmap(heatmap):
        detections = []
        kernel = gaussian_kernel(CAR_SIZE, 10)
        corr = correlate2d(heatmap, kernel, mode = 'same')
        
        max_idx = np.unravel_index(np.argmax(corr), corr.shape)
        bbox = bbox_at(corr, max_idx)
        conf = np.clip(normalize(bbox).sum() / kernel.sum(), 0, 1)
        conf_lower_threshold = 0.1 # to iterate through out of bounds
        conf_threshold = 0.1
        
        EPS_x = 5 # boundary epsilon
        EPS_y = 20
        while conf > conf_lower_threshold:
#             print(conf, max_idx)
            if not (CAR_SIZE[0] // 2 - EPS_x <= max_idx[0] <= corr.shape[0] - CAR_SIZE[0] // 2 + EPS_x) or not (CAR_SIZE[1] // 2 - EPS_y <= max_idx[1] <= corr.shape[1] - CAR_SIZE[1] // 2 + EPS_y): # out of bounds
                corr[max(0, max_idx[0] - CAR_SIZE[0] // 2):min(max_idx[0] + CAR_SIZE[0] // 2, corr.shape[0]), max(0, max_idx[1] - CAR_SIZE[1] // 2):min(max_idx[1] + CAR_SIZE[1] // 2, corr.shape[1])] = 0
#                 print('out of bound')
            else:
#                 print(max_idx, corr.shape)
                if conf > conf_threshold:
                    detections.append([max_idx[0] - CAR_SIZE[0] // 2, max_idx[1] - CAR_SIZE[1] // 2, CAR_SIZE[0], CAR_SIZE[1], conf])
                corr[max(0, max_idx[0] - CAR_SIZE[0] // 2):min(max_idx[0] + CAR_SIZE[0] // 2, corr.shape[0]), max(0, max_idx[1] - CAR_SIZE[1] // 2):min(max_idx[1] + CAR_SIZE[1] // 2, corr.shape[1])] = 0 # zero car bbox
            
            max_idx = np.unravel_index(np.argmax(corr), corr.shape)
            bbox = bbox_at(corr, max_idx)
            conf = np.clip(normalize(bbox).sum() / kernel.sum(), 0, 1)
            
        return detections
    
    max_shape = (220, 370)
    feature_map_threshold = 0.7
    detections = {}
    
    detection_model.eval()
    
    for fname, image in dictionary_of_images.items():
#         img = rescale(image, max_shape)
        inp = torch.tensor(image).view((1, 1, *image.shape))
        heatmap = detection_model(inp).detach().cpu().numpy()
        car_class = resize(normalize(heatmap[0, 1]), image.shape[::-1])
        car_class[car_class < feature_map_threshold] = 0
        detections[fname] = detections_by_heatmap(car_class)
    
    return detections


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    first_bbox = np.array(first_bbox).astype(np.int32)
    second_bbox = np.array(second_bbox).astype(np.int32)
    
    # for negative xy
    shift = np.min([first_bbox, second_bbox])
    first_bbox[:2] -= shift
    second_bbox[:2] -= shift
    
    rows = np.max([first_bbox[0] + first_bbox[2], second_bbox[0] + second_bbox[2]])
    cols = np.max([first_bbox[1] + first_bbox[3], second_bbox[1] + second_bbox[3]])
    
    mask1 = np.zeros((rows, cols), dtype = bool)
    mask2 = np.zeros((rows, cols), dtype = bool)
    
    mask1[first_bbox[0]:first_bbox[0] + first_bbox[2], first_bbox[1]:first_bbox[1] + first_bbox[3]] = True
    mask2[second_bbox[0]:second_bbox[0] + second_bbox[2], second_bbox[1]:second_bbox[1] + second_bbox[3]] = True
    
    union = mask1 | mask2
    intersection = mask1 & mask2
    
    return np.sum(intersection) / np.sum(union)


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    
    tp = []
    fp = []
    
    gt_len = 0
    
    for fname, detections in pred_bboxes.items(): 
        gt_len += len(gt_bboxes[fname])
        
        sorted_pred = sorted(detections, key = lambda x: x[-1], reverse = True)
        iou_thr = 0.5
        
        for detection in sorted_pred:
            max_iou = 0
            idx = 0
            for i, gt_detection in enumerate(gt_bboxes[fname]):
                iou = calc_iou(detection[:-1], gt_detection)
                if iou > max_iou:
                    max_iou = iou
                    idx = i

            if max_iou >= iou_thr:
                tp.append(detection)
                del gt_bboxes[fname][idx]
            else:
                fp.append(detection)

    tp = sorted(tp, key = lambda x: x[-1])
    
    all_preds = deepcopy(tp)
    all_preds.extend(fp)
    all_preds = sorted(all_preds, key = lambda x: x[-1])
    
    
    k = 0

    pr_curve = [
        (len(tp) / len(all_preds), len(tp) / gt_len, 0) # zero confidence
    ]
    
    confs = []
    for i, detection in enumerate(all_preds):
        c = detection[-1]
        if c not in confs:
            confs.append(c)
        else:
            continue
        
        tp_num = 0
        while k < len(tp) and tp[k][-1] < c:
            k += 1
        
        if k == len(tp):
            tp_num = 0
        else:
            tp_num = len(tp[k:])
        
        precision = tp_num / len(all_preds[i:])
        recall = tp_num / gt_len
        
        pr_curve.append((precision, recall, c))
    

    pr_curve.append((1, 0, 1)) # 1 confidence
    
    auc = 0
    for i in range(1, len(pr_curve)):
        h = np.abs(pr_curve[i - 1][1] - pr_curve[i][1])
        apb = (pr_curve[i][0] + pr_curve[i - 1][0]) / 2
        auc += h * apb
    return auc


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr = 0.7):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # 0.3: 0.88
    # 0.5: 0.9
    # 0.7: 0.9
    blen = 0
    tdlen = 0
    for fname, detections in detections_dictionary.items():
        blen += len(detections)
        sorted_preds = list(sorted(enumerate(detections), key = lambda x: x[1][-1], reverse = True))
        i = 0
        to_delete = []
        while i < len(sorted_preds):
            j = i + 1
            while j < len(sorted_preds):
                if calc_iou(sorted_preds[i][1], sorted_preds[j][1]) > iou_thr:
                    idx = sorted_preds[j][0]
                    to_delete.append(idx)
                    del sorted_preds[j]
                    tdlen += 1
                    j -= 1
                j += 1
            i += 1
        
        
        for idx in sorted(to_delete, reverse = True):
            del detections[idx]
        
    alen = 0
    for fname, detections in detections_dictionary.items():
        alen += len(detections)
        
    return detections_dictionary