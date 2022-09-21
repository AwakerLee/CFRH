import numpy as np
import torch
import clip
from PIL import Image
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from args import config
import scipy.io as scio
import h5py
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
#
# image = preprocess(Image.open("dog.png")).unsqueeze(0).to(device)
# print(image)
# text = clip.tokenize(["a redenvelope","a dog"]).to(device)
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     SIM = torch.cosine_similarity(image_features,text_features)
#     print(SIM)
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


if __name__ == '__main__':
    pass
#     mirflickr = h5py.File(config.IMG_DIR, 'r', libver='latest', swmr=True)
#     img= mirflickr['IAll'][0]
#     print(img)
#     img = Image.fromarray(np.transpose(img, (2, 1, 0)))
#     mirflickr.close()
#     print(img)
    # print(label_set.shape)
    # txt_set = scio.loadmat(config.TXT_DIR)
    # print(txt_set)
    # txt_set = txt_set['YAll']
    # # print(txt_set)
    # txt = torch.FloatTensor(txt_set)
    # print(txt)
    # label_set = scio.loadmat('E:/JDSH-clip/dataset/NUS-WIDE/nus-wide-tc10-lall.mat')
    # label_set = np.array(label_set['LAll'], dtype=np.float)
    # print(label_set[0])
    # txt_file = h5py.File('E:/JDSH-clip/dataset/NUS-WIDE/nus-wide-tc10-yall.mat','r')
    # txt_set = np.array(txt_file['YAll']).transpose()
    # print(txt_set.shape)
    # txt_file.close()
    #
    # nuswide = h5py.File('E:/JDSH-clip/dataset/NUS-WIDE/nus-wide-tc10-iall.mat', 'r', libver='latest', swmr=True)
    # img = nuswide['IAll']
    # print(img.shape)
    # img = Image.fromarray(np.transpose(img, (2, 1, 0)))
    # print(img.shape)
    #
    # nuswide.close()

    print(clip.available_models())