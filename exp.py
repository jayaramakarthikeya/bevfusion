import torch

orig = torch.load('/home/autokarthik/bevfusion/model/resnet50/bevfusion-det.pth')
# if orig is already a dict of weights, otherwise extract orig['state_dict']
wrapped = {
    'state_dict': orig,
    'meta': {}   # you can fill in any metadata if you want
}
torch.save(wrapped, '/home/autokarthik/bevfusion/model/resnet50/bevfusion-det-mmcv.pth')
