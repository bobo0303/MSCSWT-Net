import torch.nn.functional as F

def masked_l1_loss(Original_img, pred_img, mask, weight_known=0.7, weight_missing=0.3):

    per_pixel_l1 = F.l1_loss(Original_img, pred_img, reduction='none')    #L1 loss
    pixel_weights = mask * weight_missing + (1 - mask) * weight_known
    return (pixel_weights * per_pixel_l1).mean()

def l1_loss(Original_img, pred_img, Edge):

    per_pixel_l1 = F.l1_loss(Original_img, pred_img, reduction='none')    # L1 loss
    if Edge != None:
        per_pixel_l1 = per_pixel_l1 * (1 - Edge) + per_pixel_l1 * Edge * 10     # add edge weight
    return per_pixel_l1.mean()    # mean
