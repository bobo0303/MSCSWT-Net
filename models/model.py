import torch.nn as nn
import logging, torch, yaml
import numpy as np
# from .transformer_layer import BlockAxial, my_Block_2
from .Gated_Conv import GatedConv2d, TransposeGatedConv2d, GatedDeConv2d
from .shift_cswin_transformer import shifted_cswin_Transformer
from .shift_cswin_transformer_4blocks import cswin_Transformer
from einops import rearrange


logger = logging.getLogger(__name__)

class inpaint_model(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, args):
        super().__init__()

        #first three layers 256>128>64>32
        self.pad1 = nn.ReflectionPad2d(3)
        self.act = nn.ReLU(True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # Gated Conv (encoder)
        self.G_Conv_1 = GatedConv2d(in_channels=4, out_channels=64, kernel_size=7, stride=1, padding=0, activation=args['activation'], norm=args['norm'])
        self.G_Conv_2 = GatedConv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_Conv_3 = GatedConv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_Conv_4 = GatedConv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])

        # shift_CSwin_Transformer (encoder)
        self.shifted_cswin_Transformer_Block1 = shifted_cswin_Transformer(args, layer='first')
        self.shifted_cswin_Transformer_Block2 = shifted_cswin_Transformer(args, layer='second')
        self.shifted_cswin_Transformer_Block3 = shifted_cswin_Transformer(args, layer='third')
        self.shifted_cswin_Transformer_Block4 = shifted_cswin_Transformer(args, layer='fourth')
        self.ln = nn.LayerNorm(32)

        # residual dilated convolution
        self.RDConv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=2, stride=1, padding=2)
        self.RDConv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, dilation=3, stride=1, padding=6)
        self.RDConv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=7, dilation=3, stride=1, padding=9)
        self.RDConv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=7, dilation=3, stride=1, padding=9)
        self.RDConv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, dilation=3, stride=1, padding=6)
        self.RDConv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=2, stride=1, padding=2)

        # Gated ConvTranspose (decoder) >original DeepFillv2
        # self.G_DeConv_1 = TransposeGatedConv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, activation=args['activation'], norm=args['norm'])
        # self.G_Conv_1_2 = GatedConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, activation=args['activation'], norm=args['norm'])
        # self.G_DeConv_2 = TransposeGatedConv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, activation=args['activation'], norm=args['norm'])
        # self.G_Conv_2_2 = GatedConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, activation=args['activation'], norm=args['norm'])
        # self.G_DeConv_3 = TransposeGatedConv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, activation='none', norm='none')
        # self.G_Conv_3_2 = GatedConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, activation=args['activation'], norm=args['norm'])
        # self.G_DeConv_4 = TransposeGatedConv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=0, activation=args['activation'], norm=args['norm'])
        # self.G_Conv_4_2 = GatedConv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=0, activation='none', norm='none')

        # myself use ConvTranspose2d
        self.G_DeConv_1 = GatedDeConv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_Conv_1_2 = GatedConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_DeConv_2 = GatedDeConv2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_Conv_2_2 = GatedConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_DeConv_3 = GatedDeConv2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_Conv_3_2 = GatedConv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=0, activation='none', norm='none')

        self.batchNorm = nn.BatchNorm2d(256)

        self.padt = nn.ReflectionPad2d(3)
        self.act_last = nn.Sigmoid()    # ZitS first stage


        self.block_size = 32

        self.apply(self._init_weights)

        # calculate parameters
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):    #https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, args, new_lr):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)


        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": float(args['weight_decay'])},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=float(new_lr), betas=(0.9, 0.95))
        return optimizer

    def forward(self, img_idx, masks=None):
        img_idx = img_idx * (1 - masks)
        x = torch.cat((img_idx, masks), dim=1)

        # four layers Gated Conv  (eccoder)
        x = self.pad1(x)
        x = self.G_Conv_1(x)
        x = self.G_Conv_2(x)
        x = self.G_Conv_3(x)
        x = self.G_Conv_4(x)

        # 3 layers residual dilated Conv
        x_resitual = x.clone()
        x = self.RDConv1(x)
        x = self.act(x)
        x = self.RDConv2(x)
        x = self.act(x)
        x = self.RDConv3(x) + x_resitual
        x = self.act(x)

        # 4 layers shift CSwin transformer
        x_resitual = x.clone()
        x = self.shifted_cswin_Transformer_Block1(x)
        x = self.shifted_cswin_Transformer_Block2(x)
        x = self.shifted_cswin_Transformer_Block3(x)
        x = self.shifted_cswin_Transformer_Block4(x) + x_resitual

        # 3 layers residual dilated Conv
        x_resitual = x.clone()
        x = self.RDConv4(x)
        x = self.act(x)
        x = self.RDConv5(x)
        x = self.act(x)
        x = self.RDConv6(x) + x_resitual
        x = self.act(x)

        # 4 layers Gated DeConv (decoder)
        x = self.G_DeConv_1(x)
        x = self.G_DeConv_2(x)
        x = self.G_DeConv_3(x)
        x = self.padt(x)
        x = self.G_Conv_3_2(x)

        x = self.act_last(x)    # Sigmoid

        return x
