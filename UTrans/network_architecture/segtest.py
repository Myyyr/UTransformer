import timm.models.unest as un
import timm.models.segnest as se

import torch


def main():
    l = 6
    h = 3
    d = h*64
    net = un.UNest(img_size=512, in_chans=1, 
                  patch_size=4, num_levels=l, 
                  embed_dims=(d,)*l, num_heads=(h,)*l, 
                  depths=(3,)*l, num_classes=14, 
                  mlp_ratio=4.).float()
    net.eval()
    # net = se.SegNest(img_size=512, in_chans=1, 
    #               patch_size=4, num_levels=l, 
    #               embed_dims=(d,)*l, num_heads=(h,)*l, 
    #               depths=(3,)*l, num_classes=14, 
    #               mlp_ratio=4.).float()
    x = torch.rand(1, 1, 512, 512).float()
    print("x", x.shape)
    with torch.no_grad():
      y = net(x)
    print("len", len(y))
    for i in range(len(y)):
        print("y",i, y[i].shape)

if __name__ == '__main__':
    main()