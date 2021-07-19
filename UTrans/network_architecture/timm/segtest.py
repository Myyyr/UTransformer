import timm
import torch


def main():
    h = 3
    d = h*64
    net = timm.models.segnest.SegNest(img_size=512, in_chans=1, 
                  patch_size=2, num_levels=4, 
                  embed_dims=(d, d, d, d), num_heads=(h, h, h, h), 
                  depths=(3, 3, 3, 3), num_classes=14, 
                  mlp_ratio=4.).float()
    x = torch.rand(1, 1, 512, 512).float()
    print("x", x.shape)
    y = net(x)
    print("len", len(y))
    for i in range(len(y)):
        print("y",i, y[i].shape)

if __name__ == '__main__':
    main()