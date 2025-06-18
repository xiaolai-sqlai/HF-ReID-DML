import torch
import torch.nn as nn


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, kernel_size=3, stride=1, use_act=True):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2, bias=False)
        self.norm = nn.BatchNorm2d(out_planes)
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        out = self.norm(self.conv(x))
        out = self.act(out)
        return out


class Block(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, group_width, stride):
        super().__init__()
        self.proj_in = ConvX(in_dim, h_dim, kernel_size=1)
        self.main_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvX(h_dim, h_dim, kernel_size=1, use_act=False)

        )
        self.main_local = ConvX(h_dim, h_dim, groups=h_dim//group_width, kernel_size=3, stride=stride, use_act=False)
        self.act = nn.ReLU(inplace=True)
        self.proj_out = ConvX(h_dim, out_dim, kernel_size=1, use_act=False)
        
        if stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                ConvX(in_dim, in_dim, groups=in_dim, kernel_size=3, stride=stride, use_act=False),
                ConvX(in_dim, out_dim, kernel_size=1)
            )

    def forward(self, x):
        out = self.proj_in(x)
        out = self.main_global(out) + self.main_local(out)
        out = self.act(out)
        out = self.proj_out(out) + self.skip(x)
        return out


class StageModule(nn.Module):
    def __init__(self, layers, dim, out_dim, mlp_ratio=1.0, group_width=4):
        super().__init__()
        self.layers = []
        for idx in range(layers):
            if idx == 0:
                self.layers.append(Block(in_dim=dim, h_dim=int(mlp_ratio*out_dim), out_dim=out_dim, group_width=group_width, stride=2))
            else:
                self.layers.append(Block(in_dim=out_dim, h_dim=int(mlp_ratio*out_dim), out_dim=out_dim, group_width=group_width, stride=1))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)



class CRCNN(nn.Module):
    def __init__(self, num_classes=1000, embed_dim=64, mlp_ratio=1.0, layers=[4,7,3], group_width=4):
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            ConvX(3, embed_dim//8, kernel_size=3, stride=2),
            ConvX(embed_dim//8, embed_dim//4, kernel_size=3, stride=2)
        )

        self.stage1 = StageModule(layers[0], embed_dim//4, embed_dim, mlp_ratio=mlp_ratio, group_width=group_width)
        self.stage2 = StageModule(layers[1], embed_dim, embed_dim*2, mlp_ratio=mlp_ratio, group_width=group_width)
        self.stage3 = StageModule(layers[2], embed_dim*2, embed_dim*4, mlp_ratio=mlp_ratio, group_width=group_width)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Classifier head
        self.head = nn.Linear(embed_dim*4, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


if __name__=="__main__":
    from thop import profile, clever_format
    custom_ops = {}
    input = torch.randn(1, 3, 384, 128)

    # model = CRCNN(embed_dim=64, mlp_ratio=1.00, layers=[8,24,2])
    # model = CRCNN(embed_dim=80, mlp_ratio=1.00, layers=[6,14,2])
    model = CRCNN(embed_dim=96, mlp_ratio=1.00, layers=[3,8,2])
    model.eval()
    print(model)

    macs, params = profile(model, inputs=(input, ), custom_ops=custom_ops)
    macs, params = clever_format([macs, params], "%.3f")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print('Flops:  ', macs)
    print('Params: ', params)



