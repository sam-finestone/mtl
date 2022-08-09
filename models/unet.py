import torch
import torch.nn as nn
import torch.nn.functional as F


# MTL Network with 2 auxiliary tasks
class MTLNet(nn.Module):
    def __init__(self, c_in, c_out_seg):
        super().__init__()

        # Contracting Path (Encoder)
        self.contract1 = contract_block(c_in=c_in, c_out=32, kernel_size=7, padding=3)
        self.contract2 = contract_block(c_in=32, c_out=64, kernel_size=3, padding=1)
        self.contract3 = contract_block(c_in=64, c_out=128, kernel_size=3, padding=1)
        self.contract4 = contract_block(c_in=128, c_out=256, kernel_size=3, padding=1)

        # Segmentation - Expansive Path (Decoder)
        self.expand_seg1 = expand_block(c_in=256, c_out=128, kernel_size=3, padding=1)
        self.expand_seg2 = expand_block(c_in=128 * 2, c_out=64, kernel_size=3, padding=1)
        self.expand_seg3 = expand_block(c_in=64 * 2, c_out=32, kernel_size=3, padding=1)
        self.expand_seg4 = expand_block(c_in=32 * 2, c_out=c_out_seg, kernel_size=3, padding=1)

        # Binary Classification
        self.class_conv1 = nn.Conv2d(256, 32, 5)
        self.class_pool = nn.MaxPool2d(2, 2)
        self.class_conv2 = nn.Conv2d(32, 64, 5)
        self.class_fc1 = nn.Linear(64, 120)
        self.class_fc2 = nn.Linear(120, 84)
        self.class_fc3 = nn.Linear(84, 2)

        # Bounding Box Regressor
        self.bbox_conv1 = nn.Conv2d(256, 256, 3)
        self.bbox_pool = nn.MaxPool2d(2, 2)
        self.bbox_conv2 = nn.Conv2d(256, 256, 3)
        self.bbox_fc1 = nn.Linear(1024, 128)
        self.bbox_fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # Encoder
        x1 = self.contract1(x)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)

        # Decoder - Segmentation
        x_seg1 = self.expand_seg1(x4)
        x_seg2 = self.expand_seg2(torch.cat([x_seg1, x3], 1))
        x_seg3 = self.expand_seg3(torch.cat([x_seg2, x2], 1))
        x_seg4 = self.expand_seg4(torch.cat([x_seg3, x1], 1))

        # Binary Classification
        x_class = self.class_pool(F.relu(self.class_conv1(x4)))
        x_class = self.class_pool(F.relu(self.class_conv2(x_class)))
        x_class = torch.flatten(x_class, 1)
        x_class = F.relu(self.class_fc1(x_class))
        x_class = F.relu(self.class_fc2(x_class))
        x_class = self.class_fc3(x_class)

        # Bounding Box Regression
        x_bbox = self.bbox_pool(F.relu(self.bbox_conv1(x4)))
        x_bbox = self.bbox_pool(F.relu(self.bbox_conv2(x_bbox)))
        x_bbox = torch.flatten(x_bbox, 1)
        x_bbox = F.relu(self.bbox_fc1(x_bbox))
        x_bbox = F.dropout(x_bbox, p=0.5)
        x_bbox = F.relu(self.bbox_fc2(x_bbox))
        x_bbox = torch.sigmoid(x_bbox)

        return x_class, x_seg4, x_bbox


# Ablated network with bounding box regression task removed
class AblatedNet(nn.Module):
    def __init__(self, c_in, c_out_seg):
        super().__init__()

        # Contracting Path (Encoder)
        self.contract1 = contract_block(c_in=c_in, c_out=32, kernel_size=7, padding=3)
        self.contract2 = contract_block(c_in=32, c_out=64, kernel_size=3, padding=1)
        self.contract3 = contract_block(c_in=64, c_out=128, kernel_size=3, padding=1)
        self.contract4 = contract_block(c_in=128, c_out=256, kernel_size=3, padding=1)

        # Segmentation - Expansive Path (Decoder)
        self.expand_seg1 = expand_block(c_in=256, c_out=128, kernel_size=3, padding=1)
        self.expand_seg2 = expand_block(c_in=128 * 2, c_out=64, kernel_size=3, padding=1)
        self.expand_seg3 = expand_block(c_in=64 * 2, c_out=32, kernel_size=3, padding=1)
        self.expand_seg4 = expand_block(c_in=32 * 2, c_out=c_out_seg, kernel_size=3, padding=1)

        # Binary Classification
        self.class_conv1 = nn.Conv2d(256, 32, 5)
        self.class_pool = nn.MaxPool2d(2, 2)
        self.class_conv2 = nn.Conv2d(32, 64, 5)
        self.class_fc1 = nn.Linear(64, 120)
        self.class_fc2 = nn.Linear(120, 84)
        self.class_fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Encoder
        x1 = self.contract1(x)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)

        # Decoder - Segmentation
        x_seg1 = self.expand_seg1(x4)
        x_seg2 = self.expand_seg2(torch.cat([x_seg1, x3], 1))
        x_seg3 = self.expand_seg3(torch.cat([x_seg2, x2], 1))
        x_seg4 = self.expand_seg4(torch.cat([x_seg3, x1], 1))

        # Binary Classification
        x_class = self.class_pool(F.relu(self.class_conv1(x4)))
        x_class = self.class_pool(F.relu(self.class_conv2(x_class)))
        x_class = torch.flatten(x_class, 1)
        x_class = F.relu(self.class_fc1(x_class))
        x_class = F.relu(self.class_fc2(x_class))
        x_class = self.class_fc3(x_class)

        return x_class, x_seg4


# Ablated network with image classification task removed
class AblatedNet2(nn.Module):
    def __init__(self, c_in, c_out_seg):
        super().__init__()

        # Contracting Path (Encoder)
        self.contract1 = contract_block(c_in=c_in, c_out=32, kernel_size=7, padding=3)
        self.contract2 = contract_block(c_in=32, c_out=64, kernel_size=3, padding=1)
        self.contract3 = contract_block(c_in=64, c_out=128, kernel_size=3, padding=1)
        self.contract4 = contract_block(c_in=128, c_out=256, kernel_size=3, padding=1)

        # Segmentation - Expansive Path (Decoder)
        self.expand_seg1 = expand_block(c_in=256, c_out=128, kernel_size=3, padding=1)
        self.expand_seg2 = expand_block(c_in=128 * 2, c_out=64, kernel_size=3, padding=1)
        self.expand_seg3 = expand_block(c_in=64 * 2, c_out=32, kernel_size=3, padding=1)
        self.expand_seg4 = expand_block(c_in=32 * 2, c_out=c_out_seg, kernel_size=3, padding=1)

        # Bounding Box Regressor
        self.bbox_conv1 = nn.Conv2d(256, 256, 3)
        self.bbox_pool = nn.MaxPool2d(2, 2)
        self.bbox_conv2 = nn.Conv2d(256, 256, 3)
        self.bbox_fc1 = nn.Linear(1024, 128)
        self.bbox_fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # Encoder
        x1 = self.contract1(x)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)

        # Decoder - Segmentation
        x_seg1 = self.expand_seg1(x4)
        x_seg2 = self.expand_seg2(torch.cat([x_seg1, x3], 1))
        x_seg3 = self.expand_seg3(torch.cat([x_seg2, x2], 1))
        x_seg4 = self.expand_seg4(torch.cat([x_seg3, x1], 1))

        # Bounding Box Regression
        x_bbox = self.bbox_pool(F.relu(self.bbox_conv1(x4)))
        x_bbox = self.bbox_pool(F.relu(self.bbox_conv2(x_bbox)))
        x_bbox = torch.flatten(x_bbox, 1)
        x_bbox = F.relu(self.bbox_fc1(x_bbox))
        x_bbox = F.dropout(x_bbox, p=0.5)
        x_bbox = F.relu(self.bbox_fc2(x_bbox))
        x_bbox = torch.sigmoid(x_bbox)

        return x_seg4, x_bbox


# Baseline U-Net for Image Segmentation
class UNet(nn.Module):
    def __init__(self, c_in, c_out_seg):
        super().__init__()

        # Contracting Path (Encoder)
        self.contract1 = contract_block(c_in=c_in, c_out=32, kernel_size=7, padding=3)
        self.contract2 = contract_block(c_in=32, c_out=64, kernel_size=3, padding=1)
        self.contract3 = contract_block(c_in=64, c_out=128, kernel_size=3, padding=1)
        self.contract4 = contract_block(c_in=128, c_out=256, kernel_size=3, padding=1)

        # Segmentation - Expansive Path (Decoder)
        self.expand_seg1 = expand_block(c_in=256, c_out=128, kernel_size=3, padding=1)
        self.expand_seg2 = expand_block(c_in=128 * 2, c_out=64, kernel_size=3, padding=1)
        self.expand_seg3 = expand_block(c_in=64 * 2, c_out=32, kernel_size=3, padding=1)
        self.expand_seg4 = expand_block(c_in=32 * 2, c_out=c_out_seg, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x1 = self.contract1(x)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)

        # Decoder - Segmentation
        x_seg1 = self.expand_seg1(x4)
        x_seg2 = self.expand_seg2(torch.cat([x_seg1, x3], 1))
        x_seg3 = self.expand_seg3(torch.cat([x_seg2, x2], 1))
        x_seg4 = self.expand_seg4(torch.cat([x_seg3, x1], 1))

        return x_seg4


# Contracting block
class contract_block(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, padding):
        super().__init__()
        self.contract = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.contract(x)

        return x


# Expansion block
class expand_block(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, padding):
        super().__init__()
        self.expand = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size, stride=1, padding=padding),
                                    nn.BatchNorm2d(c_out),
                                    nn.ReLU(),
                                    nn.Conv2d(c_out, c_out, kernel_size, stride=1, padding=padding),
                                    nn.BatchNorm2d(c_out),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(c_out, c_out, kernel_size=3, stride=2, padding=1,
                                                       output_padding=1)
                                    )

    def forward(self, x):
        x = self.expand(x)

        return x