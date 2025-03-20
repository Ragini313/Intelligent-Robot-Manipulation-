import torch.nn.functional as F
import torch.nn as nn
import torch
import os

class SimpleCNN(nn.Module):

    def __init__(self, num_classes = 10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 30 * 30, 128), # 60 * 60 (2 convs), 30 * 30 (3 convs)
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y
    



if __name__ =='__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    import torchvision.transforms as T
    torch.manual_seed(0)
    model_path = os.path.join("saved_models", "best_model", "run_0", "model.zip")
    params = torch.load(model_path)
    model = SimpleCNN()
    model.load_state_dict(params)
    model.eval()
    image = Image.open("cube_7_test_image.png")
    transformation = T.Compose([
        T.Grayscale(),
        T.Resize((240,240)),
        T.ToTensor()
    ])

    x = transformation(image)
    plt.imshow(x[0],cmap='grey')
    plt.show()
    x = x[None, :,:,:]
    y: torch.Tensor = model(x)
    print(y)
    print(y.argmax())