from torchvision import models
fruit_model = models.resnet18(pretrained=True)
number_features = fruit_model.fc.in_features
for param in fruit_model.parameters():
    param.requires_grad = False
fruit_model.fc = nn.Linear(number_features, 5) # 5 Klassen: Apfel, Bananen, Orangen, Erdbeeren, Trauben