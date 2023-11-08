import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, efficientnet_output_size, gru_hidden_size, gru_num_layers, num_classes):
        super(CombinedModel, self).__init__()
        
        # Cargar EfficientNet preentrenado
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        # Eliminar la capa Fully Connected
        self.features = nn.Sequential(*list(self.efficientnet.children())[:-1])
        
        # Pooling Global Promedio
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # GRU Layer
        self.gru = nn.GRU(input_size=efficientnet_output_size, hidden_size=gru_hidden_size, batch_first=True)
        
        # Fully Connected Layer
        self.fc = nn.Linear(gru_hidden_size, 1)
        self.dropout = nn.Dropout(0.5)
        
        # Softmax
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        
        # Pasar imágenes por EfficientNet
        x = self.features(x)
        
        # Pooling Global Promedio
        x = self.global_avg_pool(x).squeeze(-1).squeeze(-1)
        
        x = x.view(batch_size, timesteps, -1)
        # print(x.shape)
        # Pasar la secuencia de feature maps por la GRU
        out, _ = self.gru(x)
        # Usar solo la última salida de la secuencia
        out = out[:, -1, :]
        # Pasar por la capa Fully Connected
        out = self.dropout(self.fc(out))
        # Softmax
        out = self.sigmoid(out)
        
        return out
