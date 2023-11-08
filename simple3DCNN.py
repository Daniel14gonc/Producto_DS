import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        
        # Capa 3D Conv1
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Capa 3D Conv2
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        #Forzar dimensionalidad
        self.fce = nn.Linear(4800, 2400) # (64 * 75, 64*75/2)
        self.relue1 = nn.ReLU()  
        self.fce2 = nn.Linear(2400, 240)
        self.relue2 = nn.ReLU()
        self.fce3 = nn.Linear(240, 128)
        self.relue3 = nn.ReLU()
        self.fce4 = nn.Linear(128, 1)    
          
        # Capa completamente conectada
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 7)
        
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # forzar dimensionalidad
        x = x.view(-1, 64 * 75)
        x = self.fce(x)
        x = self.relue1(x)
        x = self.fce2(x)
        x = self.relue2(x)
        x = self.fce3(x)
        x = self.relue3(x)
        x = self.fce4(x)
        
        
        x = x.view(-1, 64 * 4 * 4 * 4)        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        x = self.softmax(x)
        
        return x

# # Crear una instancia del modelo
# num_classes = 7  # Número de clases de salida
# model = Simple3DCNN(num_classes)

# # Imprimir el modelo para ver su estructura
# print(model)