import datetime
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import base64
import io
from dash import *
import os
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import torch
import torch.nn as nn
from dash import dcc, html
from dash.exceptions import PreventUpdate
from simple3DCNN import Simple3DCNN
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from EffGRU import CombinedModel
from zipfile import ZipFile


# Define la función para preprocesar las imágenes para VGG16 y CombinedModel
def preprocess_image_for_combined_model(image):
    # Transformaciones para preprocesar las imágenes para VGG16
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Cambiar el tamaño a 224x224 (tamaño de entrada de la VGG16)
        transforms.ToTensor(),  # Convertir a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización requerida por VGG16
    ])

    preprocessed_image = transform(image)  # Aplica las transformaciones
    return preprocessed_image

UPLOAD_DIRECTORY = "uploaded_images"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

accuracy_C1 = pd.read_csv('Data/accuracy_C1.csv')
accuracy_C2 = pd.read_csv('Data/accuracy_C2.csv')
accuracy_C3 = pd.read_csv('Data/accuracy_C3.csv')
accuracy_C4 = pd.read_csv('Data/accuracy_C4.csv')
accuracy_C5 = pd.read_csv('Data/accuracy_C5.csv')
accuracy_C6 = pd.read_csv('Data/accuracy_C6.csv')
accuracy_C7 = pd.read_csv('Data/accuracy_C7.csv')

F1_C1 = pd.read_csv('Data/F1_C1.csv')
F1_C2 = pd.read_csv('Data/F1_C2.csv')
F1_C3 = pd.read_csv('Data/F1_C3.csv')
F1_C4 = pd.read_csv('Data/F1_C4.csv')
F1_C5 = pd.read_csv('Data/F1_C5.csv')
F1_C6 = pd.read_csv('Data/F1_C6.csv')
F1_C7 = pd.read_csv('Data/F1_C7.csv')

loss_C1 = pd.read_csv('Data/loss_C1.csv')
loss_C2 = pd.read_csv('Data/loss_C2.csv')
loss_C3 = pd.read_csv('Data/loss_C3.csv')
loss_C4 = pd.read_csv('Data/loss_C4.csv')
loss_C5 = pd.read_csv('Data/loss_C5.csv')
loss_C6 = pd.read_csv('Data/loss_C6.csv')
loss_C7 = pd.read_csv('Data/loss_C7.csv')

accuracy_3DCNN = pd.read_csv('Data/accuracy_3D.csv')
F1_3DCNN = pd.read_csv('Data/F1_3D.csv')
loss_3DCNN = pd.read_csv('Data/loss_3D.csv')

train_imagenes = pd.read_csv('train_filtrado_images.csv')
train_volumenes = pd.read_csv('train_filtrado_volumes.csv')
train_vertebras = pd.read_csv('meta_train_with_vertebrae.csv')

current_accuracy = accuracy_3DCNN
current_loss = loss_3DCNN

valores = ["", "", "", "", "", "", ""]

modelos = {
    "3DCNN": {
        "accuracy": accuracy_3DCNN,
        "F1": F1_3DCNN,
        "loss": loss_3DCNN,
        "modelo": "modelos/modelo_entrenado3D.pth",
        "name": "3DCNN"
    },
    "C1": {
        "accuracy": accuracy_C1,
        "F1": F1_C1,
        "loss": loss_C1,
        "modelo": "modelos/modelo_entrenadoC1.pth",
        "name": "C1"
    },
    "C2": {
        "accuracy": accuracy_C2,
        "F1": F1_C2,
        "loss": loss_C2,
        "modelo": "modelos/modelo_entrenadoC2.pth",
        "name": "C2"
    },
    "C3": {
        "accuracy": accuracy_C3,
        "F1": F1_C3,
        "loss": loss_C3,
        "modelo": "modelos/modelo_entrenadoC3.pth",
        "name": "C3"
    },
    "C4": {
        "accuracy": accuracy_C4,
        "F1": F1_C4,
        "loss": loss_C4,
        "modelo": "modelos/modelo_entrenadoC4.pth",
        "name": "C4"
    },
    "C5": {
        "accuracy": accuracy_C5,
        "F1": F1_C5,
        "loss": loss_C5,
        "modelo": "modelos/modelo_entrenadoC5.pth",
        "name": "C5"
    },
    "C6": {
        "accuracy": accuracy_C6,
        "F1": F1_C6,
        "loss": loss_C6,
        "modelo": "modelos/modelo_entrenadoC6.pth",
        "name": "C6"
    },
    "C7": {
        "accuracy": accuracy_C7,
        "F1": F1_C7,
        "loss": loss_C7,
        "modelo": "modelos/modelo_entrenadoC7.pth",
        "name": "C7"
    }
}

current_model = modelos["3DCNN"]

# Crear un control deslizante para seleccionar años
available_years = range(0, current_loss.shape[0])
year_slider = dcc.RangeSlider(
    id='year-slider',
    min=min(available_years),
    max=max(available_years) + 1,
    step=1,
    marks={str(available_years[i - 1]): str(available_years[i - 1]) for i in range(0, len(available_years) + 1, 80)},

    value=[min(available_years), max(available_years)]
)

# Crear un control deslizante para seleccionar años
available_years_C1 = range(0, accuracy_C1.shape[0])
year_slider_C1 = dcc.RangeSlider(
    id='year-slider',
    min=min(available_years_C1),
    max=max(available_years_C1) + 1,
    step=1,
    marks={str(available_years_C1[i - 1]): str(available_years_C1[i - 1]) for i in range(0, len(available_years_C1) + 1, 80)},

    value=[min(available_years_C1), max(available_years_C1)]
)

# Crear un control deslizante para seleccionar años
available_years_C2 = range(0, accuracy_C2.shape[0])
year_slider_C2 = dcc.RangeSlider(
    id='year-slider',
    min=min(available_years_C2),
    max=max(available_years_C2) + 1,
    step=1,
    marks={str(available_years_C2[i - 1]): str(available_years_C2[i - 1]) for i in range(0, len(available_years_C2) + 1, 80)},

    value=[min(available_years_C2), max(available_years_C2)]
)

# Crear un control deslizante para seleccionar años
available_years_C3 = range(0, accuracy_C3.shape[0])
year_slider_C3 = dcc.RangeSlider(
    id='year-slider',
    min=min(available_years_C3),
    max=max(available_years_C3) + 1,
    step=1,
    marks={str(available_years_C3[i - 1]): str(available_years_C3[i - 1]) for i in range(0, len(available_years_C3) + 1, 80)},
    value=[min(available_years_C3), max(available_years_C3)]
)

available_years_C4 = range(0, accuracy_C4.shape[0])
year_slider_C4 = dcc.RangeSlider(
    id='year-slider',
    min=min(available_years_C4),
    max=max(available_years_C4) + 1,
    step=1,
    marks={str(available_years_C4[i - 1]): str(available_years_C4[i - 1]) for i in range(0, len(available_years_C4) + 1, 80)},
    value=[min(available_years_C4), max(available_years_C4)]
)

available_years_C5 = range(0, accuracy_C5.shape[0])
year_slider_C5 = dcc.RangeSlider(
    id='year-slider',
    min=min(available_years_C5),
    max=max(available_years_C5) + 1,
    step=1,
    marks={str(available_years_C5[i - 1]): str(available_years_C5[i - 1]) for i in range(0, len(available_years_C5) + 1, 80)},
    value=[min(available_years_C5), max(available_years_C5)]
)

available_years_C6 = range(0, accuracy_C6.shape[0])
year_slider_C6 = dcc.RangeSlider(
    id='year-slider',
    min=min(available_years_C6),
    max=max(available_years_C6) + 1,
    step=1,
    marks={str(available_years_C6[i - 1]): str(available_years_C6[i - 1]) for i in range(0, len(available_years_C6) + 1, 80)},
    value=[min(available_years_C6), max(available_years_C6)]
)


available_years_C7 = range(0, accuracy_C7.shape[0])
year_slider_C7 = dcc.RangeSlider(
    id='year-slider',
    min=min(available_years_C7),
    max=max(available_years_C7) + 1,
    step=1,
    marks={str(available_years_C7[i - 1]): str(available_years_C7[i - 1]) for i in range(0, len(available_years_C7) + 1, 80)},
    value=[min(available_years_C7), max(available_years_C7)]
)




    
        



model_selector = dcc.Dropdown(
    id='model-selector',
    options=[
        {'label': '3DCNN', 'value': '3DCNN'},
        {'label': 'C1', 'value': 'C1'},
        {'label': 'C2', 'value': 'C2'},
        {'label': 'C3', 'value': 'C3'},
        {'label': 'C4', 'value': 'C4'},
        {'label': 'C5', 'value': 'C5'},
        {'label': 'C6', 'value': 'C6'},
        {'label': 'C7', 'value': 'C7'}
    ],
    value='3DCNN'
)

app = Dash(__name__, suppress_callback_exceptions=True)

# app.layout = html.Div([
#     html.H1('Predicciones de fracturas en vértebras cervicales', style={'margin-top': '30px','text-align': 'center', 'font-family': 'Helvetica'}),
#     html.Div([
#         html.Div([
#             html.Label('Selecciona el modelo que se va a utilizar:'),
#             model_selector,
#         ], style = {'font-family': 'Helvetica'}),
#         html.Div([
#             dcc.Graph(id='grafico-tiempo', style={'width': '50%', 'display': 'inline-block'}),
#             dcc.Graph(id='grafico-barras', style={'width': '50%', 'display': 'inline-block'}),
#         ])
#     ])
# ], style={'background-color': '#FFF'})

app.layout = html.Div([
    html.H1('Predicciones de fracturas en vértebras cervicales', style={'margin-top': '30px','text-align': 'center', 'font-family': 'Helvetica'}),
    html.Div([
            html.Label('Selecciona el modelo que se va a utilizar:'),
            model_selector,
        ], style = {'font-family': 'Helvetica'}),
    html.Div(id='output-div'),
], style={'background-color': '#FFF'})

@app.callback(
    Output('output-div', 'children'),
    [Input('model-selector', 'value')]
)
def execute_action(selected_action):
    # if selected_action == '3DCNN':
    if selected_action in modelos:
        global current_model
        current_model = modelos[selected_action]

        slider = None
        if selected_action == "3DCNN":
            slider = year_slider
        if selected_action == "C1":
            slider = year_slider_C1
        if selected_action == "C2":
            slider = year_slider_C2
        if selected_action == "C3":
            slider = year_slider_C3
        if selected_action == "C4":
            slider = year_slider_C4
        if selected_action == "C5":
            slider = year_slider_C5
        if selected_action == "C6":
            slider = year_slider_C6
        if selected_action == "C7":
            slider = year_slider_C7
        
        
        return html.Div([
                    html.Div([
                        dcc.Graph(id='grafico-tiempo', style={'width': '50%', 'display': 'inline-block'}),
                        dcc.Graph(id='grafico-loss', style={'width': '50%', 'display': 'inline-block'}),
                    ]),
                    html.Div([
                            html.Label('Selecciona el rango de épocas:'),
                            slider,
                        ], style = {'font-family': 'Helvetica'}),
                    
                    # Agrega la sección de carga de archivos
                    html.Div([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Button('Cargar archivo'),
                            multiple=False  # Para permitir la carga de un solo archivo a la vez
                        ),
                        html.Div(id='output-data-upload')  # Aquí se mostrará el resultado de la carga
                    ]),
                ])


# # Callback para actualizar el gráfico de serie de tiempo en función de los filtros
# @app.callback(
#     Output('grafico-tiempo1', 'figure'),
#     [Input('grafico-tiempo1', 'relayoutData'),
#      Input('year-slider', 'value')]
# )
# def update_time_series_graph(relayoutData, selected_years):
#     filtered_df = df1[(df1['Fecha'].dt.year >= selected_years[0]) & (df1['Fecha'].dt.year <= selected_years[1])]

#     merged_df = pd.merge(filtered_df, pred_df1, on='Fecha', how='left')
#     merged_df['Diesel_conjunto_pred'] = merged_df['Diesel conjunto']
#     merged_df.drop(columns=['Diesel conjunto'], inplace=True)

#     figura = px.line(merged_df, x='Fecha', y=['Diesel_conjunto', 'Diesel_conjunto_pred'], 
#                      title='Diesel conjunto importada en galones', labels={'Diesel_conjunto': 'Diesel', 'Gasolina_regular_pred': 'Diesel predicho'})
#     figura.update_layout(paper_bgcolor='#ECEBE4')
#     # figura = px.line(filtered_df, x='Fecha', y='Gasolina_regular', title='Serie de Tiempo')
#     # figura.update_traces(line=dict(color='#153B50'))
#     figura.update_traces(line=dict(color='#153B50'), selector=dict(name='Diesel_conjunto'))
#     figura.update_traces(line=dict(color='#CC998D'), selector=dict(name='Diesel_conjunto_pred'))
    
#     figura.update_traces(line=dict(color='#153B50'), selector=dict(name='Diesel_conjunto'), 
#                          name='Diesel consumido real')
#     figura.update_traces(line=dict(color='#CC998D'), selector=dict(name='Diesel_conjunto_pred'), 
#                          name='Diesel Conjunto Predicho')
#     figura.update_yaxes(title_text='Diesel consumido predicho')
#     return figura

# Callback para actualizar el gráfico de serie de tiempo en función de los filtros
@app.callback(
    Output('grafico-tiempo', 'figure'),
    [Input('grafico-tiempo', 'relayoutData'),
    Input('year-slider', 'value')]
)
def update_time_series_graph(relayoutData, selected_years):
    global current_model
    df = current_model["accuracy"]["Value"]
    name = current_model["name"]

    epochs = range(selected_years[0], selected_years[1])
    df = df.iloc[epochs]

    figura = px.line(df, x=epochs, y='Value', 
                    title=f'Accuracy {name}', labels={'Value': 'Value'})
    figura.update_layout(paper_bgcolor='#ECEBE4')
    # figura = px.line(filtered_df, x='Fecha', y='Gasolina_regular', title='Serie de Tiempo')
    # figura.update_traces(line=dict(color='#153B50'))
    figura.update_traces(line=dict(color='#153B50'), selector=dict(name='Value'))
    # figura.update_traces(line=dict(color='#CC998D'), selector=dict(name='Gasolina_regular_pred'))
    
    # figura.update_traces(line=dict(color='#153B50'), selector=dict(name='Gasolina_regular'), 
    #                      name='Gasolina Regular')
    # figura.update_traces(line=dict(color='#CC998D'), selector=dict(name='Gasolina_regular_pred'), 
    #                      name='Gasolina Regular Predicha')
    # figura.update_yaxes(title_text='Galones importados')
    return figura

@app.callback(
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'), State('upload-data', 'last_modified')]
)

def save_uploaded_file(contents, filename, last_modified):
    global current_model
    if contents is None:
        raise PreventUpdate
    
    # Aquí, contents es una cadena en formato "data: tipo_de_dato;base64,datos"
    content_type, content_string = contents.split(',')

    # Decodificar la parte base64 de la cadena (segundo elemento del split)
    decoded = base64.b64decode(content_string)
    # Guardar el archivo en el sistema de archivos
    with open(os.path.join(UPLOAD_DIRECTORY, filename), 'wb') as f:
        f.write(decoded)

    
    # predict
    ruta = UPLOAD_DIRECTORY + '/' + filename

    path_modelo = current_model["modelo"]
    
    if current_model["name"] == "3DCNN":
        div = predicciones3DCNN(ruta, path_modelo, contents)
    else:
        div = prediccionesEfficientNet(ruta, path_modelo, contents)

    return div

def get_images(path):
    batch_images = torch.zeros((1, 70, 3, 224, 224))
    with ZipFile(path, 'r') as zipObj:
        image_files = zipObj.namelist()
        for i, image_file in enumerate(image_files):
            if i >= 70:
                break
            if image_file.endswith('.jpg'):
                with zipObj.open(image_file) as img_file:
                    image = Image.open(img_file)
                    image = preprocess_image_for_combined_model(image)
                    batch_images[0, i] = image
    
    return batch_images

def prediccionesEfficientNet(ruta, path_modelo, contents):
    images = get_images(ruta)
    predicted_labels_np = None
    vgg_output_size = 1280  # Tamaño de la salida de la VGG16
    gru_hidden_size = 128  # Tamaño del estado oculto de la GRU
    gru_num_layers = 2  # Número de capas en la GRU
    num_classes = 7  # Reemplaza con el número de clases en tu problema
    combined_model = CombinedModel(vgg_output_size, gru_hidden_size, gru_num_layers, num_classes)

    
    combined_model.load_state_dict(torch.load(path_modelo))
    combined_model.to('cuda')
    combined_model.eval()

    with torch.no_grad():
        image_tensor = images.to('cuda')
        predictions = combined_model(image_tensor)
        threshold = 0.5
        predicted_labels = (predictions > threshold).float()  # 1 si es mayor al umbral, 0 de lo contrario
        predicted_labels_np = predicted_labels.cpu().numpy()
    
    prediction = ""
    if predicted_labels_np[0][0] == 1:
        prediction = "Fractura"
    else:
        prediction = "No fractura"

    return html.Div([
        # "Archivo cargado y guardado: {}".format(filename),
        # Mostrar la imagen subida
        html.Img(src=contents, height=200),
        "Predicción: {}".format(prediction)
    ])
    


def predicciones3DCNN(ruta, path_modelo, contents):
    model3D = Simple3DCNN(7)
    model3D.load_state_dict(torch.load(path_modelo))
    model3D.to('cuda')

    volume = np.load(ruta)
    image_tensor = torch.from_numpy(volume).unsqueeze(0).float().to('cuda')
    model3D.eval()

    with torch.no_grad():
        predictions = model3D(image_tensor)
        for i in range(len(predictions)):
            predicted = predictions[i].cpu().numpy()

            # Convierte los valores en 'predicted' en una lista de cadenas
            valores = [str(int(x)) for x in predicted]
            
            for i in valores:
                if i == '1':
                    valores[valores.index(i)] = 'Fractura'
                else:
                    valores[valores.index(i)] = 'No fractura'

            # Ahora 'valores' contiene cada valor como cadena
            print(valores)

    return html.Div([
        # "Archivo cargado y guardado: {}".format(filename),
        # Mostrar la imagen subida
        html.Img(src=contents, height=200),
        html.Table([
        # Encabezado de la tabla
        html.Tr([html.Th("  C1  "), html.Th("   C2  "), html.Th("   C3  "), html.Th("   C4  "), html.Th("   C5  "), html.Th("   C6  "), html.Th("   C7  ")]),
        # Fila de valores
        html.Tr([html.Td(valores[0]), html.Td(valores[1]), html.Td(valores[2]), html.Td(valores[3]), html.Td(valores[4]), html.Td(valores[5]), html.Td(valores[6])])])

    ])


@app.callback(
    Output('grafico-loss', 'figure'),
    [Input('grafico-loss', 'relayoutData'),
    Input('year-slider', 'value')]
)
def update_time_series_graph(relayoutData, selected_years):
    global current_model
    df = current_model["loss"]["Value"]
    name = current_model["name"]
    epochs = range(selected_years[0], selected_years[1])
    df = df.iloc[epochs]

    figura = px.line(df, x=epochs, y='Value', 
                    title=f'Loss {name}', labels={'Value': 'Value'})
    figura.update_layout(paper_bgcolor='#ECEBE4')
    # figura = px.line(filtered_df, x='Fecha', y='Gasolina_regular', title='Serie de Tiempo')
    # figura.update_traces(line=dict(color='#153B50'))
    figura.update_traces(line=dict(color='#153B50'), selector=dict(name='Value'))
    # figura.update_traces(line=dict(color='#CC998D'), selector=dict(name='Gasolina_regular_pred'))
    
    # figura.update_traces(line=dict(color='#153B50'), selector=dict(name='Gasolina_regular'), 
    #                      name='Gasolina Regular')
    # figura.update_traces(line=dict(color='#CC998D'), selector=dict(name='Gasolina_regular_pred'), 
    #                      name='Gasolina Regular Predicha')
    # figura.update_yaxes(title_text='Galones importados')
    return figura



if __name__ == '__main__':
    app.run_server(debug=True)