from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

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

modelos = {
    "3DCNN": {
        "accuracy": accuracy_3DCNN,
        "F1": F1_3DCNN,
        "loss": loss_3DCNN,
        "modelo": "modelos/modelo_entrenad3D.pth"
    },
    "C1": {
        "accuracy": accuracy_C1,
        "F1": F1_C1,
        "loss": loss_C1,
        "modelo": "modelos/modelo_entrenado_C1.pth"
    },
    "C2": {
        "accuracy": accuracy_C2,
        "F1": F1_C2,
        "loss": loss_C2,
        "modelo": "modelos/modelo_entrenado_C2.pth"
    },
    "C3": {
        "accuracy": accuracy_C3,
        "F1": F1_C3,
        "loss": loss_C3,
        "modelo": "modelos/modelo_entrenado_C3.pth"
    },
    "C4": {
        "accuracy": accuracy_C4,
        "F1": F1_C4,
        "loss": loss_C4,
        "modelo": "modelos/modelo_entrenado_C4.pth",
        "name": "C4"
    },
    "C5": {
        "accuracy": accuracy_C5,
        "F1": F1_C5,
        "loss": loss_C5,
        "modelo": "modelos/modelo_entrenado_C5.pth",
        
    },
    "C6": {
        "accuracy": accuracy_C6,
        "F1": F1_C6,
        "loss": loss_C6,
        "modelo": "modelos/modelo_entrenado_C6.pth",
        "name": "C6"
    },
    "C7": {
        "accuracy": accuracy_C7,
        "F1": F1_C7,
        "loss": loss_C7,
        "modelo": "modelos/modelo_entrenado_C7.pth",
        "name": "C7"
    }
}

current_model = modelos["3DCNN"]

available_years = range(0, current_loss.shape[0])
year_slider = dcc.RangeSlider(
    id='year-slider',
    min=min(available_years),
    max=max(available_years),
    step=1,
    marks={str(year): str(year) for year in available_years},
    value=[min(available_years), max(available_years)]
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

app = Dash(__name__)

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
    if selected_action == '3DCNN':
        return  html.Div([
            html.Div([
                dcc.Graph(id='grafico-tiempo', style={'width': '50%', 'display': 'inline-block'}),
            ]),
            html.Div([
                    html.Label('Selecciona el rango de épocas:'),
                    year_slider,
                ], style = {'font-family': 'Helvetica'}),
        ])

@app.callback(
    Output('grafico-tiempo', 'figure'),
    [Input('grafico-tiempo', 'relayoutData'),
     Input('year-slider', 'value')]
)
def update_time_series_graph(relayoutData, selected_years):
    global current_model
    df = current_model["accuracy"]["Value"]

    figura = px.line(df, x='Epochs', y='Accuracy', 
                     title='Accuracy', labels={'Value': 'Value'})
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