from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import json
import os
import numpy as np
from io import StringIO
from picture_text.picture_text import PictureText
from picture_text.src.feedback_form import contact_form
import dash_bootstrap_components as dbc

model = 'gpt4'
extract_schema = 'summary_entity1'
emb_model_name = 'oAI-3s'
root_path = os.environ.get('VST_SAMPLE_DATA','./sample_data')

test = os.environ.get("VST_TEST",50)

#app = Dash(__name__)
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def create_analysis_view(collection_name, trm_fig):
    if test > 0:
        test_str = f" [TESTING: {test}]"
    else:
        test_str = ""
    return html.Div([
        html.H1(children=f'{test_str} Analysing: {collection_name}', style={'textAlign':'center'}),
        dbc.Row(
            [
                dbc.Col(dbc.Container(
                    dbc.Row(
                    id='ls_cards',
                    justify="evenly",
                    ),
                    ), 
                    width=7),
                dbc.Col(  
                    html.Div(  
                        [dcc.Graph(
                            id = 'treemap',
                            figure = trm_fig
                        ),
                        contact_form(),
                        html.P(id='err', style={'color': 'red'}),
                        html.P(id='out')]
                    )
                , width=5),
            ]
        ),
    ])


def prep_data(collection_name, width = 400):
    save_to = os.path.join(root_path,f'topic_n_ent_{collection_name}_{model}_{extract_schema}_{emb_model_name}.json')
    
    text_data = json.load(open(save_to,'r'))
    print(len(text_data))
    print(text_data[0].keys())
    if test > 0:
        text_data = text_data[:test]

    txt_embeddings = [td['embedding'] for td in text_data]
    txt = [f"{td['topic_tag']}" for td in text_data]
    pt = PictureText(txt)
    pt(txt_embeddings=txt_embeddings,encoder=None,hac_method='ward', hac_metric='euclidean')
    df_res, trm_fig = pt.make_picture(layer_depth = 4,             
                        layer_min_size = 0.1,
                        layer_max_extension = 1,
                        treemap_average_score = None, 
                        treemap_maxdepth=4,)
    trm_fig.update_layout(height = int(width*1.5), width = width)
    return {
        "df_res": df_res, 
        "trm_fig": trm_fig, 
        "text_data": text_data}

all_data = {
    'lex': prep_data('lex'),
    'tr8': prep_data('tr8'),
}

######## NAVBAR ########
nav = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("About", active="exact", href="/")),
        dbc.NavItem(dbc.NavLink("Transcripts", active="exact", href="/tr8")),
        dbc.NavItem(dbc.NavLink("Lex Fridman", active="exact", href="/lex")),
    ],
    brand="Visual Storytelling",
    brand_href="#",
    color="dark",
    dark=True,
)
content = html.Div(id="page-content")

######## LAYOUT ########
app.layout = html.Div([
    dcc.Location(id="url"),
    nav,
    content,
    dcc.Store(id='intermediate-text-data'),
    dcc.Store(id='intermediate-df-res'),
])

######## CALLBACK: RENDER PAGES ########
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        html.Div([
            html.P("This is the content of the home page!"),
            html.P("", id="out"),
            html.P("", id="treemap"),       
        ])
        return html.P("This is the content of the home page!")
    elif pathname == "/tr8":
        collection_name = 'tr8'
        return create_analysis_view(collection_name, all_data[collection_name]['trm_fig'])
    elif pathname == "/lex":
        collection_name = 'lex'
        return create_analysis_view(collection_name, all_data[collection_name]['trm_fig'])
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

######## CALLBACK: DATA PREP & STORE IN BROWSER ########
@callback([Output('intermediate-text-data', 'data'),
              Output('intermediate-df-res', 'data'),
           ], Input("url", "pathname"))
def clean_data(pathname):
    text_data = all_data[pathname.replace('/','')]['text_data']
    df_res = all_data[pathname.replace('/','')]['df_res']
    return text_data, df_res.to_json(date_format='iso', orient='split')

######## CALLBACK: TRACK SELECTION ########
@callback(
    Output("out", "children"),
    Input("treemap", "clickData"))
def first_callback(selected_data):
    if selected_data is None:
        return ".."
    else:
        return json.dumps(selected_data)

######## CALLBACK: SHOW CARDS ########
@callback(
    Output("ls_cards", "children"),
    [Input("treemap", "clickData"),
     Input('intermediate-text-data', 'data'),
     Input('intermediate-df-res', 'data')])
def show_cards(selected_data, text_data, jsonified_cleaned_data):
    df_res = pd.read_json(StringIO(jsonified_cleaned_data), orient='split')
    if selected_data is None:
        return []
    else:
        select_id = selected_data['points'][0]['id']
        label = selected_data['points'][0]['label']
        cluster_members = df_res[df_res['id'] == select_id]\
            .to_dict(orient='records')[0]['cluster_members']
        def list_ents(item):
            #[t['named_entity'] for t in text_data[mmb_id]['mentioned_entities']]
            ent_list = eval(item['mentioned_entities'])
            if len(ent_list) == 0:
                return []
            return [
                dbc.Badge(f"{ent['named_entity']}", color="dark", className="me-1")
                for ent in ent_list
            ]
        return [
            dbc.Card(
                [
                    dbc.CardHeader(text_data[mmb_id]['summary_title']),
                    dbc.CardHeader(text_data[mmb_id]['topic_tag']),
                    dbc.CardBody(
                        dbc.ListGroup(
                            [dbc.ListGroupItem(b) for b in text_data[mmb_id]["summary_bullets"].split('\n')]
                        ),
                    ),
                    dbc.CardFooter(text_data[mmb_id]['nickname']),
                    dbc.CardFooter(list_ents(text_data[mmb_id])),
                ],
                style={"width": "18rem"},
            )
            for mmb_id in cluster_members
        ]
    
import smtplib, ssl
@app.callback(Output('div-button', 'children'),
     Input("button-submit", 'n_clicks')
     ,Input("example-email-row", 'value')
     ,Input("example-name-row", 'value')
     ,Input("example-message-row", 'value')
    )
def submit_message(n, email, name, message):
    
    port = 465  # For SSL
    sender_email = email
    receiver_email = '<your email address here>'
      
    # Create a secure SSL context
    context = ssl.create_default_context()       
    
    if n > 0:
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login("<you email address here>", '<you email password here>')
            server.sendmail(sender_email, receiver_email, message)
            server.quit()
        return [html.P("Message Sent")]
    else:
        return[dbc.Button('Submit', color = 'primary', id='button-submit', n_clicks=0)]

if __name__ == '__main__':
    app.run(debug=True)
