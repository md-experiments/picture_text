from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import json
import os
import numpy as np
from io import StringIO
from picture_text.picture_text import PictureText
from picture_text.src.explainers import ABOUT, SAMPLE_DETAILS
from picture_text.src.feedback_form import contact_form
import dash_bootstrap_components as dbc
import smtplib, ssl

model = 'gpt4'
extract_schema = 'summary_entity1'
emb_model_name = 'oAI-3s'
root_path = os.environ.get('VST_SAMPLE_DATA','./sample_data')

test = int(os.environ.get("VST_TEST",-1))

#app = Dash(__name__)
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def create_analysis_view(collection_name, trm_fig):
    """This is the view screen of each page. 
    It shows the treemap and the cards of the selected cluster."""
    if test > 0:
        test_str = f" [TESTING: {test}]"
    else:
        test_str = ""
    return html.Div([
        html.H1(children=f'{test_str} Analysing: {SAMPLE_DETAILS[collection_name]["title"]}'),
        html.P(children=f'Motivation: {SAMPLE_DETAILS[collection_name]["motivation"]}'),
        html.P(children=f'Details: {SAMPLE_DETAILS[collection_name]["details"]}'),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Container(
                        children=[dbc.Spinner(
                        children = [dbc.Row(
                            id='ls_cards',
                            justify="evenly",
                    )],
                    )]), 
                    width=7),
                dbc.Col(  
                    html.Div(  
                        [dcc.Graph(
                            id = 'treemap',
                            figure = trm_fig
                        ),
                        contact_form(),
                        ]
                        +([html.P(id='out')] if test > 0 else []),
                    )
                , width=5),
            ]
        ),
    ])


def prep_data(collection_name, width = 500):
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
    try:
        save_csv = os.path.join(root_path,f'df_res_{collection_name}_{model}_{extract_schema}_{emb_model_name}.csv')
        df_res.to_csv(save_csv)
    except: 
        pass
    del txt_embeddings
    del txt
    for e in text_data:
        del e['embedding']
    print('Prepped data for', collection_name)
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
    brand_href="/",
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
        return html.Div([
            dcc.Markdown(ABOUT),
            html.P("", id="out"),
            html.P("", id="treemap"),       
        ])
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

    if pathname == "/":
        return None, None
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
    if text_data is None or jsonified_cleaned_data is None:
        return []
    df_res = pd.read_json(StringIO(jsonified_cleaned_data), orient='split')
    if selected_data is None:
        return []
    else:
        select_id = selected_data['points'][0]['id']
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
                    dbc.CardHeader(f'Item: {mmb_id} Heading: ' + text_data[mmb_id]['summary_title']),
                    dbc.CardHeader('Tag: ' + text_data[mmb_id]['topic_tag']),
                    dbc.CardBody(
                        dbc.ListGroup(
                            [dbc.ListGroupItem(b) for b in text_data[mmb_id]["summary_bullets"].split('\n')]
                        ),
                    ),
                    dbc.CardFooter('Source: ' + text_data[mmb_id]['nickname']),
                    dbc.CardFooter(list_ents(text_data[mmb_id])),
                ],
                #style={"width": "24rem"},
                style = {'margin': '10px'}
            )
            for mmb_id in cluster_members
        ]
    
######## CALLBACK: SEND EMAIL ########
@app.callback(Output('div-button', 'children'),
     Input("button-submit", 'n_clicks')
     ,Input("example-email-row", 'value')
     ,Input("example-name-row", 'value')
     ,Input("example-message-row", 'value')
    )
def submit_message(n, email, name, message):
    if os.path.exists('email.key'):
        with open('email.key','r') as f:
            receiver_email, receiver_pass = f.readlines()
    else:
        receiver_email = os.environ.get('VST_EMAIL','<your email address here>')
        receiver_pass = os.environ.get('VST_PASS','<your email password here>')
    port = 465  # For SSL
    # Create a secure SSL context
    context = ssl.create_default_context()       
    msg = f'''\
From: {email}
Subject: Feedback: {name}

EMAIL:
{email}

NAME:
{name}

MESSAGE:
{message} '''
    if n > 0:
        if (email is None) or (name is None) or (message is None):
            return [dbc.Alert("Please fill in all fields", color="secondary"),
                    dbc.Button('Submit', color = 'dark', id='button-submit', n_clicks=0)]
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(receiver_email, receiver_pass)
            server.sendmail(email, receiver_email, msg)
            server.quit()
        return [html.P("Message Sent")]
    else:
        return [dbc.Button('Submit', color = 'dark', id='button-submit', n_clicks=0)]

if __name__ == '__main__':
    app.run(debug=True)
