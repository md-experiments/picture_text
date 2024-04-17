from dash import Dash, html, dcc

import pandas as pd
import json
import os
import numpy as np
from io import StringIO
from picture_text.picture_text import PictureText
from picture_text.src.utils import flatten_list, hash_text
from dash import Dash, html, Input, Output, callback
import dash_bootstrap_components as dbc

#app = Dash(__name__)
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
model = 'gpt4'
extract_schema = 'summary_entity1'

emb_model_name = 'oAI-3s'
collection_name = 'sama'
collection_name = 'lex'
collection_name = 'tr8'
#collection_name = 'tr_snow'
nicknames = {
    "392_bach": "J Bach",
    "398_zuck": "Zuck",
    "400_musk": "Musk",
    "405_bezos": "Bezos",
    "416_lecun": "LeCun",
    "419_sama": "SamA",
    "ADSK_Q4": "ADSK",
    "BBY_Q4": "BBY",
    "BUD_Q4": "BUD",
    "CRM_Q4": "CRM",
    "DOCU_Q4": "DOCU",
    "JWN_Q4": "JWN",
    "KR_Q4": "KR",
    "SNOW_Q4": "SNOW",
}


def add_embedding(tpx, path, emb_model_name):
    txt_to_emb = tpx['topic_tag'] + ": " + tpx['topic_description']  
    emb_models = {
        'oAI-3s':'text-embedding-3-small',
    }
    hs_txt = hash_text(txt_to_emb)
    save_to = os.path.join(path,f'emb_{emb_model_name}_{hs_txt}.json')
    if os.path.exists(save_to):
        js_payload = json.load(open(save_to,'r'))
        return js_payload['embedding']
    else:
        print('Missing',save_to)
        return None
"""
path_p3 = os.path.join('../cor-ai-flask/data','collections',collection_name,'p3_embeddings')
path_p2 = os.path.join('../cor-ai-flask/data','collections',collection_name,'p2_merged')

path_read = os.path.join(path_p2,f'{collection_name}_{model}_{extract_schema}.csv')
print("Reading", path_read)

df = pd.read_csv(path_read).fillna('')
try:
    df.to_csv(f'df_{collection_name}.csv',index=False)
except:
    pass

df['summary_topics'] = df['summary_topics'].apply(lambda x: eval(x) if x != '' else [])
df = df.explode('summary_topics').reset_index(drop=True)
df['nickname'] = df['file'].apply(lambda x: nicknames.get(x,x))

df = pd.concat([
    df.drop(['summary_topics'], axis=1), 
    df['summary_topics'].apply(pd.Series)], axis=1).copy()
df['embedding'] = df.apply(lambda x: add_embedding(x, path_p3, emb_model_name), axis=1)
from picture_text.src.utils import makedirs
text_data = df.to_dict(orient='records')
makedirs(['./data'])
json.dump(text_data,open(f'./data/{collection_name}_{model}_{extract_schema}_text_data.json','w'))
"""
def prep_data(collection_name):
    text_data = json.load(open(f'./data/{collection_name}_{model}_{extract_schema}_text_data.json','r'))
    txt_embeddings = [td['embedding'] for td in text_data]
    txt = [f"{td['topic_tag']}" for td in text_data]
    pt = PictureText(txt)
    pt(txt_embeddings=txt_embeddings,encoder=None,hac_method='ward', hac_metric='euclidean')
    df_res, trm_fig = pt.make_picture(layer_depth = 4,             
                        layer_min_size = 0.1,
                        layer_max_extension = 1,
                        treemap_average_score = None, 
                        treemap_maxdepth=4,)
    return {
        "df_res": df_res, 
        "trm_fig": trm_fig, 
        "text_data": text_data}

all_data = {
    'lex': prep_data('lex'),
    'tr8': prep_data('tr8'),
}

nav = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("About", active="exact", href="/")),
        dbc.NavItem(dbc.NavLink("Transcripts", active="exact", href="/tr8")),
        dbc.NavItem(dbc.NavLink("Lex Fridman", active="exact", href="/lex")),
    ],
    brand="NavbarSimple",
    brand_href="#",
    color="dark",
    dark=True,
)
content = html.Div(id="page-content")

app.layout = html.Div([
    dcc.Location(id="url"),
    nav,
    content,
    dcc.Store(id='intermediate-text-data'),
    dcc.Store(id='intermediate-df-res'),
])

@callback([Output('intermediate-text-data', 'data'),
              Output('intermediate-df-res', 'data'),
           ], Input("url", "pathname"))
def clean_data(pathname):
     # some expensive data processing step
    text_data = all_data[pathname.replace('/','')]['text_data']
    df_res = all_data[pathname.replace('/','')]['df_res']

     # more generally, this line would be
     # json.dumps(cleaned_df)
    return text_data, df_res.to_json(date_format='iso', orient='split')

def analyse_data(collection_name, trm_fig):
    width = 400
    trm_fig.update_layout(height = int(width*1.5), width = width)
    return html.Div([
        html.H1(children=f'Analysing Collection: {collection_name}', style={'textAlign':'center'}),
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
                        html.P(id='err', style={'color': 'red'}),
                        html.P(id='out')]
                    )
                , width=5),
            ]
        ),
    ])


@callback(
    Output("out", "children"),
    Input("treemap", "clickData"))
def first_callback(selected_data):
    return json.dumps(selected_data)



@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("This is the content of the home page!")
    elif pathname == "/tr8":
        collection_name = 'tr8'
        return analyse_data(collection_name, all_data[collection_name]['trm_fig'])
    elif pathname == "/lex":
        collection_name = 'lex'
        return analyse_data(collection_name, all_data[collection_name]['trm_fig'])
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

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
        cluster_members = df_res[df_res['id'] == select_id].to_dict(orient='records')[0]['cluster_members']

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
                    dbc.CardFooter(text_data[mmb_id]['file']),
                ],
                style={"width": "18rem"},
            )
            for mmb_id in cluster_members
        ]

if __name__ == '__main__':
    app.run(debug=True)
