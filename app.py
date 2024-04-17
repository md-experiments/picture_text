from dash import Dash, html, dcc

import pandas as pd
import json
import os
import numpy as np

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
path_p3 = os.path.join('../cor-ai-flask/data','collections',collection_name,'p3_embeddings')
path_p2 = os.path.join('../cor-ai-flask/data','collections',collection_name,'p2_merged')

path_read = os.path.join(path_p2,f'{collection_name}_{model}_{extract_schema}.csv')
print("Reading", path_read)
df = pd.read_csv(path_read).fillna('')
try:
    df.to_csv(f'df_{collection_name}.csv',index=False)
except:
    pass
ls_topics = []
for tpx, fl_nm in zip(df['summary_topics'].values,df['file'].values):
    if tpx != '':
        tp_ls = eval(tpx)
        for t in tp_ls:
            t['nickname'] = nicknames.get(fl_nm, "Unknown")
        # NB: I only allow one topic per document
        tp_ls = tp_ls[0]
        ls_topics.append(tp_ls)

#ls_topics = flatten_list(ls_topics)

def store_embedding(text, path, emb_model_name):    
    emb_models = {
        'oAI-3s':'text-embedding-3-small',
    }
    hs_txt = hash_text(text)
    save_to = os.path.join(path,f'emb_{emb_model_name}_{hs_txt}.json')
    if os.path.exists(save_to):
        js_payload = json.load(open(save_to,'r'))
        return js_payload
    else:
        print('Missing',save_to)
        return None
    
text_data = []
for tpx in ls_topics:
    txt_to_emb = tpx['topic_tag'] + ": " + tpx['topic_description']
    js_payload = store_embedding(txt_to_emb, path_p3, emb_model_name)
    if js_payload is None:
        continue
    js_payload['topic_title'] = tpx['topic_tag']
    js_payload['nickname'] = tpx['nickname']
    text_data.append(js_payload)

txt_embeddings = [td['embedding'] for td in text_data]
#txt_embeddings = np.array(txt_embeddings)
txt = [f"[{td['nickname']}]: {td['topic_title']}" for td in text_data]
txt = [f"{td['topic_title']}" for td in text_data]
pt = PictureText(txt)
pt(txt_embeddings=txt_embeddings,encoder=None,hac_method='ward', hac_metric='euclidean')
df_res, trm_fig = pt.make_picture(layer_depth = 4,             
                    layer_min_size = 0.1,
                    layer_max_extension = 1,
                    treemap_average_score = None, 
                    treemap_maxdepth=4,)

width = 400
trm_fig.update_layout(height = int(width*1.5), width = width)
try:
    df_res.to_csv(f'df_res_{collection_name}.csv',index=False)
except:
    pass
app.layout = html.Div([
    html.H1(children=F'Analysing Collection: {collection_name}', style={'textAlign':'center'}),
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

@callback(
    Output("ls_cards", "children"),
    Input("treemap", "clickData"))
def show_cards(selected_data):
    if selected_data is None:
        return []
    else:
        select_id = selected_data['points'][0]['id']
        label = selected_data['points'][0]['label']
        cluster_members = df_res[df_res['id'] == select_id].to_dict(orient='records')[0]['cluster_members']
        cluster_members_content = df.loc[cluster_members].to_dict(orient='records')
        return [
            dbc.Card(
                [dbc.CardHeader(mem['summary_title']),
                dbc.CardBody(
                    dbc.ListGroup(
                        [dbc.ListGroupItem(b) for b in mem["summary_bullets"].split('\n')]
                    ),
                ),
                dbc.CardFooter(mem['file']),
                ],
                style={"width": "18rem"},
            )
            for mem in cluster_members_content
        ]

if __name__ == '__main__':
    app.run(debug=True)
