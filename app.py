from dash import Dash, html, dcc, Input, Output, callback, State
import pandas as pd
import json
import os
import numpy as np
from io import StringIO
from picture_text.picture_text import PictureText
from picture_text.src.treemap import build_sunburst, build_tree_map
from picture_text.src.explainers import ABOUT, SAMPLE_DETAILS
from picture_text.src.feedback_form import contact_form
import dash_bootstrap_components as dbc
import smtplib, ssl

model = 'gpt4'
extract_schema = 'summary_entity1'
emb_model_name = 'oAI-3s'
root_path = os.environ.get('VST_SAMPLE_DATA','./sample_data')
treemap_width = 400
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
        dbc.Button("Collapse All", color="dark", id="collapse-all-button", n_clicks=0),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Container(
                        children=[dbc.Spinner(
                        children = [html.Div(
                            id='ls_cards',
                    )],
                    )]), 
                    width=7),
                dbc.Col(  
                    html.Div(  
                        [dcc.Graph(
                            id = 'treemap',
                            figure = trm_fig
                        ),
                        contact_form(width=treemap_width),
                        ]
                        +([html.P(id='out')] if test > 0 else []),
                    )
                , width=5),
            ]
        ),
    ])


def prep_data(collection_name, chart_type, width = treemap_width):
    save_to = os.path.join(root_path,f'topic_n_ent_{collection_name}_{model}_{extract_schema}_{emb_model_name}.json')
    
    text_data = json.load(open(save_to,'r'))

    if test > 0:
        text_data = text_data[:test]

    txt_embeddings = [td['embedding'] for td in text_data]
    txt = [f"{td['topic_tag']}" for td in text_data]
    txt_file = [f"{td['file']}" for td in text_data]
    tag_to_file = {txt[i]:txt_file[i] for i in range(len(txt))}
    pt = PictureText(txt)
    pt(txt_embeddings=txt_embeddings,encoder=None,hac_method='ward', hac_metric='euclidean')
    df_res = pt.hac_to_treemap(pt.linkage_table, 
                   depth=4, 
                   nr_splits=3, 
                   min_size=0.1,
                   max_extension=1,)
    df_res['labels'], df_res['score']= zip(*df_res.apply(lambda x: \
            pt.cluster_summary_simple([np.array(txt[m]) for m in x['cluster_members']], \
                                [np.array(txt_embeddings[m]) for m in x['cluster_members']]), axis=1))
    df_res['tag_file'] = df_res['labels'].apply(lambda x: tag_to_file.get(x,x))
    color_discrete_map={'(?)':'black'}
    nickname_colors = {
        "392_bach": "gold", "398_zuck": "blue", "darkblue": "Musk", "405_bezos": "grey",
        "416_lecun": "orange", "419_sama": "red", "ADSK_Q4": "gold", "BBY_Q4": "blue",
        "BUD_Q4": "darkblue", "CRM_Q4": "grey", "DOCU_Q4": "orange", "JWN_Q4": "green",
        "KR_Q4": "red", "SNOW_Q4": "purple",
    }
    color_discrete_map={**color_discrete_map, **nickname_colors}
    df_res['tag_color'] = df_res['tag_file'].apply(lambda x: color_discrete_map.get(x,'black'))
    if chart_type == 'treemap':
        trm_fig = build_tree_map(df_res)
    elif chart_type == 'sunburst':
        trm_fig = build_sunburst(df_res)
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
    print('Finished prepping', collection_name, ':::', chart_type, ':::', len(text_data),text_data[0].keys())
    return {
        "df_res": df_res, 
        "trm_fig": trm_fig, 
        "text_data": text_data}

all_data = {
    'lex-sunburst': prep_data('lex','sunburst'),
    'tr8-sunburst': prep_data('tr8','sunburst'),
    'lex-treemap': prep_data('lex','treemap'),
    'tr8-treemap': prep_data('tr8','treemap')
}

######## NAVBAR ########
nav = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("About", active="exact", href="/")),
        dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem("Treemap", href="/tr8-treemap"), 
                dbc.DropdownMenuItem("Sunburst", href="/tr8-sunburst"),
                ],
            label="8 Transcripts",
            in_navbar=True,
            nav=True,
        ),
        dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem("Treemap", href="/lex-treemap"),
                dbc.DropdownMenuItem("Sunburst", href="/lex-sunburst"),
                ],
            label="6 Lex Fridman AI Podcasts",
            in_navbar=True,
            nav=True,
        ),
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
    elif pathname in ["/tr8-treemap",'/lex-treemap','/tr8-sunburst','/lex-sunburst']:
        collection_name = pathname.replace('/','').split('-')[0]
        collection_key = pathname.replace('/','')
        return create_analysis_view(collection_name, all_data[collection_key]['trm_fig'])
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )
"""
######## CALLBACK: DATA PREP & STORE IN BROWSER ########
@callback([Output('intermediate-text-data', 'data'),
              Output('intermediate-df-res', 'data'),
           ], Input("url", "pathname"))
def clean_data(pathname):

    if pathname == "/":
        return None, None
    text_data = all_data[pathname.replace('/','').split('-')[0]]['text_data']
    df_res = all_data[pathname.replace('/','').split('-')[0]]['df_res']
    return text_data, df_res.to_json(date_format='iso', orient='split')"""

######## CALLBACK: TRACK SELECTION ########
@callback(
    Output("out", "children"),
    Input("treemap", "clickData"))
def first_callback(selected_data):
    if selected_data is None:
        return ".."
    else:
        return json.dumps(selected_data)

def list_ents(item):
    #[t['named_entity'] for t in text_data[mmb_id]['mentioned_entities']]
    ent_list = eval(item['mentioned_entities'])
    if len(ent_list) == 0:
        return []
    return [
        dbc.Badge(f"{ent['named_entity']}", color="dark", className="me-1")
        for ent in ent_list
    ]

######## CALLBACK: SHOW CARDS ########
@callback(
    Output("ls_cards", "children"),
    [Input("treemap", "clickData"),
     #Input('intermediate-text-data', 'data'),
     #Input('intermediate-df-res', 'data'),
     Input("url", "pathname"),
     Input("collapse-all-button", "n_clicks")],)
def show_cards(selected_data, pathname, nr_clicks):
    cluster_members = []
    current_path = 'Full/'
    text_data = all_data[pathname.replace('/','')]['text_data']
    df_res = all_data[pathname.replace('/','')]['df_res']
    #if text_data is None or jsonified_cleaned_data is None:
    #    list_cards = []
    #df_res = pd.read_json(StringIO(jsonified_cleaned_data), orient='split')
    if selected_data is None:
        cluster_members = list(range(min(50,df_res.shape[0])))
    elif not 'id' in selected_data['points'][0]:
        cluster_members = list(range(min(50,df_res.shape[0])))
    else:
        select_id = selected_data['points'][0]['id']
        current_path = selected_data['points'][0]['currentPath']
        cluster_members = df_res[df_res['id'] == select_id]\
            .to_dict(orient='records')[0]['cluster_members']
    def make_card(mmb_id, nr_clicks):
        if nr_clicks % 2:
            card = dbc.Card(
                dbc.CardHeader(f'{text_data[mmb_id]["nickname"]}#{mmb_id} Heading: {text_data[mmb_id]["summary_title"]}'),
                style = {"width": "24rem", 'margin': '10px'},
            )
        else:
            card = dbc.Card(
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
                dbc.Button(f"Read full ({len(text_data[mmb_id]['topic_text'].split())} words)", 
                            color="dark", id=f"btn-{mmb_id}"),
                dbc.Popover(
                    [
                        dbc.PopoverHeader(text_data[mmb_id]['summary_title']),
                        dbc.PopoverBody(text_data[mmb_id]['topic_text']),
                    ],
                    target=f"btn-{mmb_id}",
                    placement="bottom-end",
                    trigger="click",
                    style={"maxWidth": "80%"},
                ),
            ],
            style = {"width": "24rem", 'margin': '10px'},
        )
        return card
    list_cards = [
        make_card(mmb_id,nr_clicks)
        for mmb_id in cluster_members
    ]
    return [
        html.P(children=f'Showing: {len(cluster_members)} items, current path {current_path}'),
        dbc.Row(list_cards, justify="evenly",)
    ]

######## CALLBACK: SEND EMAIL ########
@app.callback(Output('div-button', 'children'),
     Input("button-submit", 'n_clicks'),
     Input("example-email-row", 'value'),
     Input("example-name-row", 'value'),
     Input("example-message-row", 'value')
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
