from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

email_input = dbc.Row([
        dbc.Label("Email"
                , html_for="example-email-row"
                , width=2),
        dbc.Col(dbc.Input(
                type="email"
                , id="example-email-row"
                , placeholder="Enter email"
            ),width=10,
        )],className="mb-3"
)
user_input = dbc.Row([
        dbc.Label("Name", html_for="example-name-row", width=2),
        dbc.Col(
            dbc.Input(
                type="text"
                , id="example-name-row"
                , placeholder="Enter name"
                , maxLength = 80
            ),width=10
        )], className="mb-3"
)
message = dbc.Row([
        dbc.Label("Message"
         , html_for="example-message-row", width=2)
        ,dbc.Col(
            dbc.Textarea(id = "example-message-row"
                , className="mb-3"
                , placeholder="Enter message"
                , required = True)
            , width=10)
        ], className="mb-3")

def contact_form():
    markdown = ''' 
    Send a message if you have a comment, question,
    or concern. Thank you!'''   
    form = html.Div([ dbc.Container([
            dcc.Markdown(markdown)
            , html.Br()
            , dbc.Card(
                dbc.CardBody([
                     dbc.Form([email_input
                        , user_input
                        , message])
                ,html.Div(id = 'div-button', children = [
                    dbc.Button('Submit'
                    , color = 'primary'
                    , id='button-submit'
                    , n_clicks=0)
                ]) #end div
                ]) #end cardbody
            )#end card
            , html.Br()
            , html.Br()
        ])
        ])
    
    return form
