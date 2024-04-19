from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

email_input = dbc.Row([
        dbc.Label("Email"
                , html_for="example-email-row"
                , width=2),
        dbc.Col(dbc.Input(
                type="email"
                , id="example-email-row"
                , placeholder="Enter email (optional)"
            ),width=10,
        )],
        #className="mb-3"
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
        )], 
        #className="mb-3"
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
        ], 
        #className="mb-3"
        )

def contact_form(width):
    markdown = ''' 
    Please share your feedback:
    - How would you use this?
    - What did you like?
    - What is missing?
    '''   
    form = html.Div([ dbc.Container([
            html.Br(), 
            dbc.Card(
                dbc.CardBody([
                    dcc.Markdown(markdown),
                     dbc.Form([email_input, user_input, message])
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
        ], style = {'width': width})
    
    return form
