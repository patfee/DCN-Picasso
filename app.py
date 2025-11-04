from dash import Input, Output, callback

@callback(
    Output("page1-container", "style"),
    Output("page2-container", "style"),
    Output("page3-container", "style"),
    Input("url", "pathname"),
)
def route(pathname):
    if pathname in (None, "/", ""):
        pathname = "/page1"
    show, hide = {"display": "block"}, {"display": "none"}
    if pathname.startswith("/page1"):
        return show, hide, hide
    if pathname.startswith("/page2"):
        return hide, show, hide
    if pathname.startswith("/page3"):
        return hide, hide, show
    return show, hide, hide
