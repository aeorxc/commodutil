import pandas as pd


base_html = """
<!DOCTYPE html>
<html lang="en">
<html>
    <head>
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
      <style>{2}
      </style>
      <title>{1}</title>
    </head>
    <body>
          <p>Last Updated: {0}</p>
          {3}
    </body>
</html>
"""

bast_style = """
* {
  box-sizing: border-box;
}

.box1 {
  float: left;
  width: 100%;
  padding: 1px;
}

.box2 {
  float: left;
  width: 50.00%;
  padding: 1px;
}

.box3 {
  float: left;
  width: 33.33%;
  padding: 1px;
}

.box4 {
  float: left;
  width: 25.00%;
  padding: 1px;
}

.box5 {
  float: left;
  width: 20.00%;
  padding: 1px;
}

.clearfix::after {
  content: "";
  clear: both;
  display: table;
}
"""


def getbasehtml(title, style=base_html, body="Body to replace"):
    global base_html
    html_string = base_html
    html_string = html_string.replace("{0}", pd.datetime.now().strftime('%d %b %Y %H:%M'))
    html_string = html_string.replace("{1}", title)
    html_string = html_string.replace("{2}", style)
    html_string = html_string.replace("{3}", body)

    return html_string