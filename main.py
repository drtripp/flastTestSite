from flask import Flask, render_template
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

app = Flask(__name__)
state_data = pd.read_csv("static/state_statistics.csv")
state = state_data.pop('State')
cols = state_data.columns

@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/about')
def aboutPage():
    return render_template('about.html')

@app.route('/state_statistics/<int:x>_<int:y>')
def stateStats(x, y):
    xcol = x
    ycol = y
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=state_data[cols[xcol]], y=state_data[cols[ycol]], mode="markers+text", text=state))
    fig.update_layout(title=cols[ycol] + ' -vs- ' + cols[xcol], showlegend=False)
    fig.update_xaxes(title_text=cols[xcol])
    fig.update_yaxes(title_text=cols[ycol])

    model = LinearRegression()
    model.fit(np.array(state_data[cols[xcol]]).reshape(-1, 1), np.array(state_data[cols[ycol]]))
    corr = np.sqrt(model.score(np.array(state_data[cols[xcol]]).reshape(-1, 1), np.array(state_data[cols[ycol]])))

    fig.add_trace(go.Scatter(x=state_data[cols[xcol]], y=model.predict(np.array(state_data[cols[xcol]]).reshape(-1, 1))))

    return render_template("state_statistics.html", plot = fig.to_html(), cols ={i:col for i, col in enumerate(cols)},
                           x = xcol, y = ycol, r = np.round(corr, decimals=3), rsq = np.round(corr**2, decimals=3))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)
