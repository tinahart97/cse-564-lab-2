from sklearn.preprocessing import scale
from sklearn import decomposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import MDS

app = dash.Dash(__name__)
df = pd.read_csv(r"\Users\tinah\Desktop\HW2\NASCAR.csv")

df2= pd.read_csv(r"\Users\tinah\Desktop\HW2\NASCAR.csv")
del df['Year']
del df['Driver']
del df['Car']
del df2['Driver']
del df2['Car']
df.reset_index(inplace=True)
del df['index']
X1 = df.values



X = scale(X1)
pca = decomposition.PCA(n_components=18)
pca.fit(X)
#scores = pca.transform(X)
#loadings = pca.components_.T

#explainedvariance = pca.explained_variance_ratio_
#cumulativevariance = np.cumsum(explainedvariance)

pc_df = pd.DataFrame(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17'], columns=['PC'])


corrMatrix=df.corr().abs()
heatmap = px.imshow(corrMatrix, zmin=-1)
#heatmap.show()


#################### ELBOW TO SEE CLUSTER AMOUNT ######################
SS = []
for i in range(1,17):
        kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
        kmeans.fit(df)
        SS.append(kmeans.inertia_)
plt.plot(range(1, 17), SS,"-o")
plt.fill_between(range(1, 17), SS, alpha=0.2)
#plt.show()

########## I chose to pick 5 using this

####################################################


####################################################
############### HOW TO FIND WHICH ATTRIBUTES ARE HIGHLY CORRELATED!

corrMatrix=df2.corr().abs()
corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1)

already_in = set()
result = []
for col in corrMatrix:
    highlycorr = corrMatrix[col][corrMatrix[col] > .7].index.tolist()
    if highlycorr and col not in already_in:
        already_in.update(set(highlycorr))
        highlycorr.append(col)
        result.append(highlycorr)

print(result)
df1= df[['WINS', 'Avg Start', 'Avg Mid Race', 'Avg Finish', 'Avg Pos',
       'Pass Diff', 'GF passes', 'GF passed', 'Quality Passes',
       '% Quality Passes', '# Fast laps', 'Laps Top 15', '% Laps Top 15',
       'Laps Lead', '% Laps Lead', 'Total Laps', 'Driver Rating', 'Points']]


#heatmap = px.imshow(corrMatrix, zmin=-1)

#heatmap.show()

####################################################

app.layout = html.Div([

    html.Div([
        html.H2("NASCAR Drivers Standings"),
        html.Img(src="/assets/NASCAR.png")

    ], className="banner"),

    html.Label(['Please click a point on the line below to determine the % variance that you would like to account for in the Scatterplot Matrix:'],
                   style={'font-weight': 'bold', "text-align": "left", 'color': '#3E45BD', 'fontSize': 20}, className='Twelve columns'),

    html.Label(['Scree plot'],
                   style={'font-weight': 'bold', "text-align": "left", 'color': '#3E45BD', 'fontSize': 20}, className='Twelve columns'),
    html.Div([
        dcc.Graph(id='Scree', figure={}, clickData=None, hoverData=None, config={'staticPlot': False,'scrollZoom': False ,'doubleClick': 'reset','showTips': False,'displayModeBar': True,'watermark': True,},)
    ], className='twelve columns'),


    html.Label(['Scatter Matrix'],
                   style={'font-weight': 'bold', "text-align": "left", 'color': '#3E45BD', 'fontSize': 20}, className='Twelve columns'),

    html.Div([
        dcc.Graph(id='ScatterMat', )
    ], className='twelve columns'),

    html.Label(['Table of highest PCA Loadings'],
                   style={'font-weight': 'bold', "text-align": "left", 'color': '#3E45BD', 'fontSize': 20}, className='Twelve columns'),

    html.Div([
        dcc.Graph(id= "table"),
    ], className='twelve columns'),

    html.Label(['Biplot'],
                   style={'font-weight': 'bold', "text-align": "left", 'color': '#3E45BD', 'fontSize': 20}, className='Twelve columns'),

    html.Div([
        dcc.Graph(id='Biplot', )
    ], className='twelve columns'),

    html.Label(['Parallel Coordinates'],
               style={'font-weight': 'bold', "text-align": "left", 'color': '#3E45BD', 'fontSize': 20},
               className='Twelve columns'),
    html.Div([
        dcc.Graph(id='Parcoord', )
    ], className='twelve columns'),


    html.Label(['MDS of Data'],
               style={'font-weight': 'bold', "text-align": "left", 'color': '#3E45BD', 'fontSize': 20},
               className='Twelve columns'),

    html.Div([
        dcc.Graph(id='MDS1', )
    ], className='twelve columns'),

    html.Label(['MDS of features'],
               style={'font-weight': 'bold', "text-align": "left", 'color': '#3E45BD', 'fontSize': 20},
               className='Twelve columns'),

    html.Div([
        dcc.Graph(id='MDS2', )
    ], className='twelve columns'),

    html.Label(['Parallel Coordinates Extra Credit'],
               style={'font-weight': 'bold', "text-align": "left", 'color': '#3E45BD', 'fontSize': 20},
               className='Twelve columns'),
    html.Div([
        dcc.Graph(id='ParcoordEC', )
    ], className='twelve columns'),

    html.Div([
        dcc.Checklist(id='dropPC',options=[], className='one columns'),
    ]),

])



######################### Scree plot ##############################

@app.callback(
    Output(component_id='Scree', component_property='figure'),
    [Input(component_id='dropPC', component_property='value')]
)

def scree(val):
    pcadf = pd.DataFrame(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17'], columns=['PC'])
    explainedvariancedf = pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance'])
    cumulativevariancedf = pd.DataFrame(np.cumsum(pca.explained_variance_ratio_), columns=['Cumulative Variance'])
    dfexplainedvariance = pd.concat([pcadf, explainedvariancedf, cumulativevariancedf], axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfexplainedvariance['PC'],y=dfexplainedvariance['Cumulative Variance'],marker=dict(size=15, color="#ff0")))
    fig.add_trace(go.Bar(x=dfexplainedvariance['PC'],y=dfexplainedvariance['Explained Variance'], marker=dict(color="#3E45BD")))
    fig.update_xaxes(title_text="Components")
    fig.update_yaxes(title_text="Explained Variance %")
    return fig

######################### Scatter Matrix ##############################

@app.callback(
    Output(component_id='ScatterMat', component_property='figure'),
    [Input(component_id='Scree', component_property='clickData')]
)

def updated(clk_data):
    if clk_data is None:
        pca = decomposition.PCA(n_components=17)
        pca.fit(X)
        loadings = pca.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1)) #Sum of square loadings
        df_loadings = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
                                                      'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17'], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)    #order the sum of square loadings biggest to smallest
        top4 = desending[:4]           #chose the top 4 loadings
        del top4['SSS']  #Removing Sum of square loadings from the dataframe
        first= (top4.index[0])
        second= (top4.index[1])
        third= (top4.index[2])
        fourth= (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans= KMeans(n_clusters=5)
        kmeansfitval= kmeans.fit_predict(dfforscatmat)

        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color= kmeansfitval)
        scatter.update_traces(diagonal_visible=False)


    elif clk_data['points'][0]['x'] == "PC1":
        pca1 = decomposition.PCA(n_components=1)
        pca1.fit(X)
        loadings = pca1.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1'],
                                       index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color= kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC2":
        pca2 = decomposition.PCA(n_components=2)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color= kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC3":
        pca2 = decomposition.PCA(n_components=3)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color= kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC4":
        pca2 = decomposition.PCA(n_components=4)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4"],
                                       index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color= kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC5":
        pca2 = decomposition.PCA(n_components=5)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5"], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible= False)

    elif clk_data['points'][0]['x'] == "PC6":
        pca2 = decomposition.PCA(n_components=6)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6"], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible= False)

    elif clk_data['points'][0]['x'] == "PC7":
        pca2 = decomposition.PCA(n_components=7)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC8":
        pca2 = decomposition.PCA(n_components=8)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC9":
        pca2 = decomposition.PCA(n_components=9)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8" , "PC9"], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC10":
        pca2 = decomposition.PCA(n_components=10)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC11":
        pca2 = decomposition.PCA(n_components=11)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7",
                                                      "PC8", "PC9", "PC10", "PC11"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC12":
        pca2 = decomposition.PCA(n_components=12)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC13":
        pca2 = decomposition.PCA(n_components=13)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12" , "PC13"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC14":
        pca2 = decomposition.PCA(n_components=14)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings,
                                   columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
                                            "PC11", "PC12", "PC13", "PC14"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC15":
        pca2 = decomposition.PCA(n_components=15)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings,
                                   columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
                                            "PC11", "PC12", "PC13", "PC14", "PC15"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC16":
        pca2 = decomposition.PCA(n_components=16)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings,
                                   columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
                                            "PC11", "PC12", "PC13", "PC14", "PC15", "PC16"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    elif clk_data['points'][0]['x'] == "PC17":
        pca2 = decomposition.PCA(n_components=17)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings,
                                   columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
                                            "PC11", "PC12", "PC13", "PC14", "PC15", "PC16", "PC17"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        dfforscatmat = df[[first, second, third, fourth]]
        kmeans = KMeans(n_clusters=5)
        kmeansfitval = kmeans.fit_predict(dfforscatmat)
        scatter = px.scatter_matrix(dfforscatmat, dimensions=[first, second, third, fourth], color=kmeansfitval)
        scatter.update_traces(diagonal_visible=False)

    return scatter



######################### TABLE ##############################
@app.callback(
    Output(component_id='table', component_property='figure'),
    [Input(component_id='Scree', component_property='clickData')]
)

def table(clk_data):

    if clk_data is None:
        pca2 = decomposition.PCA(n_components=17)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings,
                                   columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
                                            'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17'],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        #creating a table of the top 4 attributes based on sum of square loadings and the user interaction
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
                                            'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17',"SSS", "Attributes"],
                font=dict(size=13),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=10)))

    elif clk_data['points'][0]['x'] == "PC1":
        pca2 = decomposition.PCA(n_components=1)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1'], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"]= dict[first, second, third, fourth]
        dfforscatmat = df[[first, second, third, fourth]]

        top4["Attributes"] = [first, second, third, fourth]

        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=["PC1","Attributes"],
                font=dict(size=20),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))


    elif clk_data['points'][0]['x'] == "PC2":
        pca2 = decomposition.PCA(n_components=2)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2"], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]

        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=["PC1", "PC2", "SSS","Attributes"],
                font=dict(size=20),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))

    elif clk_data['points'][0]['x'] == "PC3":
        pca2 = decomposition.PCA(n_components=3)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3"], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=["PC1", "PC2", "PC3","SSS", "Attributes"],
                font=dict(size=20),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))

    elif clk_data['points'][0]['x'] == "PC4":
        pca2 = decomposition.PCA(n_components=4)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4"], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=["PC1", "PC2", "PC3", "PC4" ,"SSS", "Attributes"],
                font=dict(size=20),
                align="left",height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))


    elif clk_data['points'][0]['x'] == "PC5":
        pca2 = decomposition.PCA(n_components=5)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5"], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"]= dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]


        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=["PC1", "PC2", "PC3", "PC4", "PC5","SSS","Attributes"],
                font=dict(size=20),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))


    elif clk_data['points'][0]['x'] == "PC6":
        pca2 = decomposition.PCA(n_components=6)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6"], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"]= dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]


        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "SSS", "Attributes"],
                font=dict(size=20),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))

    elif clk_data['points'][0]['x'] == "PC7":
        pca2 = decomposition.PCA(n_components=7)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6","PC7","SSS", "Attributes"],
                font=dict(size=20),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))

    elif clk_data['points'][0]['x'] == "PC8":
        pca2 = decomposition.PCA(n_components=8)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"], index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        del top4['SSS']
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "Attributes"],
                font=dict(size=20),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))

    elif clk_data['points'][0]['x'] == "PC9":
        pca2 = decomposition.PCA(n_components=9)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]

        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9","SSS", "Attributes"],
                font=dict(size=20),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))

    elif clk_data['points'][0]['x'] == "PC10":
        pca2 = decomposition.PCA(n_components=10)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10","SSS", "Attributes"],
                font=dict(size=20),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))

    elif clk_data['points'][0]['x'] == "PC11":
        pca2 = decomposition.PCA(n_components=11)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11","SSS", "Attributes"],
                font=dict(size=20),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))

    elif clk_data['points'][0]['x'] == "PC12":
        pca2 = decomposition.PCA(n_components=12)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings, columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12","SSS", "Attributes"],
                font=dict(size=20),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))

    elif clk_data['points'][0]['x'] == "PC13":
        pca2 = decomposition.PCA(n_components=13)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings,
                                   columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
                                            "PC11", "PC12", "PC13"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]

        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12", "PC13","SSS",
                        "Attributes"],
                font=dict(size=20),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=15)))

    elif clk_data['points'][0]['x'] == "PC14":
        pca2 = decomposition.PCA(n_components=14)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings,
                                   columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
                                            "PC11", "PC12", "PC13", "PC14"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12", "PC13", "PC14","SSS",
                        "Attributes"],
                font=dict(size=13),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=10)))

    elif clk_data['points'][0]['x'] == "PC15":
        pca2 = decomposition.PCA(n_components=15)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings,
                                   columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
                                            "PC11", "PC12", "PC13", "PC14", "PC15"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12", "PC13", "PC14", "PC15","SSS",
                        "Attributes"],
                font=dict(size=13),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=10)))

    elif clk_data['points'][0]['x'] == "PC16":
        pca2 = decomposition.PCA(n_components=16)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings,
                                   columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
                                            "PC11", "PC12", "PC13", "PC14", "PC15", "PC16"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12", "PC13", "PC14", "PC15", "PC16","SSS",
                        "Attributes"],
                font=dict(size=13),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=10)))

    elif clk_data['points'][0]['x'] == "PC17":
        pca2 = decomposition.PCA(n_components=17)
        pca2.fit(X)
        loadings = pca2.components_.T
        Sumvals = (np.sum(loadings ** 2, axis=1))
        df_loadings = pd.DataFrame(loadings,
                                   columns=['PC1', "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
                                            "PC11", "PC12", "PC13", "PC14", "PC15", "PC16", "PC17"],
                                   index=df.columns)
        df_loadings['SSS'] = Sumvals
        desending = df_loadings.sort_values("SSS", ascending=False)
        top4 = desending[:4]
        first = (top4.index[0])
        second = (top4.index[1])
        third = (top4.index[2])
        fourth = (top4.index[3])
        top4["Attributes"] = dict[first, second, third, fourth]
        top4["Attributes"] = [first, second, third, fourth]

        ## Adding the table
        table = px.scatter()
        table.add_trace(go.Table(
            header=dict(
                values=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
                        'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17',"SSS", "Attributes"],
                font=dict(size=13),
                align="left", height=60
            ),
            cells=dict(
                values=[top4[k].tolist() for k in top4.columns[0:]],
                align="left", height=60, font_size=10)))

    return table




######################### BIPLOT ##############################
@app.callback(
    Output(component_id='Biplot', component_property='figure'),
    [Input(component_id='dropPC', component_property='value')]
)

def biplot(input):
    ##plotting the scores for the biplot
    X = scale(X1)
    #we only want the top 2 components
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    scores1 = pca.transform(X)
    loadings1 = pca.components_.T
    #plotting the scores into a scatterplot
    biplot = px.scatter(scores1, x=0, y=1)
    for i, colnames in enumerate(df.columns):
        #adding the lines to my biplot, have points begin at 0 and going to the first value of my individual loadings as x and the second value of my loadings as y
        biplot.add_shape(type='line',x0=0, y0=0,x1=loadings1[i, 0]*10, y1=loadings1[i, 1]*10)
        ## adding the names to each line at the end to better visualize it
        biplot.add_annotation(x=loadings1[i, 0]*10, y=loadings1[i, 1]*10, ax=0, ay=0,xanchor="center", yanchor="bottom", text= colnames)
        biplot.update_xaxes(title_text="PC1")
        biplot.update_yaxes(title_text="PC2")

    return biplot






@app.callback(
    Output(component_id='Parcoord', component_property='figure'),
    [Input(component_id='dropPC', component_property='value')]
)
def parallelcoordplot(input):
    kmeans = KMeans(n_clusters=5).fit_predict(df2)
    figPC = px.parallel_coordinates(df2,dimensions=['# Fast laps', 'Laps Lead', '% Laps Lead', 'WINS',
                                                    'Avg Mid Race', 'Avg Finish', 'Avg Pos', 'Quality Passes',
                                                    '% Quality Passes', 'Laps Top 15', '% Laps Top 15',
                                                    'Driver Rating', 'Avg Start','GF passed', 'Quality Passes',
                                                    'Total Laps', 'GF passes', 'Points', 'Pass Diff'], height=1000,color=kmeans)
    return figPC






@app.callback(
    Output(component_id='MDS1', component_property='figure'),
    [Input(component_id='dropPC', component_property='value')]
)
def MDS1(input):
    mdsA = MDS(n_components=2, random_state=0, dissimilarity='euclidean')
    mdsEu = mdsA.fit_transform(df)
    mdsEu = pd.DataFrame(mdsEu, columns=['Dim 1', 'Dim 2'])
    kmeans = KMeans(n_clusters=5).fit_predict(df1)
    mdsdata = px.scatter(mdsEu, x='Dim 1', y='Dim 2', color=kmeans)
    return mdsdata





@app.callback(
    Output(component_id='MDS2', component_property='figure'),
    [Input(component_id='dropPC', component_property='value')]
)
def MDS2(input):
    cor_mat = df.corr().abs()
    dissimilarity = 1- cor_mat
    mdsB = MDS(n_components= 2, random_state=0, dissimilarity='precomputed')
    mdscorr= mdsB.fit_transform(dissimilarity)
    mdscorr = pd.DataFrame(mdscorr, columns=['Dim 1', 'Dim 2'])
    mdsfeat = px.scatter(mdscorr, x='Dim 1', y='Dim 2', text=df.columns)
    mdsfeat.update_traces(textposition='top center')
    return mdsfeat



### ADD BACK CATEGORICAL
@app.callback(
    Output(component_id='ParcoordEC', component_property='figure'),
    [Input(component_id='dropPC', component_property='value')]
)
def parallelcoordplot(input):
    kmeans = KMeans(n_clusters=5)
    kmeansfitval = kmeans.fit_predict(df2)
    figPC = px.parallel_coordinates(df2,dimensions=['# Fast laps', 'Laps Lead', '% Laps Lead', 'WINS',
                                                    'Avg Mid Race', 'Avg Finish', 'Avg Pos', 'Quality Passes',
                                                    '% Quality Passes', 'Laps Top 15', '% Laps Top 15',
                                                    'Driver Rating', 'Avg Start','GF passed', 'Quality Passes',
                                                    'Total Laps', 'GF passes', 'Points', 'Pass Diff'], height=1000,color=kmeansfitval)
    return figPC


if __name__ == '__main__':
    app.run_server(debug=True)