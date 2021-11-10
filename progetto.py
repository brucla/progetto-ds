import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

################################################################

st.set_page_config(
    page_title="Progetto Data Science",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
if 'status' not in st.session_state:
    st.session_state['status'] = 0
status=st.session_state["status"]

with st.sidebar:
    st.markdown("# Indice:")
    if st.button("Pulizia Dati",key="sb_1"): status=1
    if st.button("Studio preliminare",key="sb_2"): status=2
    if st.button("Clustering",key="sb_3"): status=3
    if st.button("Modellizzazione",key="sb_4"): status=4
    b2,b3 = st.columns(2)
    if b2.button("<<"): status=status-1
    if b3.button(">>"): status=status+1

################################################################

st.title("Progetto finale Data Science e Applicazioni in Fisica")
st.header("Studio della correlazione tra l'andamento epidemiologico della Covid-19 in Umbria nel periodo gennaio-giugno 2020 e i parametri di inquinamento atmosferico nei vari comuni.")

st.write("Il seguente report riporta i risultati dello studio di correlazione effettuato confrontando con vari metodi di statistical learning i dati di contagi, ospedalizzazione e morte a causa del virus SARS-COV-2 con i dati di inquinamento atmosferico rilevati nei vari comuni dell'Umbria nel periodo gennaio-giugno 2020.")
st.write("Sono riportati nel dettaglio le singole operazioni di cleaning dei dati, studio preliminare, riduzione della dimensionalità dei dati, clustering e modellizzazione.")


b0,bo,b1,b2,b3,b4,b5,b6 = st.columns(8)
if b1.button("Pulizia Dati"): status=1
if b2.button("Studio Preliminare"): status=2
if b3.button("Clustering"): status=3
if b4.button("Modellizzazione"): status=4


data = pd.read_csv("./data/dataset_exam.csv",index_col=0)

with open("./comuni_umbria.geojson") as comuni_file:
    comuni = json.load(comuni_file)

################################################################
##################################################
#dati_exp=st.expander(label="",expanded=True)
#with dati_exp:
if status==1:
    st.markdown("## I dati")
    """
    I dati relativi alla diffusione del contagio sono resi pubblici dal Servizio Nazionale della Protezione Civile. I dati sull'inquinamento atmosferico provengono invece da bobo bo.\n
    Questi sono stati raccolti in un unico file csv fornitomi per il progetto in cui i dati, organizzati per comune, sono stati uniti a informazioni di carattere geografico (latitudine, longitudine, orografia).
    """
    st.write(data)

    st.markdown("### Valori mancanti")
    """
    Il dataset presenta due valori mancanti.\n
    Uno riguarda la regione relativa al comune di Spello. Ovviamente si parla della regione Umbria.
    """
    data.loc[80,'Region'] = data.loc[79,'Region']
    """
    L'altro riguarda il valore minimo della deviazione standard giornaliera pm10 nell'anno 2019 nel comune di Montecastello di Vibio.\n
    Per ricostruire questo valore potremmo prendere come riferimento i valori analoghi degli altri comuni e stimare quello mancante con una banale media; in alternativa potremmo copiare il valore relativo allo stesso comune nell'anno successivo; o ancora in modo potenzialmente più accurato potremmo sviluppare un modello di previsione sfruttando una eventuale dipendenza dagli altri parametri di inquinamento.\n
    Tuttavia per il nostro studio ignoreremo i dati di inquinamento relativi al 2019. Il modo più semplice di operare è perciò eliminare direttamente le colonne relative a tale periodo.
    """
    data=data.drop(columns=data.columns[25:-2])

    st.markdown("### Valori ridondanti")
    """
    Oltre ai dati relativi al 2019, ci sono nel dataset tre colonne duplicate relative ai dati di inquinamento del periodo gennaio-giugno 2020. Possiamo perciò eliminare anche queste.\n
    Eliminando anche la colonna relativa alla regione, che è scontata per ogni dato, i valori che rimangono sono
    """
    data=data.drop(columns=data.columns[22:25])
    data=data.drop(columns=data.columns[-1])
    st.write(data)
    #st.write(data.columns)

    st.markdown("### Il caso Giove")
    """
    Invece delle coordinate del comune di Giove (TR) sono riportate le coordinate della frazione Giove di Nocera Umbra (PG).\n
    Il dato può essere corretto semplicemente sovrascrivendo con i valori corretti.
    """
    data.loc[35,'lat'] = 42.516667
    data.loc[35,'lng'] = 12.333333

#################################################
#plots_exp=st.expander(label="",expanded=True)
#with plots_exp:
if status==2:
    st.markdown("## Studio preliminare")
    """
    Il grafico seguente riporta sui due assi principali la popolazione totale di ciascun comune e la densità di abitazione dello stesso.
    Il colore dei marker può essere associato a diversi parametri epidemiologici e di inquinamento selezionando dal menu a tendina.
    """

    feature={"CovidCases_jan_jun_2020":         "Casi covid totali",  
            "AvgHospitalized_jan_jun_2020":     "Media pazienti covid ospedalizzati",
            "AvgIntensiveCare_jan_jun_2020":    "Media pazienti covid in terapia intensiva",
            "Deceased_jan_jun_2020":            "Decessi di pazienti covid totali",
            "max_pm10_ug/m3_mean_jan_jun_2020": "Massimo dei valori medi giornalieri di pm10",
            "min_pm10_ug/m3_mean_jan_jun_2020": "Minimo dei valori medi giornalieri di pm10",
            "mean_pm10_ug/m3_mean_jan_jun_2020":"Media dei valori medi giornalieri di pm10",
            "mean_pm10_ug/m3_std_jan_jun_2020": "Media delle std dei valori giornalieri di pm10",
            "mean_pm10_ug/m3_median_jan_jun_2020":"Media dei valori mediani giornalieri di pm10",
            "Zone":                             "Zona"}
    st.write("")
    col1, col2 = st.columns(2)
    col1.write("Seleziona terza feature:")
    choice_name = col1.selectbox("",feature.values())
    choice = data.loc[:,list(feature.keys())[list(feature.values()).index(choice_name)]]
    
    col2.write("Applica scala logaritmica:")
    logX = col2.checkbox("Asse x")
    logY = col2.checkbox("Asse y")
    if choice_name!="Zona":
        if st.checkbox("ogni 100 abitanti"):
            choice = 100*choice/data.Population
        if st.checkbox("Normalizza y"):
            choice=choice/choice.max()
    choice=choice.rename_axis("z")

    col1, col2 = st.columns(2)
    col1.write(px.scatter(data,
                x="Population",
                y = 'Density',
                color = choice,
                #y=choice,
                color_continuous_scale=px.colors.sequential.Bluered,
                hover_name='City',
                height=600,width=700,#900
                log_x=logX,log_y=logY))

    
#    col2.write("")
    col2.markdown("### Regressione lineare")
    col2.write(
    """
    Una semplice regressione lineare può darci qualche prima idea quantitativa della correlazine tra le feature. 
    Il fit è realizzato con il metodo dei minimi quadrati utilizzando Popolazione e Densità come variabili indipendenti (x) e la terza feature scelta dal menu a tendina come variabile dipendente (y).
    La qualità del fit può essere valutata dallo scarto quadratico medio (MSE) tra i valori veri di y e quelli previsti dal modello alla x corrispondente; 
    oppure tramite lo score R² del modello.\n
    L'MSE è sempre positivo e tanto più grande quanto peggiore è il modello. Può essere scomodo confrontare gli MSE ottenuti da set di dati diversi dato che la scala dipende dall'ordine di grandezza della variabile dipendente considerata.
    Per avere degli MSE confrontabili si possono normalizzare i valori y.\n
    L'R² ha un valore massimo di 1 corrispondente al modello che prevede perfettamente i dati, vale invece 0 se il modello è indipendente da x.
    Nel caso di regressione lineare semplice l'R² coincide con l'indice di correlazione r² tra due variabili. Infatti R²=1 implica completa correlazione, mentre R²=0 implica che i parametri sono del tutto indipendenti.
    """
    )
    if choice_name != "Zona":
        
        X = data.loc[:,["Population","Density"]]
        Y = choice
        Lin_model = LinearRegression()
        Lin_model.fit(X,Y)
        pred_Y=Lin_model.predict(X)

        col2.write("MSE = **"+str( round(mean_squared_error(pred_Y,Y,squared=False),4) )+"**")
        col2.markdown("R² = **"+str( round(Lin_model.score(X,Y),4) )+"**")

    """
    Confrontando gli score ottenuti dai fit sulle diverse features si nota una certa correlazione con le variabili epidemiologiche: score >0.8 per contagi e ospedalizzazioni, >0.7 per le terapie intensive. 
    In particolare è evidente (ed intuitivo) come il numero di contagi e ospedalizzazioni aumenti con l'aumentare del numero di abitanti del comune. 
    In controtendenza è il numero di decessi (score 0.44) a causa di alcuni comuni, relativamente piccoli che hanno avuto un'alta mortalità (e.g. il comune di Città di Castello ha avuto più morti di Perugia sebbene abbia meno di un terzo degli abitanti).
    \n
    Tutti i parametri epidemiologici, quando riportati in proporzione alla popolazione ottengono sia score che MSE molto vicini allo 0. 
    Questo significa che i parametri sono ben descritti da una distribuzione uniforme, perciò non c'è correlazione tra i due. 
    La diffusione dei contagi se rapportata alla popolazione risulta perciò piuttosto uniforme sul territorio.
    Ottenere un rapporto contagi su popolazione costante conferma la relazione lineare tra questi due parametri.
    \n
    I parametri di inquinamento a differenza di quelli epidemiologici non sembrano altrettanto legati alle caratteristiche demografiche. Troviamo infatti valori simili di concentrazione di pm10 tanto in comuni piccoli quanto in comuni di grandi dimensioni.
    I livelli di inquinamento atmosferico sembrano tendenzialmente legati più alla densità di popolazione piuttosto che alla popolazione totale del comune, ma senza ottenere in alcun caso score molto alti.\n
    \n
    Per apprezzare meglio queste relazione è utile fare dei confronti 1 a 1 tra le varie caratteristiche.
    """

    st.markdown("### Demografia vs Epidemiologia")
    col1, col2 = st.columns(2)
    X=data.Density
    y=data["CovidCases_jan_jun_2020"]#*100/data.Population

    col1.write(px.scatter(data,
                x=X,
                y=y,
                hover_name='City',
                trendline='ols',
                height=600,width=700))

    lin_model = LinearRegression()
    X=np.array(X).reshape(-1, 1)
    lin_model.fit(X,y)
    pred_Y=lin_model.predict(X)

    col2.write("MSE = **"+str( round(mean_squared_error(pred_Y,y,squared=False),4) )+"**")
    col2.markdown("R² = **"+str( round(lin_model.score(X,y),4) )+"**")
    col2.write("""
    Confrontando densità di popolazione e contagi emerge una leggerissima tendenza di correlazione (R²=0.2) che fatica però a spiegare la correlazione vista nel grafico precedente di 0.8.
    Il numero di contagi dovrebbe perciò dipendere quasi totalmente dal numero di abitanti del comune. 
    """)

    st.markdown("### Demografia vs Inquinamento")
    col1, col2 = st.columns(2)
    X=data.Density
    y=data["mean_pm10_ug/m3_mean_jan_jun_2020"]
    col1.write(px.scatter(data,
                x=X,
                y=y,
                hover_name='City',
                trendline='ols',
                height=600,width=700))

    lin_model = LinearRegression()
    X=np.array(X).reshape(-1, 1)
    lin_model.fit(X,y)
    pred_Y=lin_model.predict(X)

    col2.write("MSE = **"+str( round(mean_squared_error(pred_Y,y,squared=False),4) )+"**")
    col2.markdown("R² = **"+str( round(lin_model.score(X,y),4) )+"**")
    col2.write("""
    La dipendenza dei parametri di inquinamento dalla densità che sembrava emergere dal grafico iniziale è in realtà praticamente inesistente se non nei pochi comuni più densamente popolati.
    Per il resto dei casi c'è una variabilità che raggiunge anche il 200% tra comuni con la stessa densità. 
    """)

    st.markdown("### Inquinamento vs Epidemiologia")
    col1, col2 = st.columns(2)
    X=y
    y=data["CovidCases_jan_jun_2020"]/data.Population
    col1.write(px.scatter(data,
                x=X,
                y=y,
                hover_name='City',
                trendline='ols',
                height=600,width=700))
    
    lin_model = LinearRegression()
    X=np.array(X).reshape(-1, 1)
    lin_model.fit(X,y)
    pred_Y=lin_model.predict(X)

    col2.write("MSE = **"+str( round(mean_squared_error(pred_Y,y,squared=False),4) )+"**")
    col2.markdown("R² = **"+str( round(lin_model.score(X,y),4) )+"**")
    col2.write("""
    I parametri epidemiologici e di inquinamento appaiono completamente scorrelati (R²=0.014)
    """)

    st.markdown("### Mappe")
    """
    Rappresentare i parametri attraverso mappe cloropletiche può essere utile a visualizzare anche le relazioni geografiche tra i vari comuni 
    e come queste possano aver influenzato la diffusione della pandemia.
    """
    #col1,col2 = st.columns(2)

    st.markdown("#### Densità (ab./km^2)")
    st.write(px.choropleth_mapbox(data,geojson=comuni,locations=range(92),featureidkey='properties.id',
                        color='Density',hover_name='City',
                        hover_data={"Density":True,"Population":True,"Surface":True},
                        mapbox_style='carto-positron',center = {"lat": 43, "lon": 12.6},zoom=8,
                        labels={"Density":""},
                        #color_continuous_scale=px.colors.sequential.Bluered,
                        #color_discrete_sequence=px.colors.qualitative.D3,
                        height=800,width=1000))#1000

    st.markdown("#### Casi Covid (un.)")
    st.write(px.choropleth_mapbox(data,geojson=comuni,locations=range(92),featureidkey='properties.id',
                        color='Deceased_jan_jun_2020',hover_name='City',
                        labels={"Deceased_jan_jun_2020":""},
                        hover_data=['CovidCases_jan_jun_2020','AvgHospitalized_jan_jun_2020','AvgIntensiveCare_jan_jun_2020'],
                        mapbox_style='carto-positron',center = {"lat": 43, "lon": 12.6},zoom=8,
                        #color_continuous_scale=px.colors.sequential.Bluered,
                        #color_discrete_sequence=px.colors.qualitative.D3,
                        height=800,width=1000))#1000
    
    st.markdown("#### Concentrazione pm10 media giornaliera (ug/m^3)")
    st.write(px.choropleth_mapbox(data,geojson=comuni,locations=range(92),featureidkey='properties.id',
                        color='mean_pm10_ug/m3_mean_jan_jun_2020',hover_name='City',
                        labels={"mean_pm10_ug/m3_mean_jan_jun_2020":""},
                        hover_data=['CovidCases_jan_jun_2020','AvgHospitalized_jan_jun_2020','AvgIntensiveCare_jan_jun_2020'],
                        mapbox_style='carto-positron',center = {"lat": 43, "lon": 12.6},zoom=8,
                        #color_continuous_scale=px.colors.sequential.Bluered,
                        #color_discrete_sequence=px.colors.qualitative.D3,
                        height=800,width=1000))
    
#################################################
#cluster_exp=st.expander(label="",expanded=True)
#with cluster_exp:
if status==3:
    st.markdown("## Unsupervised Learning: Clustering")
    """
        Gli algoritmi di clustering permettono di categorizzare i dati secondo caratteristiche comuni che potrebbero sfuggire ad occhio nudo. 
        Questo permette ad esempio di poter individuare degli outlier all'interno del campione statistico o evidenziare relazioni tra i dati.
        \n
        L'algoritmo K-means utilizzato in questo studio si basa su un processo iterativo in cui N punti (centroidi) vengono posizionati casualmente all'interno dello spazio parametri p-dimensionale dei dati.
        A ciascun dato viene assegnata una categoria in base a quale centroide gli è più vicino. Vengono a questo punto generati dei nuovi centroidi posizionandoli nei punti medi di ciascun cluster.
        Il processo continua iterativamente fino alla stabilizzazione dei cluster o al raggiungimento di una condizione di arresto.
        \n 
    """
    #real_exp=st.expander(label="",expanded=True)
    st.markdown("### Clustering delle componenti reali")
    cl_params=['Depriv_idx',
        'CovidCases_jan_jun_2020/Pop',
        'MaxHospitalized_jan_jun_2020/Pop',
        'AvgHospitalized_jan_jun_2020/Pop',
        'Deceased_jan_jun_2020/Pop',
        'MaxIntensiveCare_jan_jun_2020/Pop',
        'AvgIntensiveCare_jan_jun_2020/Pop',
        'max_pm10_ug/m3_mean_jan_jun_2020',
        'max_pm10_ug/m3_std_jan_jun_2020',
        'max_pm10_ug/m3_median_jan_jun_2020',
        'mean_pm10_ug/m3_mean_jan_jun_2020',
        'mean_pm10_ug/m3_std_jan_jun_2020',
        'mean_pm10_ug/m3_median_jan_jun_2020',
        'min_pm10_ug/m3_mean_jan_jun_2020',
        'min_pm10_ug/m3_std_jan_jun_2020',
        'min_pm10_ug/m3_median_jan_jun_2020']

    """
    Un primo clustering è stato effettuato prendendo in considerazione come predittori i parametri demografici, epidemiologici e atmosferici riportati nella tabella qui di seguito.\n
    Si noti come i dati epidemiologici sono stati riportati in proporzione alla popolazione.

    """
    st.write(pd.Series(cl_params))

    """
    La mappa permette di visualizzare come l'algoritmo categorizza i comuni umbri al variare del numero di cluster richiesto.
    """

    cl_data=pd.DataFrame()
    for i,p in enumerate(cl_params):
        if i in range(1,7):
            cl_data[p] = data[p[:-4]] / data.Population
        else:
            cl_data[p]=data[p]

    cl_data=(cl_data-cl_data.mean(axis=0))/cl_data.std(axis=0)

    def clusterize(N=2,df=cl_data,draw=1,scatter=0):
        antonio=KMeans(n_clusters=N)
        antonio.fit(df)

        labels=antonio.labels_

        if draw:
            st.write(px.choropleth_mapbox(data,geojson=comuni,locations=range(92),featureidkey='properties.id',
                        color=labels.astype(str),hover_name='City',
                        mapbox_style='carto-positron',center = {"lat": 43, "lon": 12.5},zoom=8,
                        color_discrete_sequence=px.colors.qualitative.D3,
                        height=800,width=1000))
        if scatter:
            st.write(px.scatter_3d(x=df.PC1,y=df.PC2,z=df.PC3,
                        color=labels.astype(str),
                        hover_name=data.City,
                        height=600,width=800))

        return antonio.inertia_

    n_cl=st.columns(2)[0].number_input("Numero di cluster",2,10,4,1)
    clusterize(n_cl)

    """
    Un parametro importante nel clusterizzare un set di dati è l'inerzia. Questa corrisponde alla somma dei quadrati delle distanze di ciascun dato dal suo centroide più vicino.
    È utile andare a riportare in un grafico l'andamento dell'inerzia all'aumentare del numero di cluster richiesti. 
    La quantità ideale di cluster dovrebbe coincidere con quella in cui si ha un maggiore rallentamento nella decrescita dell'inerzia.
    Questa condizione infatti corrisponde ad un passaggio a cluster relativamente omogenei.
    """

    inertia=[]
    Range=range(1,18)
    for i in Range:
        inertia.append(clusterize(i,draw=0))

    st.write(px.line(x=Range,y=inertia,
                    width=700,height=500,
                    labels={"x":"numero cluster","y":"inerzia"}))


##########################
#pca_exp=st.expander(label="",expanded=True)
#with pca_exp:
if status==3:
    st.markdown("## Clustering delle componenti principali")
    """
    Dei 16 parametri tenuti in considerazione per il clustering non è detto che tutti contribuiscano allo stesso modo nella definizione dei clustering. 
    Se ad esempio alcuni di questi sono distribuiti in modo pressoché omogeneo non aggiungeranno molta informazione sulla possibile esistenza di cluster.
    \n
    Inoltre è possibile che alcune caratteristiche importanti siano contenute non tanto in un singolo parametro, ma nella combinazione di due o più.
    Può risultare utile per questi motivi andare a effettuare un operazione di riduzione della dimensionalità del dataset con tecniche come la PCA (Principal Component Analysis).
    \n
    Se ad esempio fissiamo a tre il numero di componenti principali che vogliamo ottenere, 
    l'algoritmo ci permette di ottenere per ogni entry del dataset tre valori ottenuti tramite delle combinazioni dei parametri reali che massimizzano l'informazione contenuta in ciascuno.
    \n
    I valori che si ottengono nel nostro caso sono:
    """
    X = cl_data
    n_comp=3

    peppe=PCA(n_components=n_comp)
    X_pca = peppe.fit_transform(X)
    X_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_comp)])
    st.write(X_pca)

    """
    Avendo solo tre componenti principali è ora molto più semplice avere anche una rappresentazione grafica del dataset.
    """

    clusterize(4, X_pca,draw=0,scatter=1)

    """
    Da questo grafico a dispersione è molto evidente la presenza di due chiari outlier della distribuzione corrispondenti ai comuni di Giove e Porano. 
    """

    n_cl_pca=st.columns(2)[0].number_input("Numero di cluster",2,10,4,1,key="pca")
    clusterize(int(n_cl_pca), X_pca)

    """
    Anche per questo caso si può studiare l'inerzia del clustering. 
    """

    inertia=[]
    Range=range(1,18)
    for i in Range:
        inertia.append(clusterize(i,X_pca,draw=0))

    st.write(px.line(x=Range,y=inertia,
                    width=700,height=500,
                    labels={"x":"numero cluster","y":"inerzia"}))


#################################################
#suplearn_exp=st.expander(label="",expanded=True)
#with suplearn_exp:
if status==4:
    st.markdown("## Supervised Learning: Modeling")
    """
    cose
    """
    #selezione predittori e y
    features = ['lat','lng','Surface',
                'mean_pm10_ug/m3_mean_jan_jun_2020',
                'mean_pm10_ug/m3_std_jan_jun_2020',
                'mean_pm10_ug/m3_median_jan_jun_2020']


    if st.checkbox("Escludi outlier"):
        data = data.drop(70)
        data = data.drop(35)

    X = data.loc[:,features]
    y = data.CovidCases_jan_jun_2020/data.Population
    y=pd.Series(y,name="y")

    """
    Sviluppiamo un modello utilizzando come predittori le caratteristiche seguenti
    """
    st.write(f"Predittori = {features}")

    #normalizzazione predittori
    X=(X-X.mean())/X.std()
    #y=(y-y.mean())/y.std()
    y=y/y.max()

    bodf=pd.concat([data["City"],X,y], axis=1)
    st.write(bodf)

    #splitting dataset
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=33)

#####################################################
    st.markdown("### ++ Boosted Decision Tree ++")
#####################################################

    #Gradient Boosted Decision Tree Regressor
    BT_model = GradientBoostingRegressor(learning_rate=0.1,subsample=0.8,random_state=2)
#    BT_model.fit(X_train,y_train)
    BT_model.fit(X_train,y_train)

    col1,col2=st.columns(2)
    #out of box improvement
    col2.write("Out-of-Box improvement:")
    col2.write(px.line(x=range(len(BT_model.oob_improvement_)),
                    y=BT_model.oob_improvement_))

    #score training
    col1.write("Score del training:")
    col1.write(px.line(x=range(len(BT_model.train_score_)),
                    y=BT_model.train_score_))
#    st.write(BT_model.train_score_)

    prediction = BT_model.predict(X_test)
    st.write(f"Mean Squared Error on predictions of the test set = {round(mean_squared_error(prediction,y_test),4)}")

    st.write(f"Score = {round(BT_model.score(X_test,y_test),4)}")

    plot1=pd.DataFrame()
    plot1["x"]=range(len(y_test))
    plot1["test set"]=np.array(y_test)
    plot1["predizione"]=np.array(prediction)
    col1,col2 = st.columns(2)
    col1.write(px.line(plot1,x="x",
                    y=["test set","predizione"]
                    )
            )
    col2.write(" ")
    col2.write(" ")
    col2.write(plot1)

#    mean_errs=[]
#    for i in range(4):
    # Valutiamo l'accuratezza del modello effettuando 10 fold del dataset
    k=len(y)
    #kf = KFold(n_splits=k,shuffle=True)
    kf = LeaveOneOut()
    scarti=[]
    errs=[]
    pred=[]
    test_set=[]
    for train,test in kf.split(X):
        #print(train, test)
        BT_model.fit(X.iloc[train],y.iloc[train])
        prediction = BT_model.predict(X.iloc[test])

        err = mean_squared_error(prediction,y.iloc[test])
        errs.append(err)
        #print(err)
        scarti.append((prediction[0]-y.iloc[test].tolist()[0]))
        pred.append(prediction[0])
        test_set.append(y.iloc[test].tolist()[0])
    #print()

#    st.write(scarti)
    # st.write(px.line(
    #                 x=range(len(scarti)),
    #                 y=scarti
    #                 )
    #         )

    plot2_y=["test set","predizione"]
    if st.checkbox("scarto predizione-verità"):
        plot2_y=["scarti"]
    plot2 = pd.DataFrame()
    plot2["x"]=range(len(test_set))
    plot2["test set"]=test_set
    plot2["predizione"]=pred
    plot2["scarti"]=scarti
    st.write(px.line(plot2,
                    x="x",
                    y=plot2_y,
                    hover_name=data["City"]
                    )
    )

    STD=np.array(errs).std()
    MEAN=np.array(errs).mean()
    #mean_errs.append(MEAN)
    st.write(f"Mean Squared Error on predictions through the {k} folds: {round(MEAN,6)}")
    st.write(f"Median Squared Error on predictions through the {k} folds: {round(np.median(np.array(errs)),6)}")
    st.write(f"Variance of Mean Squared Error on predictions through the {k} folds: {round(STD,6)}")

    st.write(px.line(x=range(k),y=errs))

#    st.write(px.line(x=range(4),y=mean_errs))
#    st.write(f"Errore medio = {round(np.array(mean_errs).mean(),6)}")

##################################################
    st.markdown("### ++ Support Vector Machine ++")
##################################################

    #Support Vector Machine Regressor
    SVR_model = SVR(C=2)
#    BT_model.fit(X_train,y_train)
    SVR_model.fit(X_train,y_train)

    col1,col2=st.columns(2)
    # #out of box improvement
    # col2.write("Out-of-Box improvement:")
    # col2.write(px.line(x=range(len(SVR_model.oob_improvement_)),
    #                 y=SVR_model.oob_improvement_))

#     #score training
#     col1.write("Score del training:")
#     col1.write(px.line(x=range(len(SVR_model.train_score_)),
#                     y=SVR_model.train_score_))
# #    st.write(SVR_model.train_score_)

    prediction = SVR_model.predict(X_test)
    st.write(f"Mean Squared Error on predictions of the test set = {round(mean_squared_error(prediction,y_test),4)}")

    st.write(f"Score = {round(SVR_model.score(X_test,y_test),4)}")

    plot1=pd.DataFrame()
    plot1["x"]=range(len(y_test))
    plot1["test set"]=np.array(y_test)
    plot1["predizione"]=np.array(prediction)
    col1,col2 = st.columns(2)
    col1.write(px.line(plot1,x="x",
                    y=["test set","predizione"]
                    )
            )
    col2.write(" ")
    col2.write(" ")
    col2.write(plot1)

#    mean_errs=[]
#    for i in range(4):
    # Valutiamo l'accuratezza del modello effettuando 10 fold del dataset
    k=len(y)
    #kf = KFold(n_splits=k,shuffle=True)
    kf = LeaveOneOut()
    scarti=[]
    errs=[]
    pred=[]
    test_set=[]
    for train,test in kf.split(X):
        #print(train, test)
        SVR_model.fit(X.iloc[train],y.iloc[train])
        prediction = SVR_model.predict(X.iloc[test])

        err = mean_squared_error(prediction,y.iloc[test])
        errs.append(err)
        #print(err)
        scarti.append((prediction[0]-y.iloc[test].tolist()[0]))
        pred.append(prediction[0])
        test_set.append(y.iloc[test].tolist()[0])
    #print()

#    st.write(scarti)
    # st.write(px.line(
    #                 x=range(len(scarti)),
    #                 y=scarti
    #                 )
    #         )

    plot2_y=["test set","predizione"]
    if st.checkbox("scarto predizione-verità",key="svr"):
        plot2_y=["scarti"]
    plot2 = pd.DataFrame()
    plot2["x"]=range(len(test_set))
    plot2["test set"]=test_set
    plot2["predizione"]=pred
    plot2["scarti"]=scarti
    st.write(px.line(plot2,
                    x="x",
                    y=plot2_y,
                    hover_name=data["City"]
                    )
    )

    STD=np.array(errs).std()
    MEAN=np.array(errs).mean()
    #mean_errs.append(MEAN)
    st.write(f"Mean Squared Error on predictions through the {k} folds: {round(MEAN,6)}")
    st.write(f"Median Squared Error on predictions through the {k} folds: {round(np.median(np.array(errs)),6)}")
    st.write(f"Variance of Mean Squared Error on predictions through the {k} folds: {round(STD,6)}")

    st.write(px.line(x=range(k),y=errs))

#    st.write(px.line(x=range(4),y=mean_errs))
#    st.write(f"Errore medio = {round(np.array(mean_errs).mean(),6)}")


st.session_state["status"]=status
