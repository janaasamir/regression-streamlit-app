import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder , OneHotEncoder ,OrdinalEncoder , StandardScaler , MinMaxScaler , RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Lasso , Ridge ,SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
import joblib 

st.title("Regression model")

up = st.file_uploader("Please, select dataset" , type=["csv"])
if up is not None:
    df = pd.read_csv(up)
    st.write("Display the data: ")
    st.table(df.head())
    if st.checkbox("describe dataset"):
        st.write("display distribution")
        st.table(df.describe())

    if st.checkbox("check missing data"):
        st.write(df.isna().sum())
        if st.button("DELETE MISSING VALUES"):
            df.dropna(inplace=True)
            st.success("DONE")
        
    if st.checkbox("check duplication"):
        st.write(df.duplicated().sum())
        if st.button("DELETE duplication"):
            df.drop_duplicates(inplace=True)
            st.success("DONE")
    
    if st.checkbox("check outliers"):
        fig , ax = plt.subplots()
        sns.boxplot(df, ax=ax)
        st.pyplot(fig)
    
    if st.checkbox("transformation for categorical features"):
        cat = df.select_dtypes("object").columns
        st.write(cat)

        ops = st.selectbox("select Encoding way" , options=["label encoding" , "one hot encoding","ordinal encoding"])
        if ops =="label encoding":
            pass
        elif ops == "one hot encoding":
            pass
        else:
            pass

    if st.checkbox("split data to features and target"):
        target = st.selectbox("select target",options=df.columns)
        st.write(target)
        features = st.multiselect("select features",options=df.columns)
        st.write(features)  

        xtrain , xtest , ytrain , ytest = train_test_split(df[features],df[target],test_size=0.2,random_state=42,shuffle=True)
        st.success("SPLITTING DONE")     
    try:
        if st.checkbox("transformation numaric data"):
            scaler = st.selectbox("select scaler",options=["minmax","standardization","robust"])
            if scaler=="minmax":
                scaler=MinMaxScaler()
            elif scaler=="standardization":
                scaler=StandardScaler()
            elif scaler=="robust":
                scaler=RobustScaler()
            
            features = st.multiselect("select features",options=xtrain.columns)
            xtrain[features] = scaler.fit_transform(xtrain[features])
            xtest[features]=scaler.transform(xtest[features])
            st.success("SCALING DONE")

        st.header("MODELING PHASE")
        option = st.selectbox("select model",options=["KNN","LR"])
        if option =="KNN":
            k = st.slider("n_neighbor",2,20,10)
            wieghts = st.selectbox("weights",options=["uniform","distance"])
            matrix = st.selectbox("matrix",options=["euclidean","manhaten"])
            if matrix=="manhaten": p=1
            else: p=2
            model = KNeighborsRegressor(n_neighbors=k,weights=wieghts,p=p)
        elif option=="LR":
            mops = st.selectbox("select linear regression version",options=["lasso","ridge" , "SGD","linear regression"])
            if mops == "lasso":
                
                alpha = st.number_input("alpha",0.0001,max_value=100.0)
                model = Lasso(alpha=alpha)
            elif mops == "ridge":
                
                alpha = st.number_input("alpha",0.0001,max_value=100.0)
                model = Ridge(alpha=alpha)
            elif mops == "SGD":
                
                alpha = st.number_input("alpha",0.0001,max_value=100.0)
                model = SGDRegressor(alpha=alpha)
            elif mops=="linear regression":
                model = LinearRegression()

        flag = False
        if st.checkbox("train"):
            flag= True
        if flag:
            model.fit(xtrain,ytrain)
            st.write(model)
            st.success("Trained")


        if st.checkbox("predict"):
            
            ypred = model.predict(xtest)
            st.write(f"mean absolute error >>>{mean_absolute_error(ypred,ytest)}")
            st.write(f"mean squared error >>>{mean_squared_error(ypred,ytest)}")


        
        if st.checkbox("Save model"):
            joblib.dump(model,"model.pkl")
            st.download_button("Download model",data=open("model.pkl","rb"),file_name="model.pkl")



    except:
        st.warning("check that the train test split is working")

    
