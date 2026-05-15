import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging as lg
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,classification_report,precision_score,confusion_matrix
from sklearn.model_selection import train_test_split
import concurrent.futures
from dotenv import load_dotenv

def find_cols(files_dictionary: dict[str,str], seperator=',',out=False) -> dict[str,list[str]]:
    """Find column names for a set of CSV/TSV files.

    Args:
        files_dictionary: Mapping of friendly names to file paths.
        seperator: Field separator used when reading files (e.g. '\t' for TSV, ',' for CSV).
        out: If True, print the discovered columns for each file.

    Returns:
        Dictionary where each key is the friendly name and each value is the list of column names.

    Example:
        files = {'Basics': 'data/title.basics.tsv'}
        cols = find_cols(files, seperator='\t', out=True)
    """
    try:
        cols={}
        for name,loc in files_dictionary.items():
            df=pd.read_csv(loc,nrows=5,sep=seperator)
            cols[name]=df.columns.to_list()
        
        if out:
            for i,j in cols.items():
                print(f"{i}: {j}")
        else:
            return cols;
    except pd.errors.ParserError as e:
        lg.error("Invalid seperator type")
        lg.warning("Please Mention the seperator type")
        
def col_origin(cols_dictionary: dict, file_path: str, seperator=',',out=False,target_col=None,index_col=None):
    """Match columns from a source file to a set of candidate files.

    Args:
        cols_dictionary: Mapping of file names to column name lists.
        file_path: Path of the file whose columns should be matched.
        seperator: Separator used to read the source file.

    Returns:
        None. Prints which candidate file contains each matching column.

    Example:
        cols = {'AKAs': ['tconst', 'title', 'language']}
        col_origin(cols, 'Original malayalam movies.csv')
    """
    try:
        df=pd.read_csv(file_path,nrows=5,sep=seperator)
        cols=df.columns.to_list()
        if target_col:
            cols.remove(target_col)
        dict={}
        print(cols)
        for key,columns in cols_dictionary.items():
            lst=[]
            for i in columns:
                if i in cols:
                    lst.append(i) 
            useful=[c for c in lst if c!=index_col]
            if len(useful)>0:
                dict[key]=lst
        if not out:
            return dict
    except pd.errors.ParserError as e:
        lg.error("Invalid seperator type")
        lg.warning("P please Mention the seperator type")

def create_session():
    """Create a requests session with retry handling for transient errors.

    Returns:
        A configured requests.Session instance.

    Example:
        session = create_session()
        response = session.get('https://api.themoviedb.org/3/')
    """
    session=requests.session()
    retry=Retry(total= 5,backoff_factor=2,status_forcelist=[429, 500, 502, 503, 504],connect=5,read=2)  #count is for retrying for the connection error & read for data transfer error
    adapter=HTTPAdapter(max_retries=retry)
    session.mount('https://',adapter)
    return session

def chunk_unpack(Chunk,lang1=None,lang2=None,merge_on=None,merge_df=None,use_for="unpack",how='inner'):
    """Process chunked DataFrames either by filtering languages or by merging.

    Args:
        Chunk: Iterable of DataFrame chunks returned by pandas.read_csv(..., chunksize=...).
        lang1: Primary language filter value for the 'language' column.
        lang2: Secondary language filter value for the 'language' column.
        merge_on: Column name used to join when use_for='merge'.
        merge_df: DataFrame to merge with filtered chunk rows when use_for='merge'.
        use_for: Either 'unpack' to filter language rows or 'merge' to join with merge_df.
        how: Merge strategy used when use_for='merge' (default 'inner').

    Returns:
        A DataFrame containing either the filtered rows or the merged result.

    Examples:
        chunks = pd.read_csv('title.akas.tsv', sep='\t', chunksize=50000)
        malayalam = chunk_unpack(chunks, lang1='ml')

        merged = chunk_unpack(chunks, use_for='merge', merge_on='tconst', merge_df=existing_df)
    """
    mv=[]
    if use_for.lower()=="unpack":
        if lang1 & lang2:
            for i in Chunk:
                lst=i[(i['language']==lang1)|(i['language']==lang2)]
                mv.append(lst)
        elif lang1:
            for i in Chunk:
                lst=i[(i['language']==lang1)]
                mv.append(lst)
        return pd.concat(mv)
    elif use_for.lower()=='merge':
        merge_lst=merge_df[merge_on].unique()
        for i in Chunk:
            lst=i[i[merge_on].isin(merge_lst)]
            mv.append(lst)
        mv_df=pd.concat(mv)
        return pd.merge(merge_df,mv_df,how=how,on=[merge_on])
      
def tmdb_fetch(id,session,headers):
    url=f'https://api.themoviedb.org/3/find/{id}'

    params={
        "external_source": 'imdb_id'
        }
    response=session.get(url,params=params,headers=headers)
    return id,response.json()

def f_native(dataset_path,out_path,key,session,index_col=None,drop_mismatch=True,n_rows=None):
    """Filter a dataset by whether the TMDB original language matches the dataset language.

    Args:
        session: A requests.Session configured with retry handling.
        key: TMDB Bearer token string.
        dataset_path: Path to a CSV file containing a 'tconst' and 'language' column.
        out_path: Path location to save the filtered file.
        index_col: Optional pandas index column for reading the dataset.
        drop: drop the the non native movies from the dataset (Default value: True).

    Returns:
        None. Saves a filtered file named 'Class 0 native.csv'.

    Example:
        session = create_session()
        f_native(session, api_key, 'Class 0.csv')
    """
    headers={
        'accept':'application/json',
        'authorization':f"Bearer {key}",
        'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
    }
    if n_rows:
        df=pd.read_csv(dataset_path,index_col=index_col,nrows=n_rows)
    else:
        df=pd.read_csv(dataset_path,index_col=index_col)
    mv_id=df['tconst'].unique()
    rows=0
    indx_to_drop=[]
    results={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        fetch=[executor.submit(tmdb_fetch,id,session,headers) for id in mv_id]

        for i in concurrent.futures.as_completed(fetch):
            id,finished=i.result()
            results[id]=finished

    for i in mv_id:
        mv=df[df['tconst']==i]
        if results[i].get('movie_results'):
            lang=results[i]['movie_results'][0].get('original_language')
            if lang!=mv['language'].values[0]:
                if drop_mismatch:
                    indx_to_drop.extend(mv.index)
                else:
                    con=df['tconst']==i
                    df.loc[con,'language']=lang
                    rows+=len(mv)
        else:
            indx_to_drop.extend(mv.index)
    df=df.drop(index=indx_to_drop)
    df=df.drop_duplicates(keep='first',subset=['tconst'])
    df.to_csv(out_path,index=False)
    print(f"Filtered {rows} rows  \n Dropped {len(indx_to_drop)} rows \n total_native_movies_found: {len(df)}")


def get_response(movie_id: str,session,key,is_json=False):
    """Retrieve TMDB find results for a given IMDb movie ID.

    Args:
        movie_id: IMDb title ID such as 'tt1234567'.
        session: requests.Session object for HTTP requests.
        key: TMDB Bearer token string.

    Returns:
        requests.Response object from the TMDB API.

    Example:
        response = get_response('tt0075860', session, api_key)
        data = response.json()
    """
    url=f'https://api.themoviedb.org/3/find/{movie_id}'
    header={
        'accept':'application/json',
        'Authorization':f'bearer {key}',
        'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
    }
    parameters={
        'external_source':'imdb_id'
    }
    response=session.get(url,params=parameters,headers=header,timeout=10)
    if is_json:
        return response.json()
    else:
        return response

def comma_split(text):
    if type(text)!= str:
        return []
    return text.split(',')

def load_model(file_path):
    """Load the saved model from the given path
    Parameters:
        file_path: path of the stored model"""
    return joblib.load(file_path)

def train_model(file_path,model_path,out_metrics=False,save=False,save_path=None):
    """
    Train a machine learning classification model on movie metadata.

    This function:
    - Loads and preprocesses the dataset
    - Converts the target language column into a binary label
    - Splits the data into training and testing sets
    - Loads a preconfigured model pipeline
    - Trains the model
    - Optionally prints evaluation metrics
    - Optionally saves the trained model

    Parameters
    ----------
    file_path : str
        Path to the CSV dataset file.

    model_path : str
        Path to the serialized model/pipeline file that will be loaded
        using `load_model()`.

    out_metrics : bool, default=False
        If True, prints:
        - Confusion matrix
        - Precision score
        - Accuracy score
        - Classification report

    save : bool, default=False
        If True, saves the trained model to:
        `'models/mal_model.joblib'`

    Returns
    -------
    model : sklearn Pipeline or estimator
        The trained machine learning model.

    Notes
    -----
    Dataset preprocessing steps:
    - Keeps only the following columns:
        ['originalTitle', 'genres', 'directors', 'writers', 'language']
    - Removes rows with missing `directors`
    - Fills remaining missing values with empty strings
    - Converts:
            language == 'ml' -> 1
            all other languages -> 0

    Train/Test Split
    ----------------
    - Test size: 20%
    - Random state: 42

    Example
    -------
    >>> model = train_model(
    ...     file_path="data/movies.csv",
    ...     model_path="models/pipeline.joblib",
    ...     out_metrics=True,
    ...     save=True
    ... )

    Output Example
    --------------
    Confusion Matrix:
    [[120  10]
     [  5  65]]

    Precision: 0.87
    Accuracy: 0.92

    Classification Report:
                  precision    recall    f1-score   support
    """
    df=pd.read_csv(file_path,usecols=['originalTitle', 'genres', 'directors', 'writers',
       'language'],na_values=r'\N')
    df=df.dropna(subset=['directors'])
    df=df.fillna('')
    df['language']=(df['language']=='ml').astype('Int64')
    y=df.pop('language')
    X=df
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=load_model(model_path)
    model.fit(x_train,y_train
              )
    predict=model.predict(x_test)
    if out_metrics:
        print(f"Confusion_matrix: {confusion_matrix(y_test,predict)}\nPrecision: {precision_score(y_test,predict)}\nAccuracy: {accuracy_score(y_test,predict)}")
        print(f" \nClassification report: {classification_report(y_test,predict)}")
    if save and save_path:
        joblib.dump(model,save_path)
        print(f'Saved as: {save_path}')
    return model

def mergeon_cols(orgcols_dict,file_dict,sep=',',n_rows=None,save_as='df',out_path=None):
    paths=[]
    cols=[]
    dfs=[]
    for key,values in file_dict.items():
        if key in orgcols_dict:
            paths.append(values)
            cols.append(orgcols_dict[key])
    if n_rows:
        for i in range(0,len(paths)):
            df=pd.read_csv(paths[i],nrows=n_rows,usecols=cols[i],sep=sep)
            dfs.append(df)
    else:
        for i in range(0,len(paths)):
            df=pd.read_csv(paths[i],usecols=cols[i],sep=sep)
            dfs.append(df)
    df=dfs[0]
    for i in range(1,len(dfs)):
        df=pd.merge(df,dfs[i],how='inner',on='tconst',)
    if save_as=='df':
        return df
    elif save_as=='csv':
        if out_path:
            df.to_csv(out_path,index=False)
            print(f"Merged File saved As: {out_path}")
        else:
            df.to_csv('ColumnsMerged.csv',index=False)
            print('Merged File Saved As: "ColumnsMerged.Csv')
    