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
import gc

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
        df=pd.read_csv(file_path,nrows=1,sep=seperator)
        cols=df.columns.to_list()
        if target_col:
            cols.remove(target_col)
        dict={}
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
        if out:
            for key,columns in dict.items():
                print(f"{key} : {columns}")
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

def f_native(dataset_path,out_path,key,session,index_col=None,drop_mismatch=True,n_rows=None,save_as='csv',drop_not_found=False):
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
    native=0
    missing=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        fetch=[executor.submit(tmdb_fetch,id,session,headers) for id in mv_id]

        for i in concurrent.futures.as_completed(fetch):
            id,finished=i.result()
            results[id]=finished

    for i in mv_id:
        mv=df[df['tconst']==i]
        if results[i].get('movie_results'):
            native+=1
            lang=results[i]['movie_results'][0].get('original_language')
            if lang!=mv['language'].values[0]:
                if drop_mismatch:
                    indx_to_drop.extend(mv.index)
                else:
                    con=df['tconst']==i
                    df.loc[con,'language']=lang
                    rows+=len(mv)
        else:
            df.loc[df['tconst']==i,'language']=r'\N'
            missing+=1
        if drop_not_found:
            indx_to_drop.extend(mv.index)
    df=df.drop(index=indx_to_drop)
    df=df.drop_duplicates(keep='first',subset=['tconst'])
    if save_as=='csv':
        df.to_csv(out_path,index=False)
    elif save_as=='df':
        return df
    else:
        lg.error('Invalid Save As type')
        return lg.info('Use either "csv"/"df"')
    print(f"Filtered {rows} rows  \n Dropped {len(indx_to_drop)} rows \n Total_native_movies_found: {native} \nTotal_missing_movies_in_api: {missing}")


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
    
def clean_features(file_path,index_col=None,drop_na=False,drop_index=True):
    """
    Loads a dataset and performs final sanitization before model prediction.

    This function handles specific database artifacts (like '\\N'), performs a 
    surgical drop on columns with minimal missing data, and safely imputes the rest.
    
    The Logic:
    1. Reads the CSV and converts literal '\\N' strings into true nulls.
    2. Identifies columns that are mostly complete (between 1 and 50 nulls).
    3. Drops only the rows missing data in those specific, high-quality columns.
    4. Drops the ID/Index column (like 'tconst') if specified, to prevent data leakage.
    5. Fills all remaining nulls across the DataFrame with empty strings ('') 
       so the text tokenizer can process them without crashing.

    Args:
        file_path (str): The local path to the merged CSV dataset.
        index_col (str, optional): The name of an ID column to drop from the 
            features (e.g., 'tconst'). Defaults to False.

    Returns:
        pd.DataFrame: A fully cleaned, imputed, and model-ready DataFrame.
    """
    df=pd.read_csv(file_path,na_values=r'\N')
    null_cols=df.isnull().sum()
    null_drop=null_cols[(null_cols<=50)&(null_cols>0)].index
    if index_col and drop_index:
        df=df.drop(index_col,axis=1)
    if drop_na:
        df=df.dropna(subset=null_drop)    
    df=df.fillna('')
    return df


def comma_split(text):
    if type(text)!=str:
        return []
    else:
        return text.split(',')

def train_model(training_file_path,model_path,out_metrics=False,save=False,save_path=None):
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
    training_path : str
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
    df=clean_features(training_file_path,index_col='tconst')
    df['language']=(df['language']=='ml').astype('Int64')
    y=df.pop('language')
    X=df
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=joblib.load(model_path)
    model.fit(x_train,y_train)
    predict=model.predict(x_test)
    if out_metrics:
        print(f"Confusion_matrix: {confusion_matrix(y_test,predict)}\nPrecision: {precision_score(y_test,predict)}\nAccuracy: {accuracy_score(y_test,predict)}")
        print(f" \nClassification report: {classification_report(y_test,predict)}")
    if save and save_path:
        joblib.dump(model,save_path)
        print(f'Saved as: {save_path}')
    return model

def mergeon_cols(origin_cols_dict,file_dict,n_rows=1000000,sep=None,skip_rows=0,save_as='df',out_path=None,group=False,group_file_name='title.principals(crew)',use_crew_cols=['tconst','crews','category'],category_group=['actor','actress','cinematographer','composer'],group_sep='\t'):
    """
    Dynamically loads, filters, and merges multiple large CSV/TSV datasets.
    Includes a built-in memory safety valve and chunking engine for massive files.

    Args:
        origin_cols_dict (dict): Maps file keys to the specific columns to load.
        n_rows (int): The absolute maximum number of rows to read per file.
        file_dict (dict): Dictionary of file keys and their file paths.
        sep (str): Delimiter for standard files (default is '\\t').
        skip_rows (int): Number of rows to fast-forward (Warning: Must handle headers!).
        save_as (str): Output format. 'df' returns Pandas DataFrame, 'csv' saves to disk.
        out_path (str): File path for saving the CSV if save_as='csv'.
        group (bool): Flag to trigger specialized processing for a specific file.
        group_file_name (str): The key name of the massive file that requires chunking (e.g., 'title.principals').
        group_sep (str): Delimiter specifically for the chunked file.
        category_group (list): List of strings to filter the chunked file (e.g., ['actor', 'director']).
        use_crew_cols (list): Explicit columns to load from the chunked file (e.g., ['tconst', 'nconst', 'category'] or ['tconst', 'crews', 'category']).

    Returns:
        pd.DataFrame or None: Returns the merged DataFrame if save_as='df', otherwise saves to disk and returns None.
        
    Notes:
        - Includes a safety check: (n_rows - skip_rows <= 500000) to prevent RAM meltdowns.
        - The 'group' file is processed using a chunksize of 50,000 to safely extract 
          and collapse string IDs (like nconst) without overflowing memory.
    """
    paths=[]
    cols=[]
    dfs=[]
    crews=None
    if group_file_name.endswith('.tsv'):
        group_file_name=group_file_name.removesuffix('.tsv')
    elif group_file_name.endswith('.csv'):
        group_file_name=group_file_name.removesuffix('.csv')         
    if n_rows-skip_rows<=1500000:
        for key,values in file_dict.items():
            if key in origin_cols_dict:
                if group and key==group_file_name :
                    crews=values
                else:
                    paths.append(values)
                    cols.append(origin_cols_dict[key])
        for i in range(0,len(paths)):
            sep=sep if sep else('\t' if paths[i].endswith('.tsv') else ',')
            true_headers = pd.read_csv(paths[i], sep=sep, nrows=0).columns.tolist()
            df=pd.read_csv(paths[i],sep=sep,usecols=cols[i],nrows=n_rows,skiprows=skip_rows + 1,na_values=r'\N',names=true_headers)
            df.drop_duplicates(subset=['tconst'],keep='first',inplace=True,ignore_index=True)
            del i
            dfs.append(df)
            del df
            gc.collect()
        if crews:
            df=dfs[0]
            tconst=df['tconst'].to_list()
            if use_crew_cols:
                if group_sep:
                    chunk=pd.read_csv(crews,usecols=use_crew_cols,sep=group_sep,chunksize=750000,na_values=r'\N')
                else:
                    if crews.endswith('.tsv'):
                        chunk=pd.read_csv(crews,usecols=use_crew_cols,sep='\t',chunksize=750000,na_values=r'\N')
                    elif crews.endswith('.csv'):
                        chunk=pd.read_csv(crews,usecols=use_crew_cols,sep=',',chunksize=750000,na_values=r'\N')
                    else:
                        lg.error('Invalid extension')
                        return lg.info('Please mention group_sep')
                mvs=[]
                count=0
                for i in chunk:
                    count+=1
                    filtered=i[(i['tconst'].isin(tconst))&(i['category'].isin(category_group))]
                    if not filtered.empty:
                        grouped_lst=filtered.groupby('tconst')['crews'].apply(lambda x: ','.join(x.astype(str))).reset_index()
                        mvs.append(grouped_lst)
                    del i
                    del filtered
                    gc.collect()
                if mvs:
                    crew_df=pd.concat(mvs)
                    crew_df=crew_df.groupby('tconst')['crews'].apply(lambda x: ','.join(x.astype(str))).reset_index()
                    dfs.append(crew_df)
        df=dfs[0]
        df['tconst']=df['tconst'].astype(str)
        for i in range(1,len(dfs)):
            dfs[i]['tconst']=(dfs[i]['tconst']).astype(str)
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
                del df
                gc.collect()
    else:
        lg.warning("Please keep the range of rows in between 300000")
        lg.info('Toggle "n_rows/skip_rows" to keep rows range in between 500000')    

def chunk_mergeon_cols(origin_cols_dict,file_dict,total_rows=None,index_col_file=None,skip_rows=0,chunk_size=50000,out_path=None,group=False,group_file_name='title.principals(crew)',use_crew_cols=['tconst','crews','category'],category_group=['actor','actress','cinematographer','composer'],group_sep='\t',append_only=False):
    if not total_rows:
        if index_col_file:  
            if index_col_file.endswith('.tsv'):   
                total_rows=sum([len(i) for i in pd.read_csv(index_col_file,chunksize=50000,sep='\t')])
            elif index_col_file.endswith('.csv'):
                total_rows=sum([len(i) for i in pd.read_csv(index_col_file,chunksize=50000,sep=',')])
            else:
                return lg.error('Invalid sep type in index_col_file')
        else:
            return lg.error('Please mention either "total_rows"/"index_col_file"!!')
    if skip_rows and append_only:
        start_from=skip_rows
    else:
        start_from=0
    num=0
    for i in range(start_from,total_rows+chunk_size,chunk_size):
        if not append_only:
            if skip_rows:
                mergeon_cols(origin_cols_dict,file_dict,n_rows=chunk_size,skip_rows=skip_rows,out_path=out_path,group=group,group_file_name=group_file_name,use_crew_cols=use_crew_cols,category_group=category_group,group_sep=group_sep,save_as='csv')
            else:
                mergeon_cols(origin_cols_dict,file_dict,n_rows=chunk_size,out_path=out_path,group=group,group_file_name=group_file_name,use_crew_cols=use_crew_cols,category_group=category_group,group_sep=group_sep,save_as='csv')
            append_only=True
        else:
            app=mergeon_cols(origin_cols_dict,file_dict,n_rows=chunk_size,skip_rows=i,group=group,group_file_name=group_file_name,use_crew_cols=use_crew_cols,category_group=category_group,group_sep=group_sep,save_as='df')
            if app is not None:
                app.to_csv(out_path,mode='a',index=False,header=False)
                num+=1
                print(f"Chunk Appended: chunk{num}")
                del app
                gc.collect()
            else:
                print('Data for append not returning as Dataframe')
                break
    print('Merged Datas!!!!')
    print('Total Rows: ',total_rows)