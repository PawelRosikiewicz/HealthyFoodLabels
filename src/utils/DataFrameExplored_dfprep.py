# ********************************************************************************** #
#                                                                                    #
#   Project: Data Frame Explorer                                                     #                         
#   Author: Pawel Rosikiewicz                                                        #
#   Contact: prosikiewicz(a)gmail.com                                                #
#                                                                                    #
#   License: MIT License                                                             #
#   Copyright (C) 2021.01.30 Pawel Rosikiewicz                                       #
#                                                                                    #
# Permission is hereby granted, free of charge, to any person obtaining a copy       #
# of this software and associated documentation files (the "Software"), to deal      #
# in the Software without restriction, including without limitation the rights       #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell          #
# copies of the Software, and to permit persons to whom the Software is              #
# furnished to do so, subject to the following conditions:                           #
#                                                                                    # 
# The above copyright notice and this permission notice shall be included in all     #
# copies or substantial portions of the Software.                                    #
#                                                                                    #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR         #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,           #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE        #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER             #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,      #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE      #
# SOFTWARE.                                                                          #
#                                                                                    #
# ********************************************************************************** #


# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import random
import glob
import re
import os
import seaborn as sns

from IPython.display import display
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype






# Function, ............................................................................
def find_and_display_patter_in_series(*, series, pattern):
    "I used that function when i don't remeber full name of a given column"
    res = series.loc[series.str.contains(pattern)]
    return res



# Function, ...........................................................................................
def load_csv(*, path, filename, sep="\t", verbose=True):
    """ 
        Loads csv into pandas df, based on pandas.read_scv(), 
        Returns error, if file or directoy not found
        
        Parameters/Input              
        _________________   _______________________________________________________________________________  

        * path              full path to directory
        * csv_name.         full csv file name
        * separator         "\t", by default
        * display_head      bool, True, by default, display df.head(), 
                            irrespectively when the futions was called. 
        Returns              
        _________________   _______________________________________________________________________________  

        * DataFrame         by Pandas

    """
    
    os.chdir(path)
    if len(glob.glob(filename))==1:  
        df = pd.read_csv(filename,  sep=sep, low_memory=False)
        
        # display example,
        if verbose==True:
            display(df.head(3))
            print(df.shape)
        else:
            pass
            
        # return,
        return df
    
    else:
        if verbose==True:
            print(f"""ERROR :csv file {filename}, was not found in: \n {path}""")
        else:
            pass


          
          
          
          
# Function, ............................................................................
def find_patter_in_series(*, s, pat, tolist=True):
    '''
        I used that function when i don't remeber full name of a given column
    '''
    res = s.loc[s.str.contains(pat)]
    
    if tolist==True:
        return res.values.tolist()
    else:
        return res 
    
    
    
    

  
# Function, ........................................................................................... 
def format_to_datetime(*, data, pattern_list, timezone='UTC', unixtime=False, dt_format='%Y-%m-%d %H:%M:%S', verbose=False):
    '''
        formats columns in df into datetime dtype, and set all times to UTC
        work with unix time units, ie. second number since 1970
        columns in df, are find using full comlumn name or keywords in column name
    '''
    assert type(data)==pd.DataFrame, "please provide data in pandas dataframe format"
    
    if isinstance(pattern_list, str):
        pattern_list = [pattern_list]
    else: 
        pass
    
    for pat in pattern_list:            
        # find column names using provided patterns or their full names, 
        columns_with_potential_datetime_obj = list(find_and_display_patter_in_series(series=pd.Series(data.columns), pattern=pat))
        
        # replace 
        for i in columns_with_potential_datetime_obj:
            # keep example of old cell 
            before_formatting = str(data.loc[0, i])
            
            # convert to one format
            if unixtime==True:
                s = pd.to_datetime(data.loc[:, i], errors="coerce", unit='s').copy()#,format cannot be used with unit="s", but it will be the same
                data.loc[:, i] = s
                if timezone!=None:
                    data.loc[:, i] = data.loc[:, i].dt.tz_localize(timezone)
                else:
                    pass
                
            else: 
                s = pd.to_datetime(data.loc[:, i], errors="coerce",format=dt_format).copy()
                data.loc[:, i] = s
                if timezone!=None:
                    data.loc[:, i] = data.loc[:, i].dt.tz_convert(timezone)
                else:
                    pass
            
            # info
            if verbose==True:
                print(f"date time formatted in: {i}") 
                print(f" - {data.loc[:, i].isnull().sum()} NaN were instroduced by coerce")
                print(f" - Example: {before_formatting} -->> {str(data.loc[0, i])}", end="\n")
            else:
                pass

    return data    
    
    
    
    
    
    
    
# Function, ...........................................................................................
def replace_text(*,df ,pat="", colnames="all", fillna=np.nan, verbose=True):
    """ 
        searches string with a given pattern and replace it with a new patter (fillna), eg: nan,
                            
        Parameters/Input              
        _________________   _______________________________________________________________________________  

        * df                Pandas Dataframe
        * searched_pattern  "", str literal, used by pd.Series.str.contains() 
        * colnames          default, "all", or list with selected colnames in df
        * fillna            default numpy.nan, or str literal 
                            - what do you want to place instead of searched pattern in df
    
        Returns              
        _________________   _______________________________________________________________________________  

        * DataFrame         DataFramne.copy() with new values,
        * display messages. number of replaced straings in each column, and examples of replcaced values
    """
    
    # for older version, 
    searched_pattern = pat
    col_names = colnames
    
    # check col_names with values to replace,  
    if col_names=="all": 
        sel_col_names = list(df.columns)
    else:                
        sel_col_names = col_names  

    # display message header, 
    if verbose==True:
        print(f"""\nReplacing Text in {len(sel_col_names)} columns: {sel_col_names}\n""") 
      
    if verbose==False:
        pass

    # exchnage searched pattern in each column separately,   
    for i, col_name in enumerate(sel_col_names):
         
        # .. test if you really have string values in that column, otherwise it masy be float for all NaN in a column, and no action will be taken 
        if is_string_dtype(df[col_name]):
        
            try:
                # .... find postions with a given pattern and select three examples to display for the user, 
                positions_to_replace = df[col_name].str.contains(searched_pattern, na=False).values# arr
                examples_to_display = [str(x) for x in list(df.loc[list(positions_to_replace), col_name].str[0:20].values.tolist()[0:3])]

                # .... replace postions, and find examples of unchnaged postions,
                df.loc[list(positions_to_replace), col_name] = [fillna]*positions_to_replace.sum()  
                examples_of_positions_that_were_not_replaced = [str(x) for x in list(df.loc[list(positions_to_replace==False), col_name].str[0:20].values.tolist()[0:3])]

                # .... diplay info,
                if verbose==True:
                    perc_of_replaced_pos_in_col = "".join([str(positions_to_replace.sum()/df.shape[0]*100),"%"])
                    print(f"{i} - {col_name} - - {positions_to_replace.sum()} positions out of {df.shape[0]}, were replaced with {fillna}, ie. {perc_of_replaced_pos_in_col}")
                    print(f" - three examples of replaced postions:  {'; '.join(examples_to_display)}", end="\n")
                    print(f" - three examples of unchanged postions: {'; '.join(examples_of_positions_that_were_not_replaced)}", end="\n\n")
                          # the second print returns three first examples of exchanged values, just to see what i did,
                else:
                    pass
                
            except:
                if verbose==True:
                    print(f"{i} - {col_name} - - probably only missing data datected, Values were not replaced! \n") 
                else:
                    pass
        
        else:
            if verbose==True:
                print(f"{i} - {col_name} - - is not of string type, Values were not replaced! \n")  
            else:
                pass
            
    return df.copy()


  
  
  



# Function, ...........................................................................................
def replace_numeric_values(*, df, colnames="all", lower_limit="none", upper_limit="none", equal=False, replace_with=np.nan, verbose=True):
    """ 

        Replace numerical values that are outside of range of a values 
        prediced with a theoretical limits of a given variable, 
        eg less then 0 in weight of a product, 
        Provide examples and numbers of replaced instances
                            
        Parameters/Input              
        _________________   _______________________________________________________________________________  

        * df                : Pandas DataFrame
        * cols_in_df        : list, exact colnames of selected or all columns in df
        * lower_limit       : int,float,"none", if "none" no action is taken
        * upper_limit       : int,float,"none", if "none" no action is taken
        * replace_with      : str, np.nan, int, float
        * equal             : bool, if True, >= and <= values then limits will be replaced,
                              if False (default), > and < values then limits will be replaced,
    
        Returns              
        _________________   _______________________________________________________________________________  

        * DataFrame         DataFramne.copy() with new values,
        * display messages. number of replaced straings in each column, and examples of replcaced values
    """      

    
    cols_names = colnames
    
    # .. check provided col_names,
    if cols_names=="all": 
        cols = list(df.columns)
    else:                
        cols = cols_names        

    # .. info, header, 
    if verbose==True:
        print(f"""\n{"".join(["-"]*80)} \n Replacing Numerical Values in {len(cols)} columns""") 
        print(f"     lower filter={lower_limit},    upper filter ={upper_limit}")
        if equal==True:
            print(f"     Caution, equal=True, ie. values >= and <= then requested limits will be replaced")
        print(f'{"".join(["-"]*80)}\n') 
        
    if verbose==False:
        pass
        
    
    # .. intelligent info,
    total_count=[]

    # .. count, to limit the number of displayed messages,
    count = 0

    # .. replace values and collect examples, 
    for i, j in enumerate(cols):

        # ..... assume no values were replaced, so the messages work later,  
        info_lower_filter = 0
        info_upper_filter = 0        
        
        # ..... test if the column is of the numeric type:
        # from pandas.api.types import is_numeric_dtype
        if is_numeric_dtype(df[j]):
            
            
            # * replace values < or <= lower limit,
            # - ----------------------------------
            if lower_limit!="none":                
                if equal == True:
                    lower_filter = df.loc[:,j]<=lower_limit
                if equal == False:
                    lower_filter = df.loc[:,j]<lower_limit
                    
                # info,
                info_lower_filter=lower_filter.sum()
                df.loc[list(lower_filter),j]=replace_with
                
                
            # * replace values > or >= upper limit,
            # - ----------------------------------
            if upper_limit!="none":  
                if equal == True:
                    upper_filter = df.loc[:,j]>=upper_limit
                if equal == False:
                    upper_filter = df.loc[:,j]>upper_limit
                    
                # info,
                info_upper_filter=upper_filter.sum()
                df.loc[list(upper_filter),j]=replace_with  
            
            # * find how many values were replaced, and add that to the total_count list 
            total_count.append(info_upper_filter+info_lower_filter)
            
            # * display examples for 3 first columns with replaced values,
            if verbose==True:
                if info_upper_filter+info_lower_filter>0 and count <4:
                    print(f"eg: {i}, {j}  : {info_lower_filter} values <{lower_limit}, ...{info_upper_filter} values <{upper_limit}")
            else:
                pass

            # * add 1 to count, to limit the number of displayed examples,
            count += 1    
                
        else:
            if verbose==True:
                print(f"{i, j} is not of numeric type, values were not replaced !")
            else:
                pass
            
    # .. additional message, if more then 2 columns had replaced values, 
    if verbose==True:
        if len(total_count)>3 and pd.Series(total_count).sum()>0:
            print(f". and {len(total_count)-3} other columns had in total  {pd.Series(total_count).sum()} replaced values \n")

        # .. message in case no values vere replaced at all, 
        if pd.Series(total_count).sum()==0:
            print("No values were replaced in requested columns....")
            
    else:
        pass
    
    # .. return, 
    return df.copy()
  
  
  
  

  
  
# function, ...................................................
def drop_nan(df, method="any", row=True, verbose=True):    
    '''
         function to dropna with thresholds from rows and columns
         . method
             . any : row/column wiht any missing data are removed
             . all : row/column only wiht missing data are removed
             . int, >0 : keeps row/clumns wiht this or larger number of non missing data
             . float, >0 : as in the above, as fraction
         
    '''
    
    assert type(df)==pd.DataFrame, "incorrect df dtype"
    df = df.copy()
    
    if verbose==True:
        print(df.shape)
    else:
        pass
    
    # set funtion for rows or columns, 
    if row==True:
        shapeidx, dfaxis = 1, 0
    else:
        shapeidx, dfaxis = 0, 1
            
    # use threshold or "all", or None for do nothing, 
    if method==None:
        pass

    elif isinstance(method, str):
        df = df.dropna(how=method, axis=dfaxis) # removes rows with NaN in all columns 

    elif isinstance(method, int):
        tr = method
        if tr==0:
            pass
        else:
            if tr>=df.shape[shapeidx]:
                tr=df.shape[shapeidx]
            else:
                pass    
            df = df.dropna(thresh=tr, axis=dfaxis) # eg  Keep only the rows with at least 2 non-NA value

    elif isinstance(method, float):
        tr = int(np.ceil(df.shape[shapeidx]*(method)))
        if tr==0:
            pass
        else:
            if tr>=df.shape[shapeidx]:
                tr=df.shape[shapeidx]
            else:
                pass    
            df = df.dropna(thresh=tr, axis=dfaxis) # eg  Keep only the rows with at least 2 non-NA value
    else:
        pass
    
    # info and return
    if verbose==True:
        print(df.shape)
    else:
        pass
    return df
  
  
  
  
  
  
  
  
# Function, ...........................................................................................
def drop_columns(*, df, columns_to_drop, verbose=True):
    """
        Small function to quickly remove columns from, 
        by column names stored in the list
        - created to give info on removed columns and whether I am chnaging df in proper way,
        - the function allows for column name duplicates, 
    """
    
    assert type(df)==pd.DataFrame, "please provide df in pandas dataframe format"
    df = df.copy()
    
    # find unique values in a list, just in case I made the mistake, 
    columns_to_drop = list(pd.Series(columns_to_drop).unique())

    # .. info, header, 
    if verbose==True:
        print(f"""Removing {len(columns_to_drop)} columns from df""") 
    else:
        pass

        
    # remove columns one by one,  
    for i,j  in enumerate(columns_to_drop):
        try:
            df.drop(columns=[j], axis=1, inplace=True)
            if verbose==True:
                print(f"{i} removing: {j}, ==> new df.shape: {df.shape}")
            else:
                pass
            
        except:
            if verbose==True:
                print(f"{i} .... column: {j}, was not found in df, check if name is correct....")
            else:
                pass
            
    return df

