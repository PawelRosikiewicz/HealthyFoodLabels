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

from matplotlib import colors
import matplotlib.patches as patches
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype

from src.utils.DataFrameExplorer_summary import find_and_display_patter_in_series
from src.utils.DataFrameExplorer_summary import create_class_colors_dict
from src.utils.DataFrameExplorer_summary import annotated_pie_chart_with_class_and_group
from src.utils.DataFrameExplorer_summary import prepare_img_classname_and_groupname
from src.utils.DataFrameExplorer_summary import annotated_pie_chart
from src.utils.DataFrameExplorer_summary import annotated_barplot
from src.utils.DataFrameExplorer_summary import df_summary_table
from src.utils.DataFrameExplorer_summary import df_summary_plot



  

# Function, ...........................................................................................
def calculate_similarity_index(*, df, df_ex, select_in_df="all", df_row_perc=10, verbose=False, detailed_info_verbose=False):
    """ 

        Perfomrs cross-validation of all or selected columns in df with any dtype 
        (text, dattime and numeric), and returns similarity index for each comparison (0, 1]
        
                           
        Parameters/Input            
        _________________   _______________________________________________________________________________     

        . Input .
        * df                DataFrame, column names are stributes that will be cross-validate
        * df_ex             Summary DataFrame, for the above df, where each row, describes one column in the above df
        
        
        . Parameters .
        * select_in_df      {"all", "text", "datetime", "numeric"}, 
                            name of dtype in df, "all" by default
        * df_row_perc       "all", or int >0 & <100, percentage of randomly selected rows in df, used to reduce 
                            the size of compared 
        * display_messages  Bool, if True, progress messages on each comparison are diplayed

        Returns               
        _________________   _______________________________________________________________________________
        
        * DataFrame         with two statistics, col1_unique_el_nr/col1+col2_unique_el_nr and 
                            col2_unique_el_nr/col1+col2_unique_el_nr
                            names of compared columns in df, raw data, and number of unique records 
                            in each combination of columns

        Returned Values              
        _________________   _______________________________________________________________________________     

        * Similarity Index  if SI =1, unique items, suplicated in different columns are distributed in the same way
                                      ie. these columns may be a potential duplicates
                            if SI ~0, all groups of unique record in both columns are distributed randomly 
                            
        * NaN               NaN are removed in each combination of compared column, only if present in both columns

                            
    """
    

    # FIRST TEST:  check if df and df_ex contain the same data, 
    if df.shape[1]!=df_ex.shape[0]: 
        if verbose==True:
            print(f"\n\n - ERROR - please meke sure df and df_ex arer from the same dataset ! \n\n")
        else:
            pass
    
    else: 
        
        # .. YOU CONTINUE, and you may extract list with column names to cross-validate,  
        if select_in_df!="all": 
            cols_to_crossvalidate = list(df_ex.name[(df_ex.summary!="all nonnull values are unique") & (df_ex.dtype==select_in_df)])
        else: 
            cols_to_crossvalidate = list(df_ex.name[df_ex.summary!="all nonnull values are unique"])
            
            
        # .. SECOND TEST: check if you have at least two columns to compare,
        if len(cols_to_crossvalidate)<2:
            
            # .. YOU HAVE TO SPOT HERE because there is nothing to cross-validate ! 
            if verbose==True:
                print(f"\n\n - ERROR - you have less then 2 columns to compare, maybe one of them had only unique information and was removed ! \n\n")
            else:
                pass
        # .. if yes continue and perform cross-validation, 
        else: 
            # .. YOU CONTINUE, and perfomr at least one comparison, each comparison is again evaluated, individually,
            
            
            # ....................................................................................................     
            # Extract sub-set of rows, if requested, this allow to speed up the process, 
                        
            # .... reduce df size, ie. row nr to speed up, with df_row_perc parameter, 
            if df_row_perc=="all": 
                "all rows will be used to calaulate SI"
            if df_row_perc==100: 
                "all rows will be used to calaulate SI"
            else:
                dfseq = list(range(df.shape[0]))
                row_list = random.sample(dfseq, k= int(df.shape[0]*df_row_perc/100))
                df = df.iloc[row_list,:] # done:


                
            # ....................................................................................................     
            # build object to store the results, and print info,
            
            # .... First, find all unieque combinations of indexes, of cols to crossvalidate, 
            n = len(cols_to_crossvalidate)
            a = list(range(n))
            b = list(range(n))
            combination_list = []
            for i, ai in enumerate(a):
                for j, bj in enumerate(b):
                    combination_list.append((ai, bj))
                    if j==len(b)-1: b.remove(ai) # to have only unique combinations            
            
            
            # .... create df to store results,
            cv_results_list = [0]*len(combination_list)
            cv_results_df = pd.DataFrame({"col1/col1col2":cv_results_list, "col2/col1col2":cv_results_list, "nr_of_compared_items":cv_results_list,
                                          "col1_name":cv_results_list, "col2_name":cv_results_list, "col1_grid_pos":cv_results_list, "col2_grid_pos":cv_results_list,
                                          "col1_class_nr":cv_results_list, "col2_class_nr":cv_results_list,"col1col2_class_nr":cv_results_list,
                                          "stat_mean":cv_results_list,"pcr5":cv_results_list})
            # .... and short info to display,
            if verbose==True:
                print(f"\n Performing Cross-validation of {cv_results_df.shape[0]} combinationas of columns in df ::::::::""")
            else:
                pass
        
            # ....................................................................................................     
            # Run cross-validation, for loop over it, 
                  
            # .... run for loop, ouver each comparison, 
            for i, cv_comb in enumerate(combination_list):    

                # ...... more detailed info to display,
                if detailed_info_verbose==True:
                    i_space = np.array([int(x) for x in list(np.ceil(np.linspace(0,len(combination_list), 10)).clip(0,len(combination_list)))])
                    i_space_names = pd.Series([ "".join([str(x),'%']) for x in list(range(0,101,10)) ])
                    print(f"{i}; ", end="")
                    if np.sum(i_space==i)>0: 
                        print(f"""{i_space_names.loc[list(i_space==i)].values[0]} eg: {cols_to_crossvalidate[cv_comb[0]]} vs {cols_to_crossvalidate[cv_comb[1]]} at {pd.to_datetime("now")}""",end='\n')
                    else:
                        pass
                else:
                    pass
                   
                
                # ....................................................................................................     
                # Extract two columns and remove rows with nan in both of them,   
                    
                # ...... extract pair of compared columns, and put info into results df, 
                two_cols_df = df.loc[:,[cols_to_crossvalidate[cv_comb[0]], cols_to_crossvalidate[cv_comb[1]]]]
                two_cols_df.columns=["one", "two"]
                cv_results_df.iloc[i,[3,4]]=[cols_to_crossvalidate[cv_comb[0]], cols_to_crossvalidate[cv_comb[1]]]
                cv_results_df.iloc[i,[5,6]]=[cv_comb[0], cv_comb[1]]

                
                # ...... remove paris of NaN from compared columns, 
                two_cols_df = two_cols_df.dropna(how="all", axis=0)
                two_cols_df.reset_index(drop=True) 
                
                
                # ...... add number of compared items, 
                cv_results_df.iloc[i, 2] = two_cols_df.shape[0]

               
            
                # ....................................................................................................     
                # Test if you have anything left to wrk with,                
                
                # ...... continue if you have any data to compare,
                if two_cols_df.shape[0]<=1:
                    cv_results_df.iloc[i,[0,1]] = [np.nan, np.nan] # place nan if there is nothign to compare !
                    cv_results_df.iloc[i, 11] = np.nan
                    cv_results_df.iloc[i, 10] = np.nan
                    # done, no more work with that pair, 
                    
                else:
                    
                    # ....................................................................................................
                    # Find information require for calulating similarity index, 
                    
                    
                    # .......... start by replacing non-duplicated misssing data in each column, wiht some string, 
                    #            to not risk having zero groups!
                    two_cols_df.loc[two_cols_df.one.isna(), "one"]="NaGroup"
                    two_cols_df.loc[two_cols_df.two.isna(), "two"]="NaGroup"

                    # .......... count unique items in each column and in combined column, 
                    c1_unique_el    =  two_cols_df.groupby(["one"]).size().shape[0]
                    c2_unique_el    =  two_cols_df.groupby(["two"]).size().shape[0]
                    c1c2_unique_el  =  two_cols_df.groupby(["one", "two"]).size().shape[0]
                    cv_results_df.iloc[i,[7,8,9]]=[c1_unique_el, c2_unique_el, c1c2_unique_el]

                    
                    
                    # ....................................................................................................
                    # calculate similarity indexes, with different methods,
                               
                        
                    # +++++++++ similiraty index for each row in each combination, 
                    cv_results_df.iloc[i,[0,1]] = [c1_unique_el/c1c2_unique_el, c2_unique_el/c1c2_unique_el]
                    cv_results_df.iloc[i,[5,6]]=[cv_comb[0], cv_comb[1]]
                    
                    # +++++++++ mean,
                    cv_results_df.iloc[i,10] = (c1_unique_el/c1c2_unique_el + c2_unique_el/c1c2_unique_el)/2
                    
                    # ++++++++ Use proportional conflict redistribution rule no. 5 (PCR5) (Smarandache & Dezert, 2006) 
                    #          to combine two SI values, for each colummn combination, creates log distrib over two combined values,
                    A = float(c1_unique_el/c1c2_unique_el)
                    B = float(c2_unique_el/c1c2_unique_el)

                    if A+B==2: PM = 1 
                    if A>0 and B>0 and (A+B)<2: PM = (A*B) + ((A**2)*(1-B))/(A+(1-B)) + ((B**2)*(1-A))/(B+(1-A))
                    if A+B==0: PM = 0

                    cv_results_df.iloc[i,11] = PM

            # .... return,
            if verbose==False:
                pass
            
            return cv_results_df.copy()

            

            

# Function, ...........................................................................................
def plot_heatmap_with_cv_results(*,df, fig_size=(12,12), title_fontsize=20, axes_fontsize=8, method="pcr5"):
    
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          Generate Traangle shaped Heatmap with similarity 
                            index calulated for dataframe columns
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        
        * df                DataFrame, returned by calculate_similarity_index()
        .
        * fig_size          tuple, (int, int)
        * title_fontsize    int, fontsize for plot titlke and axes labels
        * axes_fontsize     int, fontsize for tick labesl and labels of pixes on the plot
        * method            str, {"mean", "pcr5", "min", "max"}; what values to display on heatmap
                            . pcr5 - SI calulated with proportional conflict redistribution rule no. 5 (PCR5) (Smarandache & Dezert, 2006)¨
                            . mean - SI is the mean of two individual SI values found for each dataset.
                            . min  - lower individual SI
                            . max - upper individual SI
                            
        Returns             
        _________________   _______________________________________________________________________________
        
        * plt.figure
        * comments          no colorbar available in that version,
        
        
    """
 
    # ...............................................................................
    # format data,
    
    # .. select values to display
    if method=="pcr5": si_values = pd.Series(df.pcr5)
    if method=="mean": si_values = pd.Series(df.stat_mean)
    if method=="min":  si_values = pd.Series(df.iloc[:,[0,1]].min(axis=1))
    if method=="max":  si_values = pd.Series(df.iloc[:,[0,1]].max(axis=1))    
    
    # .. create array for the result, lower part will be empty because we did all comarisons just once, 
    arr_res = np.zeros((df.col1_grid_pos.max()+1, df.col2_grid_pos.max()+1))
    arr_res.fill(-1) # -1 so everything is now white on a heatpmap.

    # .. fill in arr_res with result, 
    for i in range(df.shape[0]):
        arr_res[int(df.col1_grid_pos[i]), int(df.col2_grid_pos[i])] = float(si_values[i])

    # .. reverse complement, to nicely position triangle heatmap, on upper left corner of the plot, 
    arr_res = arr_res[:, ::-1]    
  
    # .. ticklabels -  ie. find column names associasted with each index, 
    col_name_and_index_pos = df.groupby(['col1_name','col1_grid_pos']).size().reset_index().rename(columns={0:'count'}).iloc[:,[0,1]]
    col_name_and_index_pos = col_name_and_index_pos.sort_values(by="col1_grid_pos")

       
    # ...............................................................................
    # figure,
     
    # .. colors, 
    #.        on heatmap, all empty cells shoudl be white,
    #.        import matplotlib as mpl; my_cmap = mpl.colors.ListedColormap(['white', 'white', 'white', 'yellow', 'orange', "darkred"])
    #.        https://www.pluralsight.com/guides/customizing-colormaps
    bottom = plt.cm.get_cmap('YlOrRd', 128)
    top = plt.cm.get_cmap('binary', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    my_cmap = mpl.colors.ListedColormap(newcolors, name='binary_YlOrRd')

    # .. figure, 
    sns.set_context(None)
    fig, ax = plt.subplots(figsize=fig_size, facecolor="white")
    im = plt.imshow(arr_res, aspect = 1, origin="upper", interpolation='nearest', cmap=my_cmap)
    fig.suptitle("""Similarity Index - Modified Jaccard Similarity Index - (0,1]\nif SI==1, the classes in two compared columns are distributed in the same way\n if SI->0, the classes in two compared columns are ideally mixed with each other""", fontsize=title_fontsize)
 

    # ...............................................................................
    # ticks and other aestetics, 
    
    # .. build groups in tick positions to separate 
    ax.set_xticks(np.arange(col_name_and_index_pos.shape[0]))
    ax.set_yticks(np.arange(col_name_and_index_pos.shape[0]))

    # .. and label them with the respective list entries
    ax.set_xticklabels(col_name_and_index_pos.iloc[::-1,0], fontsize=axes_fontsize)
    ax.set_yticklabels(col_name_and_index_pos.iloc[:,0], fontsize=axes_fontsize)

    # .. select tick labels colots
    tick_pos = list(range(col_name_and_index_pos.shape[0]))
    tick_label_colors =[]
    step=5 # how many times each color will be repated
    for i in range(10000):
        tick_label_colors.extend(["black"]*step)
        tick_label_colors.extend(["red"]*step)
        #    tick_label_colors  = tick_label_colors[0:col_name_and_index_pos.shape[0]]
    tick_label_colors = tick_label_colors[0:len(tick_pos)] 
        #    I produces large number of color combinations, and now I am cutting it as it shoudl be,

    # .. modify ticklabels colors, 
    for xtick, ytick, xcolor, ycolor in zip(ax.get_xticklabels(), ax.get_yticklabels(), 
                                            tick_label_colors, tick_label_colors[::-1]):
        xtick.set_color(xcolor)
        ytick.set_color(ycolor)

    # .. Rotate the tick labels and set their alignment,
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right", rotation_mode="anchor")

    # .. Remove ticks, and axes that you dot'n want, format the other ones,
    ax.spines['top'].set_visible(False) # remove ...
    ax.spines['right'].set_visible(False) # remove ...
    ax.yaxis.set_ticks_position('none')# shows only that
    ax.xaxis.set_ticks_position('none')# shows only that
     
        
    # ...............................................................................
    # add text annotation to selected cells, to not have problems with colorbar,  
    
    # .. Loop over data dimensions and create text annotations,
    for i in range(arr_res.shape[0]):
        for j in range(arr_res.shape[1]):
            if arr_res[i, j]<0.85: 
                pass
            else:
                text = ax.text(j, i, np.round(arr_res[i, j],2), ha="center", va="center", color="white", fontsize=axes_fontsize, zorder=10)

    # .. spine description
    ax.set_xlabel("Column name", fontsize=title_fontsize )
    ax.set_ylabel("Column name", fontsize=title_fontsize )
                
                 
    # ...............................................................................
    # grid,  
    
    # .. add, pseudogrid from lines, 
    for xy in range(arr_res.shape[0]+1):
        plt.axvline(x=xy+0.5, color="lightgrey", linestyle="-" )
        plt.axhline(y=xy+0.5, color="lightgrey", linestyle="-" )      

    # .. finally add, THICKED pseudogrid from lines and DOTS, to separate differently colored ticklabels,  
    for xy in range(-1,arr_res.shape[0]+1,5):
        lcolor = "black"
        plt.axvline(x=xy+0.5, color="orange", linestyle="-", linewidth=1)
        plt.axvline(x=xy+0.5, color=lcolor, linestyle=":", linewidth=1)
        #
        plt.axhline(y=xy+0.5, color="orange", linestyle="-", linewidth=1) 
        plt.axhline(y=xy+0.5, color=lcolor, linestyle=":", linewidth=1)  
  
    
    # ...............................................................................
    # show,  
    
    # show the figure
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()





# Function, ...........................................................................................
# Function, ...........................................................................................
def show_similar_columns(*, df_summary, df_cv_results, SI_threshold=0.9, display_table=False):
    """
        Function that uses Cross-Valiadation results and data_examples
        to display columns/attribtes of Dataframe that are potentially duplicated
        
        Parameters/Input              
        _________________   _______________________________________________________________________________        
        * df_summary        DataFrame with large Dataframe summary, 
                            generated with  summarize_data_and_give_examples()
        * df_cv_results     DataFrame with Similarity Index returned by 
                            calculate_similarity_index()
        
        * SI_threshold      SI cutoff, it works on mean SI calulated with both compared columns in each pair
                            
        
        Returns               
        _________________   _______________________________________________________________________________
        
        * DataFrame         non-identical with column pairs, with SI>=SI_threshold
    """    
      
    # extract relevant info
    similar_cols_df = df_cv_results.loc[df_cv_results.stat_mean >= SI_threshold, ["col1_name", "col2_name", "pcr5"]]
    similar_cols_df = similar_cols_df.loc[(similar_cols_df.col1_name==similar_cols_df.col2_name)==False,].sort_values("pcr5", ascending=False)
    similar_cols_df.reset_index(drop=True, inplace=True)

    # add examples
    for i in range(len(similar_cols_df)):
        ex1 = df_summary.loc[(df_summary.name==similar_cols_df.col1_name.iloc[i]),"examples"]
        ex2 = df_summary.loc[(df_summary.name==similar_cols_df.col2_name.iloc[i]),"examples"]
        if i==0:
            ex1list = [ex1] 
            ex2list = [ex2]
        else:
            ex1list.append(ex1)
            ex2list.append(ex2)

    # add exampes to df,
    similar_cols_df = pd.concat([similar_cols_df , pd.Series(ex1list), pd.Series(ex2list)], axis=1, sort=False) 
    similar_cols_df.columns=["attribute 1", "attribute 2", "similarity index", "examples attribute 1", "examples attribute 2"]

    if display_table==True:
        print(f"""\n{"".join(["-"]*50)} \n Pairs of Columns with Similarity Index >={SI_threshold}\n{"".join(["-"]*50)}""")
        display(similar_cols_df.iloc[:,0:3])
    else:
        pass
    
    # return
    return similar_cols_df

