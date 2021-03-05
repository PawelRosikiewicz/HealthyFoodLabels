# ********************************************************************************** #
#                                                                                    #
#   Project: Heathly Food Labels analysis using food facts dataset                   # 
#            helpr functions,                                                        # 
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

from src.utils.DataFrameExplorer_summary import annotated_pie_chart_with_class_and_group
from src.utils.DataFrameExplored_dfprep import find_and_display_patter_in_series
from src.utils.DataFrameExplored_dfprep import load_csv
from src.utils.DataFrameExplored_dfprep import find_patter_in_series
from src.utils.DataFrameExplored_dfprep import format_to_datetime
from src.utils.DataFrameExplored_dfprep import replace_text
from src.utils.DataFrameExplored_dfprep import replace_numeric_values
from src.utils.DataFrameExplored_dfprep import drop_nan
from src.utils.DataFrameExplored_dfprep import drop_columns





# creat save and re-load subset of food facts data
'''
    Use 'en.openfoodfacts.org.products.tsv'
    idx = np.unique(np.random.randint(0, data.shape[0], 5000)).tolist()
    data_sub = data.iloc[idx,:]
    data_sub.to_csv('small_table.tsv', sep="\t", header=True, index=False)
    
    
    # save cleaned dataset
    os.chdir(PATH_interim)
    filename = 'open_food_facts_cleaned_data.csv'
    cleaned_data.to_csv(filename, encoding='utf-8', index=False )
'''





# helper function, ...............................................................................
def format_and_clean_foodfactsdata(*, data, coltr=0.1, rowtr=0.5,  verbose=False, dfsize_verbose=False):
    '''
        Helper function, with previously selected values to replace or to remove 
        it replaces all incorrect values with NaN, not with most frequent, 
        (I preffer to have incomplete data instead inouted and incorrect values at this moment)
        Acrions:
        - formats datetime columns : timezone='UTC'
        - replace "to be completed", or similar values with NaN 
        - replace too high or too low values, eg 150g of salt in 100g of prduct wiht NaN
        - replace energy_100g <=0, wiht NaN in energy column
        - replace wiht NAN records with energy-amount > 3700kJ, in energy column
        - replace nutriscores, < -15 and >40 wiht NaN 
        
        . rowtr/coltr
             . None.     : no action, 
             . "any"     : row/column wiht any missing data are removed
             . "all"     : row/column only wiht missing data are removed
             . int, >0   : keeps row/clumns wiht this or larger number of non missing data
             . float, >0 : as in the above, as fraction
        
    '''

    assert type(data)==pd.DataFrame, "please provide data in pandas dataframe format"

    if dfsize_verbose==True:
        print(f"before chnages: {data.shape}")
    else:
        pass
    
    # format datetime columns
    data = format_to_datetime(data=data, pattern_list=r'datetime', timezone='UTC', unixtime=False, verbose=verbose)
    data = format_to_datetime(data=data, pattern_list=r'_t$', timezone='UTC', unixtime=True, verbose=verbose)

    # replace "to be completed", or similar with NaN 
    dct = {        "states" : ".+to-be-completed.+|.+to-be-checked.+|.+be checked.+|.+be completed.+", 
               "pnns_groups":"unknown"}
    for k,v in dct.items():
        colnames = find_patter_in_series(s=pd.Series(data.columns), pat=k )
        data = replace_text(df=data ,pat=v, colnames=colnames, fillna=np.nan, verbose=verbose) 


    # replace too high or too low values, eg 150g of salt in 100g of prduct wiht NaN
    pat = r".+_100g"
    pat_to_exlude = r"energy|carbon-footprint|nutrition-score|water-hardness|glycemic-index|collagen-meat-protein-ratio"
    cols = find_patter_in_series(s=pd.Series(data.columns), pat=pat, tolist=False)
    cols = list(cols[cols.str.contains(pat_to_exlude)==False])
    data = replace_numeric_values(df=data, colnames=cols, lower_limit=0, upper_limit=100, replace_with=np.nan, verbose=verbose)

    # energy column
    cols = find_patter_in_series(s=pd.Series(data.columns), pat=r"energy")

    # .. replace energy_100g <=0, wiht NaN
    data = replace_numeric_values(df=data, colnames=cols, lower_limit=0,  equal=True, replace_with=np.nan, verbose=verbose)

    # .. replace wiht NAN records with energy-amount > 3700kJ, 
    'it is a theoretical maximum amount of energy that any product can have, if it woudl be made completely of fat'
    data = replace_numeric_values(df=data, colnames=cols,  upper_limit=float(3700), equal=False, replace_with=np.nan, verbose=verbose)

        
    # nutriscores, should be between -15 and 40 
    cols = find_patter_in_series(s=pd.Series(data.columns), pat=r"nutrition-score")
    data = replace_numeric_values(df=data, colnames=cols, lower_limit=-15, upper_limit=40, replace_with=np.nan, verbose=verbose)
    
    # remove duplicates
    data = data.drop_duplicates(keep='first')
    
    # drop rows and columns with nan
    data = drop_nan(df=data, method=coltr, row=False, verbose=False) # columns with >90% nan will be removed
    data = drop_nan(df=data, method=rowtr, row=True, verbose=False) # rows with >50% nan will be removed
    
    
    if dfsize_verbose==True:
        print(f"after removing nan and duplicates: {data.shape}")
    else:
        pass    
    
    return data


    


    
    
# helper function, ..............................................................................
def drop_selected_columns_in_foodfactsdata(*, df, columns_to_drop="default", verbose=False, dfsize_verbose=False):
    '''
        allows removing many columns, using column str wiht one, or list wiht namy column,
        if applied with columns_to_drop="default", it will use list with column names that were foudn as duplicates or with 
        unique ID names provided wiht the foodfacts database, that were not used in my analysis
        I also removed columns with majority of missing data, 
    '''
    
    assert type(df)==pd.DataFrame, "please provide df in pandas dataframe format"
    df = df.copy()
    
    # select list of columns to remove, 
    if isinstance(columns_to_drop,str):
        if columns_to_drop=="default":
            preselected_columns_to_drop = [
                "main_category", "emb_codes_tags", 
                "additives_tags", "traces_tags", 
                "categories_tags", "countries_tags", 
                "labels_tags","origins_tags", 
                "brands_tags", "packaging_tags",
                "labels", "categories_tags", 
                "categories", "traces_tags", 
                "traces", "created_t",
                "last_modified_t", "countries", 
                "manufacturing_places_tags"]
        else:
            preselected_columns_to_drop=[columns_to_drop] # trun str into the list
    else:
        preselected_columns_to_drop=columns_to_drop
    
    # remove selected columns
    df = drop_columns(df=df, 
                 columns_to_drop=preselected_columns_to_drop, 
                 verbose=verbose)
 
    
    if dfsize_verbose==True:
        print(f"after removing selected columns: {df.shape}")
    else:
        pass    


    return df
  
  
  
# Function, ............................................................................
def find_and_display_patter_in_series(*, series, pattern):
    "I used that function when i don't remeber full name of a given column"
    res = series.loc[series.str.contains(pattern)]
    return res






# helper function, .....................................................
def scatterplot_nutriscores(df):
    '''
        creates scatterlot using numberic dtypes in the first two columns in df
        the first column is on x-axis, and the second column is on y-axis
    '''
    fig, ax = plt.subplots(nrows=1, ncols=1, facecolor="white", figsize=(6,5))
    ax.scatter(x=df.iloc[:,0], y=df.iloc[:,1], 
               marker=".", c="steelblue", s=25 , edgecolor="steelblue")
    ax.set_xlim(-15,40)
    ax.set_ylim(-15,40)
    ax.set_xlabel(df.columns[0],fontsize=12, color="darkred")
    ax.set_ylabel(df.columns[1],fontsize=12, color="darkred")
    sns.despine()  
    plt.show();

    
    
    


# helper function, .....................................................
def compare_two_features_on_annotated_pie_charts(*,
    df,
    compared_features = None,
    group_features = None, 
    colors = {"equal values":"forestgreen", "different v.":"darkred"},
    threshold = 100,
    plot_all_groups = False,
    first_pieChart_dct = dict(title_ha='center'),
    second_pieChart_dct = dict(n_subplots_in_row=4, title_ha='center', ax_title_fonsize_scale=0.5),
    first_plot = True,
    first_pieChart_title = None, 
    second_pieChart_title = None,      
    min_group_size=None,
):
    '''
        helper function that will find equal and fifferent values in two compared comlumns, 
        the results will be plotted as annotated pie chart, for all values found in that column
        
        you may add, group_features (str, or lict wiht strings), that will allos dividing
        compared valuas in two or more groups
        
        limitations: please give some str names to columns in df,     
    '''

    assert type(df)==pd.DataFrame, "Error, df shoudl be pandas data frame"

    if compared_features is None:
        compared_features = list(df.columns)[0:2]
    else:
        pass
    
    if min_group_size is None:
        min_group_size = 0
    else:
        pass
    
    
    # find equal values, 
    equal_values = pd.Series(df.loc[:,compared_features[0]]==df.loc[:,compared_features[1]])
    equal_values[equal_values==True]=list(colors.keys())[0]
    equal_values[equal_values==False]=list(colors.keys())[1]

    # first make a general plot                                     
    if first_plot==True:
        # set title, 
        if first_pieChart_title is None:
            pc_title="Total number of different values"
        else:
            pc_title=first_pieChart_title
            
        annotated_pie_chart_with_class_and_group(
            title=pc_title,
            classnames=equal_values.values.tolist(), 
            class_colors=colors,
            **first_pieChart_dct
            )
    else:
        pass

    # find equal values,
    df_full = pd.concat([df, equal_values], axis=1)

    # plot how many different results are in copared features in each group
    if group_features is None:
        pass

    else:
        # make sure, group_features is a list
        if isinstance(group_features, str):
            group_features = [group_features]
        else:
            pass

        # select groups in a given feature that gave at least some chages, and plot them in that order,  
        for one_group in group_features:

            one_group_column = df_full.loc[:, one_group]
            one_group_column.loc[pd.isnull(one_group_column)]=="No Data"

            # keep only groups with changes,
            df_sub = pd.concat([equal_values, one_group_column], axis=1)
            grp = df_sub.groupby(one_group)
            grp_df_list = []
            grp_df_dct = dict()

            for name, group in grp:
                # find out how many different values is in each 
                grp_df_dct[name]=group
                grp_df_list.append(
                    {
                        name:(group.iloc[:,0]==list(colors.keys())[1]).sum()
                    })

            # order group names, 
            ordered_group_names = pd.DataFrame(grp_df_list).stack().sort_values(ascending=False)

            # rebuild group-class df using only groups wiht different items inside, 
            counter = 0
            for n_i, name in enumerate(list(ordered_group_names.index)):


                if plot_all_groups==False:
                    if ordered_group_names.loc[name]>threshold:
                        if grp_df_dct[name[1]].shape[0]>min_group_size:
                            if counter == 0:
                                selected_groups_df = grp_df_dct[name[1]]
                            else:
                                selected_groups_df = pd.concat([selected_groups_df, grp_df_dct[name[1]]], axis=0)
                            counter+=1
                        else:
                            pass 
                    else:
                        pass                    
                else:   
                    if grp_df_dct[name[1]].shape[0]>min_group_size:
                        if counter == 0:
                            selected_groups_df = grp_df_dct[name[1]]
                        else:
                            selected_groups_df = pd.concat([selected_groups_df, grp_df_dct[name[1]]], axis=0)   
                        counter+=1
                    else:
                        pass

                    
            # set title, 
            if second_pieChart_title is None:
                if plot_all_groups==False:     
                    pc_title="Number of different values in each category\ncategories, wiht no changes were omitted"        
                else:
                    pc_title="Number of different values in each category"
            else:
                pc_title=second_pieChart_title
                
            # first make a general plot                                     
            annotated_pie_chart_with_class_and_group(
                title=pc_title,
                classnames=selected_groups_df.iloc[:,0].values.tolist(), 
                groupnames=selected_groups_df.iloc[:,1].values.tolist(),
                tight_lyout=True,
                class_colors=colors,
                **second_pieChart_dct
            )

              
  
  
  

  
  
  
# Function, .........................................................................
def tokenize_series_with_relevant_data_in_index(*, ex, verbose=False):
    """
        ---------    ------------------------------------------------------------------
        Function   : Takes ingredient list, divide each substring into tokens, 
                     lowercase all, so there is no problem, removes several stop signs, NaN, 
                     "none", or substring like 5% or a, f, g made up form one letter.
                     The function also removes, or replaces, to multiple empty spaces, tabs etc...
        ---------    ------------------------------------------------------------------
        ...
        ex         : pd.Series, with food ingredient list,
        Verbose.   : bool,
        ... 
        returns.   : pd.DataSeries() or None, 
                     with the following columns: 
                     - "ingredinet_name", 
                     - "ingredient_count", 
                     - "percentage_of_products_with_the_given_imngredient", 
                     - "total_product_number"
    """
    
    #### main info:
    number_of_items_to_tokenize = int(ex.shape[0])
    number_of_empty_items = int(ex.isnull().sum())
    
    
    #### test, 

    # .. check if there are items that can be used,
    if number_of_items_to_tokenize==0:
        if verbose==True:
            print("ERROR, number_of_items_to_tokenize == 0!")
        return None
    
    # .. if thre are only missing data at each positon in series, 
    if number_of_items_to_tokenize==number_of_empty_items:
        if verbose==True:
            print("ERROR, threre are only missing data in provided Series == 0!")
        return None     
    
    # .. else do the following,
    else:
        if verbose==True:
            print(f"Cleaning ingredient list with {number_of_items_to_tokenize} items, including {number_of_empty_items} NaN, ", end="")

        
        
        #### modify and clean substrings, ....................................................
                   
        # .. turn all strings into lowercase,
        ex = ex.str.lower()

        # .. group desriptions, because they offten contain duplicates names,
        ex = ex.str.replace("\s\(.*\)", "")
        ex = ex.str.replace("\s\[.*\]", "")

        # .. items like and/or and org -> organic,
        ex = ex.str.replace(r"\sand\/or\s", ", ")
        ex = ex.str.replace(r"\sand\s", ", ")
        ex = ex.str.replace("org\s", "organic ")
        
        # .. info, small dot to not be bored,
        if verbose==True: 
            print(f" .", end="")

   

        #### Tokenize substrings, .........................................................
        
        # .. divide each substrings, and create a pd:series with one token , at each position,
        ex = ex.str.split("[\,|•|;|:|\.|-]", expand=True).stack()
        
        # .. info, small dot to not be bored,
        if verbose==True: 
            print(f" .", end="")
        
        
        
        #### move values form index into column, ..........................................

        # .. procedure,
        ex = ex.reset_index(level=[0])
        ex = ex.reset_index(drop=True)
        ex.columns = ["energy_100g","ingredient"] # rename columns,
        
  

        #### replace empty spaces, dots, comas etc.. created after string splitting, ......

        # .. procedure,
        ex.ingredient = ex.ingredient.str.replace("contains\s+or\s+less\s+of", "")
        ex.ingredient = ex.ingredient.str.replace("\d+", "")
        ex.ingredient = ex.ingredient.str.replace("\[", " ")
        ex.ingredient = ex.ingredient.str.replace("\]", " ")
        ex.ingredient = ex.ingredient.str.replace("\!", " ")
        ex.ingredient = ex.ingredient.str.replace("\%", " ")
        ex.ingredient = ex.ingredient.str.replace("\/", " ")
        ex.ingredient = ex.ingredient.str.replace("\-", " ")
        ex.ingredient = ex.ingredient.str.replace("\(", "")
        ex.ingredient = ex.ingredient.str.replace("\)", "")
        ex.ingredient = ex.ingredient.str.replace("\*", "")
        ex.ingredient = ex.ingredient.str.replace("\.$", "")
        ex.ingredient = ex.ingredient.str.replace("\s\s", " ")
        ex.ingredient = ex.ingredient.str.replace("\s$", "")
        ex.ingredient = ex.ingredient.str.replace("^\s+","")
        ex.ingredient = ex.ingredient.str.replace("_", "")
        ex.ingredient = ex.ingredient.str.replace("^\d+\s?%", "") # remove tokens like 5%, 9%, with no other decription
           

            
        #### Remove missing data, ........................................................
                
        # .. remove missing data, encoded as 'None', or found with find_nan function,
        ex = ex.loc[ex.ingredient!='None']
        ex = ex.dropna(how="all", axis=0)

        

        #### Remove all string showrten then 2, ..........................................

        # .. function for apply,
        def find_nan(x):
            if len(x)<3: return False
            else: return True

        # .. do it, 
        ex_filter = list(ex.ingredient.apply(find_nan))
        ex = ex.loc[ex_filter,:] # I had to remove that twice, and separately, or else it wa san error,       
        
        # .. info, small dot to not be bored,
        if verbose==True: 
            print(f" .", end="")
        
        
        #### Return, ......................................................................
        
        return ex

        # .. info, small dot to not be bored,
        if verbose==True: 
            print(f" done", end="\n")
            

  


  
  
        
# Function , ...............................................................................
def boxplot_X_ingredientList_Y_numericValue(*,indexed_series_with_cleaned_data, 
                                            number_of_ingredients_to_display=20,
                                            number_of_ing_to_display_on_legend = 10,
                                            Xlabel_fontsize=12,
                                            plot_title="Title", 
                                            Ylabel="Value" ,
                                            FigSize=(16,8),
                                            verbose=True):
    
    """
        ---------     ------------------------------------------------------------------
        Function   :  Creates a boxplot with list of ingredients on X axis and
                      numerical values associated with each ingredinet on Y axis
                      Boxes are ordered with median,
        ---------     ------------------------------------------------------------------
        ...
        indexed_series_with_cleaned_data    :   pd.Series, with food ingredient list,
                                                and index with numerical value that will be used to 
                                                create boxplots
        number_of_ingredients_to_display    :   int, How many ingredients will be displayed as separate boxplots
                                                default 20,
        number_of_ing_to_display_on_legend  :   int, default =10, how many most frequent ingredients 
                                                associated with prodicust with the highgest and 
                                                the lowers energy content shoudl be
                                                listed on boxplot legend 
        plot_title                          :   str, displayed as Figure title, default="Title"
        FigSize                             :   tuple, (int, int), default=(16,8)
        Ylabel                              :   str, plot ylabel, default="Value"
        Xlabel_fontsize                     :   int, xticks fontsize, default=12
        Verbose.                            :   bool,
        
        
        ... 
        returns.                            :   boxplot, figure
    """
    
    
    
    
    
    #### Step 1. Tokenize ingredients and find most common ingredients to plot, ..............

    if verbose==True: 
        print(f"Preparing data for boxplot ... ", end="")
    
    # tokenize ingredient list, without losing the info on energy content in each product, 
    tok_ingredients = tokenize_series_with_relevant_data_in_index(ex=indexed_series_with_cleaned_data, verbose=verbose)

    # find most frequent ingredients,
    counted_tok_ingredients    = tok_ingredients.groupby(by=["ingredient"]).count()
    counted_tok_ingredients    = counted_tok_ingredients.sort_values(by="energy_100g", ascending=False) # is the name given by the fuinction above
    most_frequent_ingredients  = list(counted_tok_ingredients.index[0:number_of_ingredients_to_display]) # nbut it can contain any numerical value


    
    #### Step 2. Order the data so they can be displayed as boxplots, with increasing median, ........
    
    if verbose==True: 
        print(f"\n Re-ordering the data for boxplot ...")

    # provide empty data containers for data,
    data_for_boxplot_values = []
    data_for_boxplot_means = []
    data_for_boxplot_median = []
    data_for_boxplot_std = []

    # fiil in empty container with data for boxplots, 
    for i,j in enumerate(most_frequent_ingredients):
        #print(i,j)
        j_energy = tok_ingredients.energy_100g[tok_ingredients.ingredient==j].values.flatten()
        data_for_boxplot_values.append(j_energy)
        data_for_boxplot_means.append(j_energy.mean())
        data_for_boxplot_std.append(np.std(j_energy))
        data_for_boxplot_median.append(np.median(j_energy))

    # order lists according to median values,
    box_order = list(pd.Series(data_for_boxplot_median).sort_values(ascending=True).index)
    ordered_data_for_boxplot_values = []
    ordered_ingredient_names = []
    ordered_boxplot_means = []  # for legend, later on
    ordered_boxplot_std = []    # for legend, later on
    
    for i,j in enumerate(box_order):
        ordered_data_for_boxplot_values.append(data_for_boxplot_values[j]) 
        ordered_ingredient_names.append(most_frequent_ingredients[j])  
        ordered_boxplot_means.append(data_for_boxplot_means[j])
        ordered_boxplot_std.append(data_for_boxplot_std[j])

    #### Step 3. Boxplot,.................................................................................

    if verbose==True: 
        print(f"Preparing boxplot with {plot_title} ...")    

    # fig,  
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=FigSize, facecolor="white")
    fig.suptitle(plot_title, fontsize=20)

    # boxplot and axes labels,
    box1 = ax.boxplot(ordered_data_for_boxplot_values, 
                      showfliers=False,                  # remove outliers, because we are interested in a general trend,
                      vert=True,                         # boxes are vertical
                      labels=ordered_ingredient_names,   # x-ticks labels
                      patch_artist=True);                # fill with color,
    ax.set_xlabel('Ingredient', fontsize=20)
    ax.set_ylabel(Ylabel, fontsize=20)    


    # box colors,

    # .. select colors.
    box_colors = plt.get_cmap("hsv")(np.linspace(0.4, 1, len(ordered_ingredient_names)))

    # .. add colors to each box individually,
    for i, j in zip(range(len(box1['boxes'])),range(0, len(box1['caps']), 2)) :

        median_color  ="black"
        box_color     = box_colors[i,:]

        # set properties of items with the same number as boxes,
        plt.setp(box1['boxes'][i], color=box_color, facecolor=box_color, linewidth=0)
        plt.setp(box1["medians"][i], color=median_color, linewidth=2)
        #plt.setp(box1["fliers"][i], markeredgecolor=box_color) # outliers

        # set properties of items with the 2x number of features as boxes,
        plt.setp(box1['caps'][j], color=box_color)
        plt.setp(box1['caps'][j+1], color=box_color)
        plt.setp(box1['whiskers'][j], color=box_color)
        plt.setp(box1['whiskers'][j+1], color=box_color)

        # can also be done like this:
        #for patch, box_color in zip(box1['boxes'], box_colors):
        #    patch.set_facecolor(box_color )

    # Modify ticks, axes and grid,    

    # .. Add ticks on y axis, and names for each bar,
    ax.set_xticks(list(range(1,len(ordered_ingredient_names)+1)))
    ax.set_xticklabels(ordered_ingredient_names, fontsize=Xlabel_fontsize, color="black", rotation=60, ha="right")

    # .. Format ticks,
    ax.tick_params(axis='x', colors='black', direction='out', length=4, width=2) # tick only
    ax.tick_params(axis='y', colors='black', direction='out', length=4, width=2) # tick only    
    ax.yaxis.set_ticks_position('left')# shows only that
    ax.xaxis.set_ticks_position('bottom')# shows only that

    # .. Remove ticks, and axes that you dot'n want, format the other ones,
    ax.spines['top'].set_visible(False) # remove ...
    ax.spines['right'].set_visible(False) # remove ...  
    ax.spines['bottom'].set_linewidth(1) # x axis width
    ax.spines['left'].set_linewidth(1) # y axis width 

    # .. grid and labels
    ax.yaxis.grid(True,color='grey', linestyle='--', linewidth=1)
    
    
    
    #### Step 4. Legend with info on some items,...............................................................

    # .. collect the data on ingredients associated with the lowest values, 
    bottom_ingr_list  = ordered_ingredient_names[0:number_of_ing_to_display_on_legend]
    bottom_ingr_means = ordered_boxplot_means[0:number_of_ing_to_display_on_legend]
    bottom_ingr_std   = ordered_boxplot_std[0:number_of_ing_to_display_on_legend]
    # .....
    bottom_ingr_info = ["".join([str(x),", (",str(int(y)),"±",str(int(z)), "kj)"]) for x,y,z in zip(bottom_ingr_list, bottom_ingr_means, bottom_ingr_std) ]
    bottom_box_colors = box_colors[0:number_of_ing_to_display_on_legend,:]
    
    # .. collect the data on ingredients associated with the highest values, 
    top_ingr_list     = ordered_ingredient_names[-(number_of_ing_to_display_on_legend+1):-1][::-1]
    top_ingr_means    = ordered_boxplot_means[-(number_of_ing_to_display_on_legend+1):-1][::-1]
    top_ingr_std      = ordered_boxplot_std[-(number_of_ing_to_display_on_legend+1):-1][::-1]
    # .....
    top_ingr_info = ["".join([str(x),", (",str(int(y)),"±",str(int(z)), "kj)"]) for x,y,z in zip(top_ingr_list, top_ingr_means, top_ingr_std) ]
    top_box_colors = box_colors[-(number_of_ing_to_display_on_legend+1):-1,:][::-1,:]

    
    # .. create patch list for legend, with the same ingredient colors as on each subplot,
    import matplotlib.patches as mpatches
    
    # .. Add legend to the figure
    patch_list_for_legend =[]
    for i in range(len(bottom_ingr_info)):
        ingr_patch = mpatches.Patch(color=bottom_box_colors[i,:], label="".join([str(i+1),". ",bottom_ingr_info[i]]))
        patch_list_for_legend.append(ingr_patch)
    # ....
    l = fig.legend(handles=patch_list_for_legend, loc="center", frameon=False, scatterpoints=1, ncol=1, bbox_to_anchor=(0.3, 0.75), fontsize=12)
    l.set_title("Ingredients found in products \nwith the lowest energy content")
    l.get_title().set_fontsize('12')
    
    # .. Add legend to the figure
    patch_list_for_legend =[]
    for i in range(len(top_ingr_info)):
        ingr_patch = mpatches.Patch(color=top_box_colors[i,:], label="".join([str(i+1),". ",top_ingr_info[i]]))
        patch_list_for_legend.append(ingr_patch)
    # ....
    l = fig.legend(handles=patch_list_for_legend, loc="center", frameon=False, scatterpoints=1, ncol=1, bbox_to_anchor=(0.8, 0.75), fontsize=12)
    l.set_title("Ingredients found in products \nwith the highest energy content")
    l.get_title().set_fontsize('12')

    
    #### Step 5. Layout and plot show,.........................................................................
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.6, hspace=0.5, wspace=0.6, right=1)# to adjust for main title, amd legend
    plt.show();     


            

  

        
# Function , ...............................................................................
def paired_boxplots(*,
    df, 
    pat_colname_with_all_data,
    pat_colname_with_groups,
    pat_colname_with_data_for_top_box,
    pat_colname_with_data_for_bottom_box,
    coutry_box_colors = ["blue", "red"],         
    plot_title=None, 
    pnns_grou_to_use=0, 
    make_boxplot=True, 
    figure_size=(10,12), 
    fontsize_scale=1,
    adjust_top=0.8,
    verbose=False
):

    """
        ---------    ------------------------------------------------------------------
        Function   : if make_boxplot=True,
                     creates boxplot, for pnutritional score in each pnns group (0,1)
                     each group has two boxes, one for score calulated wiht fr algorithm
                     and one for scores calculated with uk algorithm.
                     
                     if make:boxplot=False
                     function will return pd.dataFrame with pnns group names and 
                     counts, how many products were in that group, 
                     I make this to create barplots, easily
        ---------    ------------------------------------------------------------------
        ...
        df               : pd.DataFrame, with nuctri scores from fr and uk
        make_boxplot     : bool, see above
        pnns_grou_to_use : int {0,1} . there are two pnns group classificaiton in the data set, 
                           you may, use only 0, if you removed one, or >1, 
                           if you added new pnns classification, 
                           Caution: the nr of classes differe between pnns groups
        figure_size      : tuple (int, int), default as (10,12)
        verbose.         : bool,
        ... 
        returns.         : see above
                           boxplot 
                           or DataFrame and list with colors used on boxplot
    """
    
    
    
    pnns_grou_to_use=0
    

    # SELECT THE DATA, ........................

    # .. col names
    colNames_nutri_scores = list(find_and_display_patter_in_series(series=pd.Series(df.columns), pattern=pat_colname_with_all_data))
    colNames_product_groups = list(find_and_display_patter_in_series(series=pd.Series(df.columns), pattern=pat_colname_with_groups))
       #   0.  'pnns_groups_1'   # 13 classe 
       #   1.  'pnns_groups_2'   # 41 classes

    # .. build df for plot
    colNames_nutri_scores.append(colNames_product_groups[pnns_grou_to_use])
    df_data_for_plot = df.loc[:,colNames_nutri_scores]


    # PPERARE THE DATA FOR PLOT, .............

    # .. replace missing data in pnns groups with "No Group"
    df_data_for_plot.loc[:,colNames_product_groups[pnns_grou_to_use]] = df_data_for_plot.loc[:,colNames_product_groups[pnns_grou_to_use]].fillna("no group")

    # .. remove missing data,  
    df_data_for_plot = df_data_for_plot.dropna(how="any", axis=0)
    #df_data_for_plot.set_index(colNames_product_groups[pnns_grou_to_use], inplace=True)


    # .. make all letters, lowercase, to avoid having classes with lowe/upper case difference in one or more letters, 
    df_data_for_plot.loc[:,colNames_product_groups[pnns_grou_to_use]] = df_data_for_plot.loc[:,colNames_product_groups[pnns_grou_to_use]].str.lower()


    # .. prepare the data for boxplot in lists, 

    # ...........
    box_names = df_data_for_plot.loc[:,colNames_product_groups[pnns_grou_to_use]].unique().tolist()
    list_boxplot_data_fr = list()
    list_boxplot_data_uk = list()
    list_boxplot_group_size = list()
    list_boxplot_median_fr = list()
    list_boxplot_median_uk = list()

    # ...........
    for i,j in enumerate(box_names):

        # preparing, the data and filters,
        col_name_fr= list(find_and_display_patter_in_series(series=pd.Series(df_data_for_plot.columns), pattern=pat_colname_with_data_for_top_box))[0]
        col_name_uk= list(find_and_display_patter_in_series(series=pd.Series(df_data_for_plot.columns), pattern=pat_colname_with_data_for_bottom_box))[0]
        row_filter = list(df_data_for_plot.loc[:,colNames_product_groups[pnns_grou_to_use]]==j)

        # subset data for uk and fr,
        df_subset  = df_data_for_plot.loc[row_filter,:]

        # fill in,
        list_boxplot_group_size.append(pd.Series(row_filter).sum())  
        list_boxplot_data_fr.append(df_subset.loc[:,col_name_fr].values.flatten())
        list_boxplot_data_uk.append(df_subset.loc[:,col_name_uk].values.flatten())

        # add meedian or mean, depending how many values you have in each group,
        if pd.Series(row_filter).sum()==0:
            list_boxplot_median_fr.append(40)
            list_boxplot_median_uk.append(40)

        if pd.Series(row_filter).sum()>2:
            list_boxplot_median_fr.append(np.median(df_subset.loc[:,col_name_fr].values))
            list_boxplot_median_uk.append(np.median(df_subset.loc[:,col_name_uk].values))

        if pd.Series(row_filter).sum()>0 and pd.Series(row_filter).sum()<=2:
            list_boxplot_median_fr.append(np.mean(df_subset.loc[:,col_name_fr].values)) # use mean, instead!
            list_boxplot_median_uk.append(np.mean(df_subset.loc[:,col_name_uk].values)) # use mean, instead!

        # info,
        if verbose==True:
            print(f"{i} {j}, {pd.Series(row_filter).sum()} items")




    # REORDER THE DATA ..........................

    # .............
    new_box_order = list(pd.Series(list_boxplot_median_fr).sort_values(ascending=False).index)

    # .............
    sorted_box_names = list()
    sorted_list_boxplot_group_size = list()
    sorted_list_boxplot_data_fr = list()
    sorted_list_boxplot_data_uk = list()
    sorted_list_boxplot_median_fr = list()
    sorted_list_boxplot_median_uk = list()    

    # .............
    for i,j in enumerate(new_box_order):
        sorted_box_names.append(box_names[j])
        sorted_list_boxplot_group_size.append(list_boxplot_group_size[j])
        sorted_list_boxplot_data_fr.append(list_boxplot_data_fr[j])
        sorted_list_boxplot_data_uk.append(list_boxplot_data_uk[j])
        sorted_list_boxplot_median_fr.append(list_boxplot_median_fr[j])
        sorted_list_boxplot_median_uk.append(list_boxplot_median_uk[j])  

    sorted_box_names_old = sorted_box_names 

    
    
    # BOXPLOT ...................................
    
    # ....... Prep the data .....................
    
    # prepare data for boxes from fr and uk nutri scores,
    data_list        = [sorted_list_boxplot_data_fr, sorted_list_boxplot_data_uk]
    # .......
    box_positions_fr      = [x*4   for x in list(range(1,len(sorted_list_boxplot_data_fr)+1))]
    box_positions_uk      = [x*4-1 for x in list(range(1,len(sorted_list_boxplot_data_fr)+1))] 
    grid_hlines_positions = [x*4-2.5 for x in list(range(1,len(sorted_list_boxplot_data_fr)+1))] 
    # .......
    sorted_box_names    = sorted_box_names_old # because I chnage many thinks,
    # .....
    counted_items = [len(x) for x in sorted_list_boxplot_data_fr]
    sorted_box_names = ["".join([str(x)," (", str(y), " items)"]) for x,y in zip(sorted_box_names, counted_items)]
    
    # choose colors, 
    box_colors = plt.get_cmap("Dark2")(np.linspace(0.2, 1, len(sorted_box_names)))


    #### because I was an opportunist, 
    #.   and it was easier to do that, than to write another function from the beginning
    if make_boxplot==True:

        # ....... fig, ax, box colors .....................

        # fig, ax
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figure_size, facecolor="white")
        
        if plot_title is not None:
            fig.suptitle(plot_title, fontsize=16*fontsize_scale)
        else:
            pass
    
        # ....... add two series of boxes ................

        # draw two plots, on different ax, obj's,
        for box_positions, one_country_box_color, sorted_list_boxplot_data in zip([box_positions_fr, box_positions_uk], 
                                                            coutry_box_colors,
                                                            data_list):

            # add boxes,
            box1 = ax.boxplot(sorted_list_boxplot_data, 
                                  showfliers=False,                  # remove outliers, because we are interested in a general trend,
                                  vert=False,                        # boxes are vertical
                                  labels=sorted_box_names,           # x-ticks labels
                                  patch_artist=True,
                                  positions = box_positions,
                                  widths=0.3
                             )
            

            # add colors to each box individually,
            for i, j in zip(range(len(box1['boxes'])),range(0, len(box1['caps']), 2)) :
                median_color  ="black"
                # 2021.03.06 here I replaced box_colors[i,:] wiht one color for top and bottom boxes, in each pair
                box_color     = one_country_box_color # box_colors[i,:]

                # set properties of items with the same number as boxes,
                plt.setp(box1['boxes'][i], color=box_color, facecolor=median_color, linewidth=2)
                plt.setp(box1["medians"][i], color=median_color, linewidth=2)
                #plt.setp(box1["fliers"][i], markeredgecolor=box_color) # outliers

                # set properties of items with the 2x number of features as boxes,
                plt.setp(box1['caps'][j], color=median_color)
                plt.setp(box1['caps'][j+1], color=median_color)
                plt.setp(box1['whiskers'][j], color=median_color)
                plt.setp(box1['whiskers'][j+1], color=median_color)



        # ....... add ticks to y axis .....................
        tick_positions = [x*4-0.5 for x in list(range(1,len(sorted_list_boxplot_data_fr)+1))]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(sorted_box_names , fontsize=10*fontsize_scale, color="black", ha="right")
        ax.tick_params(axis='y', colors='black', direction='out', length=2, width=1) # tick only         



        # ....... limits, labels .....................

        # set limits
        ax.set_xlim(-15,40)
        ax.set_ylim(0,len(sorted_box_names)*4+1)

        # axes labels,
        ax.set_xlabel('Nutritional Score: \nfrom -15 (best for consumer)\nto 40 (worst for consumer)', fontsize=15*fontsize_scale)
        ax.set_ylabel("Pnns Group\n(number of items with FR and UK nutritional scores)", fontsize=15*fontsize_scale)    



        # ....... Modify ticks, axes and grid, .....................

        # Format ticks,
        ax.tick_params(axis='x', colors='black', direction='out', length=2, width=1) # tick only
        ax.yaxis.set_ticks_position('left')# shows only that
        ax.xaxis.set_ticks_position('bottom')# shows only that

        # Remove ticks, and axes that you dot'n want, format the other ones,
        ax.spines['top'].set_visible(False) # remove ...
        ax.spines['right'].set_visible(False) # remove ...  
        ax.spines['bottom'].set_linewidth(1) # x axis width
        ax.spines['left'].set_linewidth(1) # y axis width 

        # grid and labels
        ax.xaxis.grid(True,color='grey', linestyle='--', linewidth=1)


        
        # ....... horizonal gridlines separating paris of boxplot belongint to the same product groups, ..................... 
        
        for i,j in enumerate(grid_hlines_positions):
            ax.axhline(y=j, color="grey", linestyle="--", linewidth=0.5)


        # ....... pataches, corresponding to A,B,C,D,E nutri grades, .....................

        point_foods = {    "darkgreen":(-15,-1, 14),
                               "lightgreen":(-1,2, 3),
                               "yellow":(2,10, 8),
                               "orange":(10,18, 8),
                               "red":(18,40, 22)
                          }
        for k in list(point_foods.keys()):

            r_from  = point_foods[k][0]
            r_to    = point_foods[k][1]
            r_width = point_foods[k][2]

            rect = plt.Rectangle((r_from, 0), r_width, 1000, color=k, alpha=0.2, edgecolor=None)
            ax.add_patch(rect)

        # box with legend on nutri-scores 
        import matplotlib.patches as mpatches

        # .. data for legend
        dct_data_for_legend = dict()
        for group_color, group_name in zip(list(point_foods.keys()), ["A", "B", "C", "D", "E"]):
            dct_data_for_legend[group_name] = group_color

        # .. Add legend to the figure
        patch_list_for_legend =[]
        for i, group_name in enumerate(list(dct_data_for_legend.keys())):
            group_patch = mpatches.Patch(color=dct_data_for_legend[group_name], 
                                         label=f"Nutri Group {group_name}",alpha=0.6)
            patch_list_for_legend.append(group_patch)

        # ....
        l = fig.legend(
            handles=patch_list_for_legend, 
            loc="center", 
            frameon=False, 
            scatterpoints=1,
            ncol=5, 
            bbox_to_anchor=(0.5, 0.85), 
            fontsize=12*fontsize_scale
            )
        l.set_title("5 nutri groups shown as different colours on boxplot bacground")
        l.get_title().set_fontsize('12')

        # ....... layout, .....................

        plt.tight_layout()
        fig.subplots_adjust(top=adjust_top)# to adjust for main title, amd legend
        plt.show();  

        
    if make_boxplot==False:
        counted_items = pd.DataFrame({"pnns_group":sorted_box_names_old, "item_nr":counted_items})
        return counted_items, box_colors


      
      
      
      
# Function, ...........................................................................
def create_boxplot_with_nutriscores(cleaned_data):
    '''
        Helper function that creates boxplots wiht Nutriscores 
        in prodcust with wifferent nutriscores in UK and France, 
        Bacground of the image are showing traffic light-nutri 
        scores as on the boxes, 
    '''

    # plot data with different nutri scores,

    # prepare the data
    colNames_nutri_scores = list(find_and_display_patter_in_series(series=pd.Series(cleaned_data.columns), pattern="nutrition-score"))
    df_subset = cleaned_data.loc[:,colNames_nutri_scores]
    df_subset = df_subset.dropna(how="all",axis=0)
    df_subset["row_filter"] = (df_subset.iloc[:,0]!=df_subset.iloc[:,1])

    #..
    rows_to_use = df_subset["row_filter"]
    rows_to_use.reset_index(drop=True, inplace=True)
    rows_with_different_nutri_scores_in_uk_and_fr = pd.Series(df_subset.index)[rows_to_use]

    #..
    df_final_subset = cleaned_data.loc[rows_with_different_nutri_scores_in_uk_and_fr,:]


    input_data = dict(
        pat_colname_with_all_data = "nutrition-score",
        pat_colname_with_groups = "pnns_groups_2",
        pat_colname_with_data_for_top_box="fr_100g",
        pat_colname_with_data_for_bottom_box="uk_100g",
        coutry_box_colors = ["cornflowerblue", "red"],        
        )

    # boxplot
    paired_boxplots(
        df = df_final_subset,    
        plot_title = "nutritional scores of food products\nwith different values in French and UK scale\n(Blue boxes - French Scale, Red boxes - UK Scale)", 
        figure_size = (12,8),
        **input_data
        )

    # use the same function to get numerical data,
    data_all, colors_all = paired_boxplots(df = cleaned_data,  make_boxplot=False, **input_data)
    data_diff, colors_diff = paired_boxplots(df = df_final_subset,  make_boxplot=False, **input_data)

    # merge the data
    counted_data = pd.merge(data_all, data_diff, on="pnns_group", suffixes=["_all_data", "_diff_scores"])
    counted_data["% of items with different scores"] = np.round(counted_data.iloc[:,2]/counted_data.iloc[:,1]*100,2)

    # display the table, no matter what :)
    from IPython.display import display
    display(counted_data)
    
    
    
    
    
    

    
# Function, .............................................................................
def clean_and_tokenize_ingredient_list(*, ex, verbose=False):
    """
        ---------    ------------------------------------------------------------------
        Function   : Takes ingredient list, divide each substring into tokens, 
                     lowercase all, so there is no problem, removes several stop signs, NaN, 
                     "none", or substring like 5% or a, f, g made up form one letter.
                     The function also removes, or replaces, to multiple empty spaces, tabs etc...
        ---------    ------------------------------------------------------------------
        ...
        ex         : pd.Series, with food ingredient list,
        Verbose.   : bool,
        ... 
        returns.   : pd.DataSeries() or None, 
                     with the following columns: 
                     - "ingredinet_name", 
                     - "ingredient_count", 
                     - "percentage_of_products_with_the_given_imngredient", 
                     - "total_product_number"
    """
    
    #### main info:
    number_of_items_to_tokenize = int(ex.shape[0])
    number_of_empty_items = int(ex.isnull().sum())
    
    
    #### test, 

    # .. check if there are items that can be used,
    if number_of_items_to_tokenize==0:
        if verbose==True:
            print("ERROR, number_of_items_to_tokenize == 0!")
        else:
            pass
        return None
    
    # .. if thre are only missing data at each positon in series, 
    elif number_of_items_to_tokenize==number_of_empty_items:
        if verbose==True:
            print("ERROR, threre are only missing data in provided Series == 0!")
        else:
            pass
        return None     
    
    # .. else do the following,
    else:
        if verbose==True:
            print(f"Cleaning ingredient list with {number_of_items_to_tokenize} items, including {number_of_empty_items} NaN, ", end="")
        else:
            pass
        
        #### modify and clean substrings, ..................................
                   
        # .. turn all strings into lowercase,
        ex = ex.str.lower()

        # .. group desriptions, because they offten contain duplicates names,
        ex = ex.str.replace("\s\(.*\)", "")
        ex = ex.str.replace("\s\[.*\]", "")

        # .. items like and/or and org -> organic,
        ex = ex.str.replace(r"\sand\/or\s", ", ")
        ex = ex.str.replace(r"\sand\s", ", ")
        ex = ex.str.replace("org\s", "organic ")
        
        # .. info, small dot to not be bored,
        if verbose==True: 
            print(f" .", end="")
        else:
            pass
        
        #### Tokenize substrings, .......................................... 
        
        # .. divide each substrings, and create a pd:series with one token , at each position,
        ex = pd.Series(ex.str.split("[\,|•|;|:|\.]", expand=True).values.flatten())
         
        # info, small dot to not be bored,
        if verbose==True: 
            print(f" .", end="")
        else:
            pass
        
        # .. remove empty spaces, dots, comas etc.. created after string splitting,
        ex = ex.str.replace("\(", "")
        ex = ex.str.replace("\)", "")
        ex = ex.str.replace("\*", "")
        ex = ex.str.replace("\.$", "")
        ex = ex.str.replace("\s\s", " ")
        ex = ex.str.replace("\s$", "")
        ex = ex.str.replace("^\s+","")
        ex = ex.str.replace("_", "")
        ex = ex.str.replace("^\d+\s?%", "") # remove tokens like 5%, 9%, with no other decription
        
        # info, small dot to not be bored, 
        if verbose==True: print(f" .", end="")        
        
        
        #### Remove missing data, .......................................... 
                
        # .. remove empty substrings or substrings with just one letter,
        def find_nan(x):
            if len(x)<2: return np.nan
            else: return x

        # .. remove missing data, encoded as 'None', or found with find_nan function,
        ex = pd.Series(ex[ex!='None'])
        ex = ex.dropna()
        ex = ex.apply(find_nan)
        ex = ex.dropna() # I had to remove that twice, and separately, or else it wa san error,       
        
        # .. info
        if verbose==True: 
            print(f" , token number {ex.shape[0]},",end="")
        else:
            pass
            
        #### Count items, ...................................................          
        
           
        # .. count total number of appearances or each ingredient, and perc of items with each ingredient, 
        ex_counted = ex.value_counts()
        
        # .. ..
        ingredient_name = pd.Series(ex_counted.index, index=ex_counted.index)
        
        # .. ..
        ex_perc = ex_counted.copy()
        ex_perc =  (ex_counted/(number_of_items_to_tokenize-number_of_empty_items))*100
        
        # .. ..
        ex_products_number = pd.Series([(number_of_items_to_tokenize-number_of_empty_items)]*ex_counted.shape[0], index=ex_counted.index)
        
        
        # .. build final table & rename columns,
        ex = pd.concat([ingredient_name, ex_counted, ex_perc, ex_products_number], axis=1)
        ex.columns=["ingredinet_name", "ingredient_count", "percentage_of_products_with_the_given_ingredient", "total_product_number"]
        ex.reset_index(drop=True, inplace=True)

        # .. info
        if verbose==True: 
            print(f" with {ex.shape[0]} unique tokens",end="\n")
        else:
            pass
        
        #### return,
        return ex


    
    
    
# Function, .............................................................................
def plot_most_common_ingredients_in_a_given_country(*, 
    cleaned_data_subset, 
    country_name="", 
    ingredients_to_display_on_each_plot=5, 
    figsize=(20,6),
    adjust_top=0.6,
    cmap="tab10",
    cmap_from=0,
    cmap_to=0.5,
    verbose=False
):
    
    """
        ----------------------   ------------------------------------------------------------------
        Function                 : This function finds n most common ingredients in each country
                                   and crete barplots with their relative content in 5 different 
                                   nutritional grade groups
        ----------------------   ------------------------------------------------------------------
        ...
        cleaned_data_subset      : pd.DataFrame with the following columns 
                                   {"ingredients_text","nutrition_grade_fr","countries_en"}
                                   with, or without NaN
        country_name             : must be present in  "countries_en"
        ingredients_to_display   : numer of most common ingredient in an entire dataset 
        _on_each_plot              that will be displayed in product with each nutritional grade
        Verbose.                 : bool,
        ... 
        returns.                 : plot, with 5 subplots, for products with each nutritional grade
                                   legend contains data in all nutritional grades, 
    """
    
    ## Prepare the data,, ------------------------

    # .. constants,
    nutrition_grades = ["a", "b", "c", "d", "e"]

    # .. Find Five Most Common Ingredients In A Given Country,
    whole_data_for_plot = cleaned_data_subset.loc[cleaned_data_subset.countries_en==country_name,"ingredients_text"]
    five_most_common_ingredients_in_a_country = clean_and_tokenize_ingredient_list(ex=whole_data_for_plot, verbose=verbose)
    five_most_common_ingredients_in_a_country = five_most_common_ingredients_in_a_country.iloc[0:ingredients_to_display_on_each_plot,:]
    five_most_common_ingredients_in_a_country.reset_index(drop=True)

    

    ## select color for each ingredient,, ------------------------
    
    # .. select colors,
    bar_colors = plt.get_cmap(cmap)(np.linspace(cmap_from, cmap_to, five_most_common_ingredients_in_a_country.shape[0]))
    bar_colors = pd.DataFrame(bar_colors, columns=["color_1", "color_2", "color_3", "color_4"])


    ## select data for each subplot,, ------------------------

    #.. info
    if verbose==True: 
        print(f"Selecting data for each subplot ...", end="")
    else:
        pass
        
    # .. find percentage of the most common ingredients in products form each nutritional group grade,
    dct_data_for_plot = dict()
    for i,j in enumerate(nutrition_grades):

        if verbose==True: 
            print(i, end=", ")
        else:
            pass

        # select the data and tokenize it,
        data_for_plot = cleaned_data_subset.loc[(cleaned_data_subset.countries_en==country_name) & (cleaned_data_subset.nutrition_grade_fr==nutrition_grades[i]),"ingredients_text"]
        ith_subset_tokens = clean_and_tokenize_ingredient_list(ex=data_for_plot, verbose=False)

        # find most common ingredients in each group,

        # .. build data container for final results, 
        df_res_raw =  five_most_common_ingredients_in_a_country.copy() # just to have the same format,
        df_res_raw.iloc[:,1:4]=0 # You shoudl see ingredient list and only zeros 
        df_res_new =  df_res_raw.iloc[0:0,:].copy()

        # .. add the data, fro ith group, 
        #.   {I had to use for loop , because sometime the product was not used in a given group}
        for r_nr, ingr in enumerate(list(five_most_common_ingredients_in_a_country.iloc[:,0])):
            if sum(ith_subset_tokens.ingredinet_name==ingr)==1:
                one_row = ith_subset_tokens.loc[ith_subset_tokens.ingredinet_name==ingr,:]
            else:
                one_row = df_res_raw.loc[df_res_raw.ingredinet_name==ingr,:]
            df_res_new = pd.concat([df_res_new,one_row], axis=0)       
            df_res_new = df_res_new.reset_index(drop=True)

        # add colors, so they will be sorted together,   
        df_res_new = pd.concat([df_res_new, bar_colors], axis=1, sort=False)  

        # sort, 
        df_res_new = df_res_new.sort_values(by="percentage_of_products_with_the_given_ingredient",ascending=False)

        # add to dct,
        dct_data_for_plot[j] = df_res_new

    if verbose==True: 
        print(", done, ")
    else:
        pass


    ## make plots, ------------------------


    # .. Fig, 
    #plt.rcParams['legend.title_fontsize'] = 25
    fig, axs = plt.subplots(ncols=len(dct_data_for_plot), nrows=1, figsize=figsize, facecolor="white")
    fig.suptitle(f"{country_name}", fontsize=25, x=0.1)

    # .. subplots,
    for key, ax in zip(list(dct_data_for_plot.keys()), axs.flat):

        # _ data for plot,
        data_for_plot       = dct_data_for_plot[key].percentage_of_products_with_the_given_ingredient.values
        bar_colors_for_plot = dct_data_for_plot[key].iloc[:,4::].values
        bar_names           = list(dct_data_for_plot[key].iloc[:,0].values)
        bar_positions       = range(data_for_plot.shape[0])[::-1]

        # _ horizontal barplot,
        ax.barh(bar_positions, data_for_plot, color=bar_colors_for_plot, label=bar_names)

        # _ title,
        ax.set_title(f"""Nutritional Group {key}
        Data From {int(dct_data_for_plot[key].iloc[0,3])} Products""")    

        # _ axes description,
        ax.set(ylabel="ingredient", xlabel="% of Products with \n each ingredient")
        #ax.legend().set_visible(False) # because. i wish to have only one legend fopr all plots

        # _ Add ticks on y axis, and names for each bar,
        ax.set_yticks(bar_positions)
        ax.set_yticklabels(bar_names, fontsize=10, color="black")
        ax.set_xticks([0, 20,40,60,80, 100])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100"], fontsize=10, color="black")

        # Format ticks,
        ax.tick_params(axis='x', colors='black', direction='out', length=4, width=2) # tick only
        ax.tick_params(axis='y', colors='black', direction='out', length=4, width=2) # tick only    
        ax.yaxis.set_ticks_position('left')# shows only that
        ax.xaxis.set_ticks_position('bottom')# shows only that

        # Remove ticks, and axes that you dot'n want, format the other ones,
        ax.spines['top'].set_visible(False) # remove ...
        ax.spines['right'].set_visible(False) # remove ...  
        ax.spines['bottom'].set_linewidth(2) # x axis width
        ax.spines['bottom'].set_bounds(0,100) # Now the x axis do not go under the legend
        ax.spines['left'].set_linewidth(2) # y axis width 

        # _ grid, 
        ax.set(ylim=(0-0.5,ingredients_to_display_on_each_plot-0.5), xlim=(0,100))
        ax.xaxis.grid(color='grey', linestyle='--', linewidth=1) # horizontal lines



    ## Add legend, with bar color description, and data from the entire dataset, , ------------------------


    # .. collect the data from global data
    data_for_legend = pd.concat([five_most_common_ingredients_in_a_country,bar_colors], axis=1)


    # .. create patch list for legend, with the same ingredient colors as on each subplot,
    import matplotlib.patches as mpatches
    patch_list_for_legend =[]
    for i in range(data_for_legend.shape[0]):

        patch_label = f"{i+1}. {data_for_legend.iloc[i,0]} - Found in {data_for_legend.iloc[i,1]} Products ({data_for_legend.iloc[i,2].round(0)}%)"

        ingr_patch = mpatches.Patch(color=list(data_for_legend.iloc[i,4::].values), label=patch_label)
        patch_list_for_legend.append(ingr_patch)


    # .. Add legend to the figure
    l = fig.legend(handles=patch_list_for_legend, loc="center", frameon=True, scatterpoints=1, ncol=2, bbox_to_anchor=(0.5, 0.85), title="", fontsize=12)
    l.set_title(f"{len(patch_list_for_legend)} most common ingredients found in {int(data_for_legend.iloc[0,3])}")
    l.get_title().set_fontsize('15')

    # .. Avoid text overlapping between plots,
    plt.tight_layout()
    fig.subplots_adjust(top=adjust_top, hspace=0.5, wspace=0.6, right=1)# to adjust for main title, amd legend
    plt.show();
    
    
    
    
# Function, .............................................................................    
def plot_main_ingredients_by_country_by_nutrigroup(
    cleaned_data, 
    countries = ["United States", "France", "Germany"],
    plot_dct=dict() # for plot_most_common_ingredients_in_a_given_country
):
    
    '''
        helper function to create nice looking barplots with the most common ingredients found in products 
        * figures  = countires
        * subplots = nutrigroups
        * bars     = % of products wiht a given product
        * bar order = highest value at the top, 
        * bar colors = product name, the same on all subplots, 
    '''

    # Extract the data for plots, .............................................
    cleaned_data_subset = cleaned_data.loc[:,["ingredients_text","nutrition_grade_fr","countries_en"]]               
    cleaned_data_subset = cleaned_data_subset.dropna(how="any", axis=0)       

    # Make plots, .............................................................
    #             (Functions in Task C Helpers)
    for i, one_country in enumerate(countries):
        print(one_country,"..............")
        plot_most_common_ingredients_in_a_given_country(
            cleaned_data_subset=cleaned_data_subset, 
            country_name=one_country,
            ingredients_to_display_on_each_plot=5, # number of products to be displayed, 
            verbose=True,
            **plot_dct
        )

        