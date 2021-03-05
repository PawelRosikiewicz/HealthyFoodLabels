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
import scipy.stats as stats



# Function, ........................................................................................
def find_columns_with_selected_dtype(*, df, dtype="numerical"):
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          Finds column names of columns with requested dtype numeric or text,
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        
        . Input .
        * df                DataFrame, with unique column names
        * dtype             data type that we with to find {'text', 'numeric'}
        
        Returns             
        _________________   _______________________________________________________________________________
        
        * col_names         list wiht column names, that have eaither numeric or text dtape
    """     
    col_names = []
    
    if dtype=="numerical":
        from pandas.api.types import is_numeric_dtype # if is_string_dtype(df[j]):
        for col_name in list(df.columns):
            if is_numeric_dtype(df[col_name]):
                col_names.append(col_name)
                
    if dtype=="text":
        from pandas.api.types import is_string_dtype # if is_string_dtype(df[j]):
        for col_name in list(df.columns):
            if is_string_dtype(df[col_name]):
                col_names.append(col_name)    
    
    return col_names





# Function, ..............................................................................................

def clean_data_and_find_correlations(*, df, row_Filter=None, colName_main_variable, verbose=False):

    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          This function, takes df, with numeric data in each column, 
                            computes all column to one of them, called colName_main_variable
                            the data are formatted and cleaned in each pari of column individually, (by removing na)
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        
        . Input .
        * df                DataFrame, with unique column names
        * row_filter        list[bool], with len(row_filter )==df.shape[0]
        * verbose           bool, if True, shows info
        
        Returns             
        _________________   _______________________________________________________________________________
        
        * comparisons_dct   dictionary, where key=column name in df, 
                            inside each entry, there is another dict wiht condition names, raw and filtered data
                            and results of correlation made with 3 different methods (pearson, spearman and kendal) 

    """     

    #### filter the data:
    if row_Filter==None:
        row_Filter = [True]*df.shape[0]
    # ........
    filtered_df_main_col    = pd.Series(df.loc[row_Filter,colName_main_variable]) # pd.Series
    filtered_df             = df.loc[row_Filter, :]                               # pd Series or pd.DataFrame        
    
    #### rename column with main group to avoid having duplicates, 
    filtered_df.rename(columns={str(colName_main_variable):"".join([str(colName_main_variable),"_"])}, inplace=True) # to ensute, that we have no columns with the same name,
  
    #### Loop over each columns and compare it with main group,
    comparisons_dct = dict() 
    # .......
    for i, colName in enumerate(list(filtered_df.columns)):
        
        if verbose==True:
            print(i, colName_main_variable, " - with - ", colName)

            
            
        # some columns had to many repeats r nan. i remove them later, but warningn  were annoying, thus I removed them;
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning) 
                    
        # --------------------------
        # Prepare dct for the data,       
        # --------------------------   

        # new dict,
        one_comparison_dct = dict()
        one_comparison_dct["X_main_group_sliced_with"]    = row_Filter
        one_comparison_dct["X_main_group"]                = colName_main_variable
        one_comparison_dct["Y_compared_with"]             = colName

        # --------------------------
        # PREPARE THE DATA       
        # --------------------------    

        # ................... raw data, .........................

        # prepare the data,
        filtered_df_one_col         = filtered_df.loc[:,colName]
        data_for_comparison_df      = pd.concat([filtered_df_main_col, filtered_df_one_col], axis=1) 
        data_for_comparison_df_full = data_for_comparison_df.copy() # for hitograms

        # remove missing data,
        data_for_comparison_df      = data_for_comparison_df.dropna(how="any",axis=0) # to create X/Y_cleaned

        # names, and basic data to display,
        sample_number               = int(data_for_comparison_df.shape[0])  # int
        atribute_names              = list(data_for_comparison_df.columns) # list



        # ................... control, ............................

        # check if you can continue,
        if sample_number <= 2:
            # add to dict,
            one_comparison_dct["X_total"]                = [None]
            one_comparison_dct["X_cleaned"]              = [None]
            one_comparison_dct["X_cleaned_log"]          = [None]
            # .......
            one_comparison_dct["Y_total"]                = [None]
            one_comparison_dct["Y_cleaned"]              = [None]
            one_comparison_dct["Y_cleaned_log"]          = [None] 
            # .......
            one_comparison_dct["pearson_results"]        = [None]
            one_comparison_dct["sperman_results"]        = [None]
            one_comparison_dct["kendalltau_results"]     = [None]
            # ..........
            one_comparison_dct["linregress_results"]     = [None]
            one_comparison_dct["linregress_results_log"] = [None]

            

            if verbose==True:
                print(f"Caution, (column cobination nr {i}) -ie.- {atribute_names[0]},vs{atribute_names[1]}, has less then 3 items to compare !")
            ############################################################################
            comparisons_dct[colName] = one_comparison_dct
            ############################################################################
        
        # else,
        if sample_number > 2:

            # ... X,Y data for plots and correlation, ................

            # all data, without removing NaN in each row, - for hist,
            X_total = data_for_comparison_df_full.iloc[:, 0]
            X_total = X_total.dropna(how="any").values.flatten()
            Y_total = data_for_comparison_df_full.iloc[:, 1]
            Y_total = Y_total.dropna(how="any").values.flatten() 

            # data,
            X_cleaned = data_for_comparison_df.iloc[:, 0].values.flatten()
            Y_cleaned = data_for_comparison_df.iloc[:, 1].values.flatten()

            # transform values into log(x+2), +2 to avoid having log= inf, or zero
            X_cleaned_log = np.log(X_cleaned+16)
            Y_cleaned_log = np.log(Y_cleaned+16)

            
            # .....
            
            
            # add to dict,
            one_comparison_dct["X_total"]       = X_total
            one_comparison_dct["X_cleaned"]     = X_cleaned
            one_comparison_dct["X_cleaned_log"] = X_cleaned_log
            # .......
            one_comparison_dct["Y_total"]       = Y_total
            one_comparison_dct["Y_cleaned"]     = Y_cleaned
            one_comparison_dct["Y_cleaned_log"] = Y_cleaned_log


            # --------------------------
            # FIND CORR.     
            # --------------------------        


            # ... Correlation, ....................................


            # correlations,
            pearson_results    = stats.pearsonr(X_cleaned_log, Y_cleaned_log)   # linear
            sperman_results    = stats.spearmanr(X_cleaned, Y_cleaned)          # rank, with rho value, 
            kendalltau_results = stats.kendalltau(X_cleaned, Y_cleaned)   # rank,based on orientation of pairs of ranks
            # ............
            one_comparison_dct["pearson_results"]      = pearson_results
            one_comparison_dct["sperman_results"]      = sperman_results
            one_comparison_dct["kendalltau_results"]   = kendalltau_results

            # Compute a least-squares regression for two sets of measurements.
            LR_slope, LR_intercept, LR_r_value, LR_p_value, LR_std_err = stats.linregress(X_cleaned, X_cleaned)
            # ..........
            linregress_results = {"slope": LR_slope, 
                                      "intercept": LR_intercept, 
                                      "r_value": LR_r_value, 
                                      "p_value": LR_p_value, 
                                      "std_err": LR_std_err}
            # ..........
            one_comparison_dct["linregress_results"] = linregress_results


            # Compute a least-squares regression for two sets of measurements.
            LR_slope, LR_intercept, LR_r_value, LR_p_value, LR_std_err = stats.linregress(X_cleaned_log, Y_cleaned_log)
            # ..........
            linregress_results_log = {"slope": LR_slope, 
                                      "intercept": LR_intercept, 
                                      "r_value": LR_r_value, 
                                      "p_value": LR_p_value, 
                                      "std_err": LR_std_err}
            # ..........
            one_comparison_dct["linregress_results_log"] = linregress_results_log

            ############################################################################
            comparisons_dct[colName] = one_comparison_dct
            ############################################################################
        
    return comparisons_dct





# Function, ..............................................................................................

def make_plots_and_table_with_corr(*, data_dct, N_plots_to_show=5, display_table=True, display_plots=True, 
                                   N_plots_from=0, N_plots_to=2, figure_size= (12, 4), N_rows_in_table_to_show=10, 
                                   return_rable=False, select_group_name=None, verbose=False):

    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          Larger function that makes four things: 
                            - takes data_dct ie. dictionary generated with clean_data_and_find_correlations()
                              it collects and order all the data, in df
                            - if display_plots=True, display requested number of plots wiht scatter plot 
                              and LR line representing relationship etwen compared values, and two histogram plots,
                              showsing distribusion of values in a given variable before and sfter selection (na removal)
                              required for calculating correlation between these values
                            - if return_rable=False, it will return a df, with all values such as rvalua and tau value
                              that were collected form the dictionary,
                            - if display_table=True, it will display, top number of requested correlations with the hisest absolute values
                              (either negative or positive)
                             
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        
        . Input .
        * data_dct          distionary, created with clean_data_and_find_correlations()
        * display_plots     bool, =True, see above
        * N_plots_from      int, =0, from whereto start displaying plots, 1st pair (0), is a correlation between main value itself
        * N_plots_to        int, =3, how many plots fdo display, not included,
        * figure_size       tuple(int, int)
        * return_rable      bool, =False, see above
        * select_group_name str, depreciated....
        * N_rows_in_table_to_show=10,  : if display_table=True, how many rows to display,
        * verbose           bool, =False       
        
        Returns             
        _________________   _______________________________________________________________________________
        
        * ≥1 plots          each with 3 subplots, see above
        * df image          df.head(), see above
        * df                unordered, df with data used for plots, in all conditiona, even if not displayed.
    """     

    
    # collect and order the data from dict, 
    dct_keys          = list(data_dct.keys())
    dct_rvalues       = list() 
    dct_pvalues       = list()
    dct_rhovalues     = list()
    dct_tauvalues     = list()
    dct_sample_number = list()
    # .......
    for i, key in enumerate(dct_keys):
        if data_dct[key]["X_total"][0]!=None:
            dct_rvalues.append(data_dct[key]["pearson_results"][0])
            dct_pvalues.append(data_dct[key]["pearson_results"][1])
            dct_rhovalues.append(data_dct[key]["sperman_results"][0])
            dct_tauvalues.append(data_dct[key]["kendalltau_results"][0])
            dct_sample_number.append(data_dct[key]["X_cleaned"].shape[0])
            
        else:
            dct_rvalues.append(np.nan)
            dct_pvalues.append(np.nan)
            dct_rhovalues.append(np.nan)
            dct_sample_number.append(np.nan)  
            dct_tauvalues.append(np.nan)  
                
        
    # ....... join them into df, 
    order_in_df = pd.concat([pd.Series(dct_keys), 
                             pd.Series(dct_rvalues),
                             pd.Series(dct_pvalues),
                             pd.Series(dct_rhovalues),
                             pd.Series(dct_tauvalues),
                             pd.Series(dct_sample_number)
                            ],axis=1) # to use functions below

    # ...... re-name, 
    order_in_df.columns=["keys", "rvalues", "pvalues", "rho", "tau", "sample nr"] # for my info
    
    # ...... save to return
    df_with_all_results_unordered = order_in_df.copy()
    
    # ...... re-order, to find the 
    order_in_df["rvalues_abs"] = pd.Series(np.abs(order_in_df.iloc[:,1].values)) # use absolute values
    order_in_df.sort_values(by="rvalues_abs", ascending=False, inplace=True) # to get tghe order
    
    
    # --------------------------
    # DATA FOR PLOTS       
    # --------------------------    
    if display_plots==True:
   
        # show plots, of N top values 
        for i,key in enumerate(list(order_in_df.loc[:,"keys"])):

            if i >= N_plots_from and i<=N_plots_to:

                # --------------------------
                # DATA FOR PLOTS       
                # --------------------------            

                X_cleaned_log = data_dct[key]["X_cleaned_log"]
                Y_cleaned_log = data_dct[key]["Y_cleaned_log"]
                # ......
                X_cleaned = data_dct[key]["X_cleaned"]
                Y_cleaned = data_dct[key]["Y_cleaned"]
                # ......
                X_total = data_dct[key]["X_total"]
                Y_total = data_dct[key]["Y_total"]

                # ......
                atribute_names = [data_dct[key]["X_main_group"], data_dct[key]["Y_compared_with"]]

                # .....
                LR_intercept = data_dct[key]["linregress_results_log"]["intercept"]
                LR_slope = data_dct[key]["linregress_results_log"]["slope"]
                LR_r_value = data_dct[key]["linregress_results_log"]["r_value"]
                LR_p_value = data_dct[key]["linregress_results_log"]["p_value"]

                # .....
                pearson_results = data_dct[key]["pearson_results"]

                # INFO:
                if verbose==True:
                    print(i, key, atribute_names)
                
                # --------------------------
                # PLOTS       
                # --------------------------

                # ... fig, ax, .............................................................................

                fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figure_size, facecolor="white")
                
                if select_group_name==None:
                    plt.suptitle(f"[{i}] {atribute_names[0]} - vs - {atribute_names[1]}", fontsize=17)
                else:
                    plt.suptitle(f"[{i}] {atribute_names[0]} - vs - {atribute_names[1]}, ({select_group_name})", fontsize=17)
                    
                # colors
                selected_data_color = "deepskyblue" 
                total_data_color    = "black"
                scatter_color       = "forestgreen"
                line_color          = "black"
                
                
                # ... SUBPLOT 1, scatterplot ..............................................................

                # data,

                X = X_cleaned_log
                Y = Y_cleaned_log
                
                
                # title,
                subplot_title = f'Correlation and LR:\nr = {LR_r_value:.2f}, p = {np.round(list(pearson_results)[1],3)}\ny = {LR_intercept:.2f}+{LR_slope:.2f}x'

                # plot and add line
                axs[0].plot(X, Y, linewidth=0, marker='.', color=scatter_color )
                axs[0].plot(X, LR_intercept + LR_slope * X, color=line_color, linestyle="--", linewidth=3)
                axs[0].set_title(subplot_title)

                # axes labels
                axs[0].set_xlabel(str(atribute_names[0])+"\nlog values")
                axs[0].set_ylabel(str(atribute_names[1])+"\nlog values")
                axs[0].set_xlim(X.min(), X.max())
                axs[0].set_ylim(Y.min(), Y.max())

                # Remove ticks, and axes that you dot'n want, format the other ones,
                axs[0].yaxis.set_ticks_position('left')# shows only that
                axs[0].xaxis.set_ticks_position('bottom')# shows only that
                axs[0].spines['top'].set_visible(False) # remove ...
                axs[0].spines['right'].set_visible(False) # remove ...  
                axs[0].spines['bottom'].set_linewidth(1) # x axis width
                axs[0].spines['left'].set_linewidth(1) # y axis width 

                # grid and labels,
                axs[0].xaxis.grid(True,color='grey', linestyle='--', linewidth=0.5)
                axs[0].yaxis.grid(True,color='grey', linestyle='--', linewidth=0.5)


                #................................................................................
                # ... kwars for histograms ......................................................
                hist_kwargs         = dict(histtype="stepfilled", density=True, bins=20)    
                # ... kwars for histograms ......................................................
                #................................................................................    


                # ... SUBPLOT 2, X data histogram ....................................

                # title,
                perc_value = np.round((X_cleaned.shape[0]/X_total.shape[0])*100,2)
                subplot_title = f"{X_total.shape[0]}, non-NaN items in df\n{X_cleaned.shape[0]} items used for Corr ({perc_value}%)"

                # plot and add line    
                axs[1].hist(X_total, color=total_data_color,edgecolor=total_data_color, alpha=0.6, **hist_kwargs, label="all data")
                axs[1].hist(X_cleaned, color=selected_data_color,edgecolor=selected_data_color, alpha=0.5, **hist_kwargs, label="data used for LR")
                axs[1].set_title(subplot_title)
                
                # axes labels
                axs[1].set_xlabel(str(atribute_names[0])) 
                axs[1].set_ylabel("Density")
                
                # recompute x/y limits
                axs[1].relim() # recompute the ax.dataLim
                axs[1].autoscale_view() # update ax.viewLim using the new dataLim

                # Remove ticks, and axes that you dot'n want, format the other ones,
                axs[1].yaxis.set_ticks_position('left')# shows only that
                axs[1].xaxis.set_ticks_position('bottom')# shows only that
                axs[1].spines['top'].set_visible(False) # remove ...
                axs[1].spines['right'].set_visible(False) # remove ...  
                axs[1].spines['bottom'].set_linewidth(1) # x axis width
                axs[1].spines['left'].set_linewidth(1) # y axis width    


                # ... SUBPLOT 3, Y data histogram ....................................

                # title,
                perc_value = np.round((Y_cleaned.shape[0]/Y_total.shape[0])*100,2)
                subplot_title = f"{Y_total.shape[0]}, non-NaN items in df\n{Y_cleaned.shape[0]} items used for Corr ({perc_value}%)"

                # plot and add line    
                axs[2].hist(Y_total, color=total_data_color,edgecolor=total_data_color, alpha=0.6, **hist_kwargs, label= "all data")
                axs[2].hist(Y_cleaned, color=selected_data_color,edgecolor=selected_data_color, alpha=0.5, **hist_kwargs, label="data used for LR")
                axs[2].set_title(subplot_title)
                
                # axes labels
                axs[2].set_xlabel(str(atribute_names[1]))
                axs[2].set_ylabel("Density")

                # recompute x/y limits
                axs[2].relim() # recompute the ax.dataLim
                axs[2].autoscale_view() # update ax.viewLim using the new dataLim            

                # Remove ticks, and axes that you dot'n want, format the other ones,
                axs[2].yaxis.set_ticks_position('left')# shows only that
                axs[2].xaxis.set_ticks_position('bottom')# shows only that
                axs[2].spines['top'].set_visible(False) # remove ...
                axs[2].spines['right'].set_visible(False) # remove ...  
                axs[2].spines['bottom'].set_linewidth(1) # x axis width
                axs[2].spines['left'].set_linewidth(1) # y axis width        

                
                
                # .. create patch list for legend, ..................
                
                # . import
                import matplotlib.patches as mpatches
                patch_list_for_legend =[]
                
                # .1.
                ingr_patch      = mpatches.Patch(color=total_data_color, label=f"all available data")
                patch_list_for_legend.append(ingr_patch)
                
                # .2.
                ingr_patch      = mpatches.Patch(color=selected_data_color, label=f"data used for Corr(X,Y), and LR")
                patch_list_for_legend.append(ingr_patch)                
                   
                # . Add legend to the figure
                l = fig.legend(handles=patch_list_for_legend, loc="center", frameon=False, scatterpoints=1, ncol=2, bbox_to_anchor=(0.5, 0.82), title="", fontsize=12)
                l.set_title("histograms with:")
                l.get_title().set_fontsize('12')        
                

                # ... FIG aestetics ....................................
                sns.despine()
                fig.tight_layout()
                fig.subplots_adjust(top=0.6)# to adjust for main title, amd legend
                plt.show();
    
    
    # .... DISPLAY TABLE WITH MANY RESULTS BELOW PLOTS, .................
    if display_table == True:
                
        # remove unwanted 
        df_with_corr_results_ordered = order_in_df.drop(["rvalues_abs"], axis=1)
        df_with_corr_results_ordered.reset_index(drop=True, inplace=True)
                
        from IPython.display import display
        print(f"""{"".join(["-"]*80)}\nTable with all corelations, \n{"".join(["-"]*80)}""")
        display(df_with_corr_results_ordered.head(N_rows_in_table_to_show))        

    # ... RETURN .................................... I PLACED IT HERE TO NOT WRITE ANY FUNCTIONS AGAIN !
    if return_rable == True:
        
        return df_with_all_results_unordered

    
    
    
    
# Function, ..............................................................................................    
def corr_heatmap(*, corr_res_dct, method="rvalues"):
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          plots a histogram with correlation values generated for multiple comparisons
                            clean_data_and_find_correlations(), and then with make_plots_and_table_with_corr()
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        
        . Input .
        * corr_res_dct      pd dataFrame, rows = condition name, 
                            make_plots_and_table_with_corr() from dict, generated with clean_data_and_find_correlations()
        * method            str, {'rvalues', 'rho', 'tau'}
        
        Returns             
        _________________   _______________________________________________________________________________
        
        * heatmap

    """     

    #### collect and transpose values for heatmap, .................................................
    
    # . loop
    for i, key in enumerate(list(corr_res_dct.keys())):


        # extract,
        corr_table_one_col         = corr_res_dct[key].loc[:,method].copy()
        corr_table_one_col_for_na  = corr_res_dct[key].loc[:,method]

        # replace na with zero (to simplify histogram making),
        corr_table_one_col[corr_table_one_col_for_na.isna()] = 0

        # add to final table
        if i==0:
            corr_table      =  corr_table_one_col
        else:
            corr_table      = pd.concat([corr_table, corr_table_one_col], axis=1)

    # . add index to df, 
    corr_table["variable"]  = pd.Series(list(corr_res_dct[key]["keys"]))
    corr_table.set_index("variable", inplace=True)

    # . add column names, 
    corr_table.columns = list(corr_res_dct.keys())

    # . re-order the rows, for hist:
    row_order          = pd.Series(corr_table.sum(axis=1))     
    row_order          = list(row_order.sort_values(ascending=False).index)
    row_order
    # . .......
    corr_table         = corr_table.loc[row_order,:]

    # . transpose, to have better layout of a heatmap
    corr_table        = corr_table.transpose()
    
    
    
    #### create annotated heatmap, ..............................................................
    
    # .. figure, 
    sns.set_context(None)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10), facecolor="white")
    im = plt.imshow(corr_table.values, aspect = 1, origin="upper", interpolation='nearest', cmap="seismic")
    fig.suptitle(f"Heatmap with correlation results: {method}", fontsize=20)


    # . We want to show all ticks...
    ax.set_xticks(list(range(corr_table.shape[1])))
    ax.set_yticks(list(range(corr_table.shape[0])))
    
    # . and label them with the respective list entries
    ax.set_xticklabels(list(corr_table.columns))
    ax.set_yticklabels(list(corr_table.index))

    # . Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)

    # . Remove ticks, and axes that you dot'n want, format the other ones,
    ax.spines['top'].set_visible(False) # remove ...
    ax.spines['right'].set_visible(False) # remove ...
    ax.yaxis.set_ticks_position('none')# shows only that
    ax.xaxis.set_ticks_position('none')# shows only that
    
    # .. spine description
    ax.set_ylabel("PNNS group", fontsize=20)
    ax.set_xlabel("Variable", fontsize=20)
               
          
    # colorbar, .......................................................................................
    
    # . create an axes on the right side of ax. The width of cax will be 5%
    #   of ax and the padding between cax and ax will be fixed at 0.05 inch.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="10%", pad=0.5)
    clb = plt.colorbar(im, cax=cax, orientation="horizontal", label="Cor(X,Y), r value", extend="both")
    
    # control the stuff,
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)# to adjust for main title, amd legend
    plt.show()    
    
    