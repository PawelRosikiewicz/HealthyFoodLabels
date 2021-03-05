# ********************************************************************************** #
#                                                                                    #
#   Project: Data Frame Explorer                                                     # #                                                                                    # 
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


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os # allow changing, and navigating files and folders, 
import sys
import re # module to use regular expressions, 
import glob # lists names in folders that match Unix shell patterns
import random # functions that use and generate random numbers

import numpy as np # support for multi-dimensional arrays and matrices
import pandas as pd # library for data manipulation and analysis
import seaborn as sns # advance plots, for statistics, 
import matplotlib as mpl # to get basic plt   functions, heping with plot mnaking 
import matplotlib.pyplot as plt # for making plots, 

import matplotlib.gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from IPython.display import display
from PIL import Image, ImageDraw
from matplotlib import colors
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype
from matplotlib.font_manager import FontProperties






# Function, ............................................................................
def find_and_display_patter_in_series(*, series, pattern):
    "I used that function when i don't remeber full name of a given column"
    res = series.loc[series.str.contains(pattern)]
    return res



  
  
# Function, ............................................................................
def summarize_df(*, df, nr_of_examples_per_category=3, csv_file_name=None, save_dir=None, verbose=False):
    """
        Summary table, with basic information on column in large dataframes,
        can be applied to dafarames of all sizes, Used to create summary plots
        
        IMPORTANT: this is one of my oldest function, that I am using a lot, 
                   I will soon update it to something better, but generating the same outputs, 
                   and more resiliant to unknownw datatypes, 
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        
        . Input .
        * df                DataFrame to summarize
        * nr_of_examples_per_category
                            how many, top/most frequent records shoudl 
                            be collected in each column and used as examples of data inputs form that column
                            NaN, are ignored, unless column has only NaN
        
        . Saving .          The fgunction return Dataframe, even if file name and path to save_dir are not available
                            In that case the file are not saved.
        * csv_file_name     .csv file name that will be used to save all three dataFrames create with that function
        * save_dir          path
        
        Returns             
        _________________   _______________________________________________________________________________
        
        * data_examples.    DataFrame, summary of df, with the follwing values for each column imn df
                            . name	                 : column name in df, attribute name
                            . dtype.                 : {"nan", if ony NaN werer detected in df, "object", "numeric"}
                            . NaN_perc               : percentage of missing data (np.nan, pd.null) in df in a given attirbute
                            . summary                : shor informtaion on type and number of data we shoudl expectc:
                                                       if dtype == "numeric": return min, mean and max values
                                                       if dtype == "object" : return number of unique classes
                                                              or messsage "all nonnull values are unique"
                                                       if dtype == nan       : return "Missing data Only"                        
                            . examples                : str, with reqwuested number of most frequent value examples in a given category
                            . nr_of_unique_values	  : numeric, scount of all unique values in a category 
                            . nr_of_non_null_values   : numeric, count of all non-null values in a category
        
        * top_val_perc      DataFrame with % of the top three or most frequence records in each column 
                            in large dataframe that was summarized with summarize_data_and_give_examples()

        * top_val_names     DataFrame, same as top_val_perc, but with values, saved as string
 
    """    
    
    
    assert type(df)==pd.DataFrame, "ERROR, df must be pandas dataframe"
    
    # info
    if csv_file_name!="none" and save_dir!="none":
        if verbose == True:
            print("\n! CAUTION ! csv_file_name shoudl be provided wihtout .csv file extension!")
        else:
            pass

    # create df,
    col_names   =["All_values_are_unique", "Nr_of_unique_values", "Nr_of_non_null_values", 
                  "Examples","dtype", "nr_of_all_rows_in_original_df"]
    df_examples = pd.DataFrame(np.zeros([df.shape[1],len(col_names)]), columns=col_names, dtype="object")

    # add category names
    df_examples["name"] = df.columns

    # add NaN percentage,
    nan_counts_per_category = df.isnull().sum(axis=0)
    my_data = pd.DataFrame(np.round((nan_counts_per_category/df.shape[0])*100, 5), dtype="float64")
    df_examples["NaN_perc"] = my_data.reset_index(drop=True)

    # add nr of no NaN values
    my_data = df.shape[0]-nan_counts_per_category
    df_examples["Nr_of_non_null_values"] = my_data.reset_index(drop=True)

    # add "nr_of_all_rows_in_original_df"
    df_examples["nr_of_all_rows_in_original_df"] = df.shape[0]
    
    # these arr will be filled for future bar plot
    arr_example_percentage = np.zeros([df_examples.shape[0],nr_of_examples_per_category])
    arr_example_values     = np.zeros([df_examples.shape[0],nr_of_examples_per_category],dtype="object")
    
    
    
    # add examples and counts
    for i, j in enumerate(list(df.columns)):

        # add general data  ..............................................

        # number of unique nonnull values in each column,
        df_examples.loc[df_examples.name==j,"Nr_of_unique_values"] = df.loc[:,j].dropna().unique().size

        
        #  internal function helpers .....................................

        # categorical data, fillin_All_values_are_unique
        def fillin_All_values_are_unique(*, df_examples, df):
            if (df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"]==0).values[0]: 
                return "Missing data Only"
            elif ((df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"]>0).values[0]) and (df.loc[:,j].dropna().unique().size==df.loc[:,j].dropna().shape[0]): 
                return "all nonnull values are unique"
            elif ((df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"]>0).values[0]) and (df.loc[:,j].dropna().unique().size!=df.loc[:,j].dropna().shape[0]): 
                return f"{int(df_examples.Nr_of_unique_values[df_examples.name==j].values[0])} classes"
            else:
                pass
            
            
            
        # fill other columns ..............................................

        # this is auto-fill in case there is no data
        if df[j].isnull().sum()==df.shape[0]:
            # (df_examples.loc[df_examples.name==j,"NaN_perc"]==100).values[0]: this value was rounded up/down and was returning false positives!!!!
            df_examples.loc[df_examples.name==j,"All_values_are_unique"] = "missing data only"
            df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"] = 0 # it should be 0, but i overwrite it just in case
            df_examples.loc[df_examples.name==j,"Nr_of_unique_values"] = 0 # it should be 0, but i overwrite it just in case
            df_examples.loc[df_examples.name==j,"Examples"] = "missing data only"
            df_examples.loc[df_examples.name==j,"dtype"] = "missing data only" # because I dont want to use that in any further reading

        # in other cases, we can create data examples, from nonnull values, depending on their type,
        else:

            if is_string_dtype(df[j]): 

                # dtype,
                df_examples.loc[df_examples.name==j,"dtype"]= "text"
                
                # All_values_are_unique,
                # use helper function, to find if there are only unique categorical values eg: url, place example in dct,  
                df_examples.loc[df_examples.name==j,"All_values_are_unique"] = fillin_All_values_are_unique(df_examples=df_examples, df=df)

                # Examples,
                count_noNa_values_sorted = df.loc[:,j].dropna().value_counts().sort_values(ascending=False)
                perc_noNa_values_sorted  = count_noNa_values_sorted/np.sum(count_noNa_values_sorted)*100
                s        = perc_noNa_values_sorted[0:nr_of_examples_per_category].round(1)
                ind      = pd.Series(s.index).str[0:12].values.tolist()
                df_examples.loc[df_examples.name==j,"Examples"] = ";".join([str((x,y)) for x,y in zip(["".join([str(x),"%"]) for x in list(s)],ind)])
   
                # add examples to arr for plot
                arr_example_percentage[df_examples.name==j,0:s.values.size]=s.values
                arr_example_values[df_examples.name==j,0:s.values.size]= np.array(s.index)


            elif is_numeric_dtype(df[j]): 

                # dtype,
                df_examples.loc[df_examples.name==j,"dtype"]= "numeric"
                
                # All_values_are_unique,
                x = list(df.loc[:,j].dropna().describe()[["min", "mean", "max"]].round(3))
                df_examples.loc[df_examples.name==j,"All_values_are_unique"] = f"{round(x[0],2)} // {round(x[1],2)} // {round(x[2],2)}"

                # Examples,
                ordered_values = df.loc[:,j].dropna().value_counts().sort_values(ascending=False)
                ordered_values = (ordered_values/df.loc[:,j].dropna().shape[0])*100
                df_examples.loc[df_examples.name==j,"Examples"] = ";".join([str((x,y)) for x,y in zip(
                    ["".join([str(int(np.ceil(x))),"%"]) for x in list(ordered_values)][0:nr_of_examples_per_category],
                    list(ordered_values.index.values.round(3))[0:nr_of_examples_per_category])])

                # add examples to arr for plot
                vn = np.array(ordered_values.index)[0:nr_of_examples_per_category]
                vp = ordered_values.values[0:nr_of_examples_per_category]
                arr_example_values[df_examples.name==j,0:ordered_values.size] = vn
                arr_example_percentage[df_examples.name==j,0:ordered_values.size] = vp
                
            elif is_datetime64_any_dtype(df[j]): 

                # dtype,
                df_examples.loc[df_examples.name==j,"dtype"]= "datetime"
                
                # variable summary,
                first_and_last_date = [str(x) for x in list(df.loc[:,j].dropna().describe()[["first", "last"]].dt.strftime('%b %d %Y'))]
                df_examples.loc[df_examples.name==j,"All_values_are_unique"] = f"{first_and_last_date[0]} - {first_and_last_date[1]}"

                # Examples,
                ordered_values = df.loc[:,j].dropna().value_counts().sort_values(ascending=False)
                ordered_values = (ordered_values/df.loc[:,j].dropna().shape[0])*100
                df_examples.loc[df_examples.name==j,"Examples"] =";".join([str((x,y)) for x,y in zip(
                    ["".join([str(np.round(x,3)),"%"]) for x in list(ordered_values)][0:2], 
                    list(pd.Series(ordered_values.index[0:2]).dt.strftime('%b-%d-%Y %H:%M').values))])# i add only two, because these are really long, 

                # add examples to arr for plot
                vn = list(ordered_values.index)[0:nr_of_examples_per_category]
                vp = ordered_values.values[0:nr_of_examples_per_category]
                arr_example_values[df_examples.name==j,0:ordered_values.size] = vn
                arr_example_percentage[df_examples.name==j,0:ordered_values.size] = vp     
                
            else:
                pass

                # (df_examples.loc[df_examples.name==j,"NaN_perc"]==100).values[0]: this value was rounded up/down and was returning false positives!!!!
                df_examples.loc[df_examples.name==j,"All_values_are_unique"] = "Uknown datatype"
                df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"] = 0 # it should be 0, but i overwrite it just in case
                df_examples.loc[df_examples.name==j,"Nr_of_unique_values"] = 0 # it should be 0, but i overwrite it just in case
                df_examples.loc[df_examples.name==j,"Examples"] = "Uknown datatype"
                df_examples.loc[df_examples.name==j,"dtype"] = "Uknown datatype" # because I dont want to use that in any further reading


                
    # reorder column, so the look nicer, when displayed, 
    df_examples = df_examples.loc[:,["name", "dtype", 'NaN_perc', 'All_values_are_unique', 'Examples', 
                                     'Nr_of_unique_values', 'Nr_of_non_null_values', 'nr_of_all_rows_in_original_df']]   
    # rename some collumns, for PEP8 compliance,
    df_examples = df_examples.rename(columns={'All_values_are_unique':'summary', "Examples": "examples", 
                                              "Nr_of_unique_values": "nr_of_unique_values", "Nr_of_non_null_values":"nr_of_non_null_values"})
    

    # turn two additional elements into df's, 
    df_example_values = pd.DataFrame(arr_example_values, index=df_examples.name)
    df_example_percentage = pd.DataFrame(arr_example_percentage, index=df_examples.name)
    
    #### save the files 
    if csv_file_name!=None and save_dir!=None:
        try:
            os.chdir(save_dir)
            df_examples.to_csv("".join([csv_file_name,".csv"]), encoding='utf-8', index=False)
            df_example_values.to_csv("".join(["top_val_names_",csv_file_name,".csv"]), encoding='utf-8', index=False)
            df_example_percentage.to_csv("".join(["top_val_perc_",csv_file_name,".csv"]), encoding='utf-8', index=False)
            
            # info to display,and table example, 
            if verbose==True:
                print(f"""the file: {csv_file_name} was correctly saved \n in: {os.getcwd()} \n""") 
            else:
                pass

        except:
            if verbose==True:
                Error_message = "THE FILE WAS NOT SAVED, \n save_dir and/or csv_file_name were incorrect, or one of them was not provided"
                print(f"""{"".join(["."]*40)},\n ERROR,\n the file: {csv_file_name},\n {Error_message} \n in: {os.getcwd()} \n{"".join(["."]*40)}""")
            else:
                pass
                
    else:
        if verbose==True:
            Error_message = "THE FILE WAS NOT SAVED, \n save_dir and/or csv_file_name were not provided "
            print(f"""{"".join(["."]*40)},\n CAUTION,\n the file: {csv_file_name},\n {Error_message} \n in: {os.getcwd()} \n{"".join(["."]*40)}""")
        else:
            pass
    
    return df_examples, df_example_values, df_example_percentage
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

# Function, ..................................................................................
def create_class_colors_dict(*, list_of_unique_names, cmap_name="tab20", cmap_colors_from=0, cmap_colors_to=1):
    '''Returns dictionary that maps each class name in list_of_unique_names, 
       to to a distinct RGB color
       . list_of_unique_names : list with unique, full names of clasesses, group etc..
       . cmap_name : standard mpl colormap name.
       . cmap_colors_from, cmap_colors_to, values between 0 and 1, 
         used to select range of colors in cmap, 
     '''
    
    # create cmap
    mycmap = plt.cm.get_cmap(cmap_name, len(list_of_unique_names)*10000)
    newcolors = mycmap(np.linspace(cmap_colors_from, cmap_colors_to, len(list_of_unique_names)))

    class_color_dict = dict()
    for i, un in enumerate(list_of_unique_names):
        class_color_dict[un] = newcolors[i]
    
    return class_color_dict





# Function ...........................................................................
# new
def annotated_pie_chart_with_class_and_group(*, 
                                             # data
                                             classnames, 
                                             groupnames=None, 
                                             
                                             # general fig/plot aestetics
                                             title=None, 
                                             title_ha="right",
                                             title_fontsize_scale=1,
                                             class_colors=None, 
                                             groupname_colors=None,
                                             class_colors_cmap="tab20",
                                             cmap_colors_from =0, 
                                             cmap_colors_to =1,
                                             
                                             # fig size & layout
                                             figsze_scale=1, 
                                             figwidth_scale=1, 
                                             figheight_scale=1,                                      
                                             n_subplots_in_row=3, 
                                             subplots_adjust_top=0.9, 
                                             tight_lyout=False, 
                                             
                                             # legend, fonts and additional text
                                             legend=False, 
                                             legend_loc="center", 
                                             legend_ncol=6, 
                                             legend_fontsize_scale=1, 
                                             feature_name="In Total",
                                                
                                             # piecharts on each subplot
                                             ax_title_fontcolor=None, 
                                             ax_title_fonsize_scale=1, 
                                             wedges_fontsize_scale=1, 
                                             add_group_name_to_each_pie=True, 
                                             add_group_item_perc_to_numbers_in_each_pie=False, 
                                             mid_pie_circle_color="lightgrey", 
                                             
                                             verbose=False
                                            ):
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          function crerates annotated pie charts with empty center, 
                            annotations, have name of the class, number of instances and pecentage of instances, 
                            in the total population
                            optionally, the functions can take second argument, groupnames, of the same lenght as cvlassnames, 
                            if used, groupnames, will be used to create separate annotated pie chart, for each uniqwue groupname, 
                            with groupname in the middle of the pie chart.

        # Inputs
        .......................     ...........................................................................
        . classnames                : list, with repeated instances of items that will be counted and presented as classes on pie chart
        . groupnames                : list, with repeated instances of groupnames, used to create separate pie charts, 
                                      default=None, 
        . title                     : str, title above the figure, with all images, 
        . verbose                   : bool, default=False
        . class_colors              : dictionary,  {str <"class_name">: str <"color">} 
                                      used, to color pie classes on pie chart
        . groupname_colors          : dictionary,  {str <"group_name">: str <"color">}
                                      used to color group name, in the middle of pie chart - a gorupname, 
                                     CAUTION: colors and class names must be unique !
        # Returns
        .......................     ...........................................................................
        Matplotlib figure, 
        
        # Notes
        Pie chart idea taken from
        https://matplotlib.org/3.1.0/gallery/pie_and_polar_charts/pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py
        
        
        
    """

    # small correction, on error i did with names while creasting this function
    img_classnames = classnames
    img_groupnames = groupnames
    
    
    # .................................................................
    # DATA PREPARATION,  
    if img_groupnames==None: 
        img_groupnames =  [feature_name]*len(img_classnames)
        if verbose==True: 
            print("img_groupname were not specified ...  all images will be plotted one after anothe, as they woudl belong to one group, cluster, ...")
        else: 
            pass
    else: 
        pass
    # ...
    groups_to_plot = pd.Series(img_groupnames).unique().tolist()


    # .................................................................
    # FIGURE PARAMETERS, 
    
    # figsize aand subplot number 
    if len(groups_to_plot)<=n_subplots_in_row:
        fig_nrows = 1
        fig_height = 4.5
        # ...
        fig_ncols = len(groups_to_plot)
        figsize_width = fig_ncols*5*figsze_scale
    
    if len(groups_to_plot)>n_subplots_in_row:
        fig_nrows = int(np.ceil(len(groups_to_plot)/n_subplots_in_row))
        fig_height = fig_nrows*4
        # ...
        fig_ncols = n_subplots_in_row
        figsize_width = 5*n_subplots_in_row*figsze_scale
    # ..
    fig_size = (figsize_width*figwidth_scale, fig_height*figheight_scale)
    
    # ..
    title_fonsize    = 40
    ax_title_fonsize = title_fonsize*0.4*ax_title_fonsize_scale
    wedges_fontsize  = title_fonsize*0.25*wedges_fontsize_scale
    
    # pie dimensions, 
    pie_size_scale   = 0.8 # proportion of the plot in x,y dimensions
    pie_width_proportion = 0.33

    # class colors, - chnages because I added legend that looks really nice, 
    if class_colors==None:
        class_colors = create_class_colors_dict(
            list_of_unique_names = pd.Series(img_classnames).unique().tolist(),
            cmap_name=class_colors_cmap, 
            cmap_colors_from = cmap_colors_from, 
            cmap_colors_to = cmap_colors_to
            )
    else:
        pass
    
    # .................................................................
    # FIGURE,     
    
    # Figure and axes, 
    mpl.rcParams.update(mpl.rcParamsDefault) # to clear all settings, 
    fig, axs = plt.subplots(ncols=fig_ncols, nrows=fig_nrows, figsize=(fig_size), facecolor="white")

    # .. add title, 
    if title!=None: 
        fig.suptitle(title, fontsize=title_fonsize*0.6*title_fontsize_scale, color="black", ha=title_ha)
    else: 
        pass

    if len( groups_to_plot)==1:
        axss = [axs]
    else:
        axss = axs.flat
    
    
    # .. create each subplot with pie annotated chart, 
    for ax_i, ax in enumerate(axss):
    
    
        if ax_i>=len(groups_to_plot):
            # empty, plot, so clear axis,  and keep it that way, 
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])      
            ax.spines["right"].set_visible(False) # and below, remove white border, 
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)            
        
        else:

            # set group name for a given subplot, 
            one_groupname = groups_to_plot[ax_i]


            # clear axis, - saves a bit of space,  
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])      
            ax.spines["right"].set_visible(False) # and below, remove white border, 
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            # select classnames 
            s = pd.Series(img_classnames).loc[pd.Series(img_groupnames)==one_groupname]
            s_item_number = s.shape[0]
            s = s.value_counts()

            # find colors for pie chart
            if class_colors!=None:
                one_group_pie_colors = list()
                for j, cn in enumerate(s.index.values.tolist()):
                    one_group_pie_colors.append(class_colors[cn])
            else:
                one_group_pie_colors=None

            # create description for each calls with its percentage in df column
            pie_descr = list(s.index)
            data      = [float(x) for x in list(s.values)]
            pie_descr = [f"{y}: {str(int(x))} ({str(np.round(x/np.sum(data)*100))}%)" for x,y in zip(data, pie_descr)]
            # pie
            wedges, texts = ax.pie(
                data, 
                wedgeprops=dict(width=pie_width_proportion*pie_size_scale),  # Caution, here must be two numbers !!!
                radius=pie_size_scale,
                startangle=-60, 
                counterclock=False,
                colors=one_group_pie_colors
            )

            # params for widgets
            bbox_props = dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="k", lw=1, alpha=0.3)
            kw = dict(arrowprops=dict(arrowstyle="->"),
                      bbox=bbox_props, zorder=10, va="center", fontsize=wedges_fontsize)

            # add widgest to pie chart with pie descr
            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                y = np.sin(np.deg2rad(ang))*pie_size_scale
                x = np.cos(np.deg2rad(ang))*pie_size_scale
                # ...
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = "angle,angleA=0,angleB={}".format(ang)
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                # ...
                ax.annotate(pie_descr[i], xy=(x, y), xytext=(1*np.sign(x), 1.4*y),
                            horizontalalignment=horizontalalignment, **kw)

            # add groupname, in the center of pie chart, 
            
            # .. if, available set color for groupname
            if groupname_colors==None:
                if ax_title_fontcolor==None:
                    font_color="black"
                else:
                    font_color=ax_title_fontcolor
                # ensure that anythign is visible, eg donth have black text on black bacground    
                if mid_pie_circle_color==font_color:
                    font_color="white"
                else:
                    pass
                patch = plt.Circle((0, 0), (pie_size_scale-pie_width_proportion), zorder=0, alpha=1, color=mid_pie_circle_color)
                ax.add_patch(patch)
            else:
                if ax_title_fontcolor==None:
                    font_color="white"
                else:
                    font_color=ax_title_fontcolor
                # ensure that anythign is visible, eg donth have black text on black bacground
                if groupname_colors[one_groupname]==font_color:
                    font_color="white"
                else:
                    pass                   
                one_groupname_color=groupname_colors[one_groupname]
                patch = plt.Circle((0, 0), (pie_size_scale-pie_width_proportion), zorder=0, alpha=1, color=one_groupname_color)
                ax.add_patch(patch)
            
            # .. add group name with larger font, and number associated with that group (item count and % in total dataset)
            if len(groups_to_plot)>1 or add_group_name_to_each_pie==True:
                font = FontProperties()
                # ..
                font.set_weight("bold")
                font.set_size(ax_title_fonsize)
                ax.text(0, 0, one_groupname, fontsize=ax_title_fonsize, ha="center", color=font_color, fontproperties=font)
                # ...
                font.set_size(wedges_fontsize)
                if add_group_item_perc_to_numbers_in_each_pie==True:
                    ax.text(0, -0.2, f"{s_item_number}, ({np.round((s_item_number/len(img_classnames)*100),1)}%)", 
                          fontsize=wedges_fontsize, ha="center", fontproperties=font, color=font_color)           
                else:
                    ax.text(0, -0.2, f"{s_item_number}", 
                          fontsize=wedges_fontsize, ha="center", fontproperties=font, color=font_color)           



    # .............................................................................
    # LEGEND 
    if legend==True:
        # create patch for each dataclass, - adapted to even larger number of classes then selected for example images, 
        patch_list_for_legend =[]
        count_items = 0
        for i, cl_name in enumerate(list(class_colors.keys())):
            cl_color = class_colors[cl_name]
            patch_list_for_legend.append(mpatches.Patch(color=cl_color, label=cl_name))

        # add patches to plot,
        fig.legend(
            handles=patch_list_for_legend, 
            frameon=False, 
            scatterpoints=1, ncol=legend_ncol, 
            fontsize=ax_title_fonsize*0.8*legend_fontsize_scale,
            loc=legend_loc
        )                
    else:
        pass

    # .............................................................................
    # END   
    
    if tight_lyout==True:
        plt.tight_layout()
    else:
        pass
    plt.subplots_adjust(top=subplots_adjust_top)
    plt.show();
    
    

    
    
# Function ...........................................................................
def prepare_img_classname_and_groupname(*, data_for_plot, groupname_prefix="Cluster ", number_of_img_examples=100, plot_img_from=None, plot_img_to=None):
    """
        Helper function to get img class name and group name for annotated pie charts, 
        from results obtained after images examples were plotted with plot_img_examples_from_dendrogram()
    """

    # set idx 
    if plot_img_from!=None and plot_img_to!=None:
        img_idx = data_for_plot['img_order_on_dedrogram'][plot_img_from:plot_img_to].tolist()
    else:
        temp = np.unique(np.floor(np.linspace(0, data_for_plot['batch_labels'].shape[0], number_of_img_examples, endpoint=False)).astype(int))
        img_idx = data_for_plot['img_order_on_dedrogram'][temp]

    # find idx if images in batch_labels, but ordered as on dendrogram, 
    selected_df_for_plot = data_for_plot['batch_labels'].loc[img_idx,:]
    selected_df_for_plot.reset_index(drop=False, inplace=True)

    # preapre df with selected data for the plot¨
    img_classname = selected_df_for_plot.classname.values.tolist()
    img_groupname = ["".join([groupname_prefix,str(x)]) for x in selected_df_for_plot.loc[:, "dendrogram_clusters"].values.tolist()]
    
    return img_classname, img_groupname






# Function, ............................................................................ 
def annotated_pie_chart(*, ax, s, title, font_size=10):    
    """
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          Pie chart to diplay categorical data, with % and numerical values in ticks
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        * ax                fig axis from matplotlib.pyplot
        * s                 Pandas, series with repeated records that will be counted and displayes as pie chart pieces
                            ! caution ! Mosre then 5 classes may cause problmes, in that case its better to to use
                            barplot.
        .
        * title             str, ax.set_title("title")
        * font_size         int, ticks fontsize
        
        Returns             
        _________________   _______________________________________________________________________________
        
        * matplotlib 
          figure axis object
        
        * example           https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_and_donut_labels.html
    """
        
    # create description for each calls with its percentage in df column
    s = s.value_counts()
    pie_descr = list(s.index)
    data      = [float(x) for x in list(s.values)]
    pie_descr = ["".join([str(int(x))," colums with ",y,
                 " (",str(np.round(x/np.sum(data)*100)),"%)"]) for x,y in zip(data, pie_descr)]

    # pie
    pie_size_scale =0.8
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5*pie_size_scale), radius=pie_size_scale,startangle=-45)
    
    # params for widgets
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    kw = dict(arrowprops=dict(arrowstyle="->"),
              bbox=bbox_props, zorder=0, va="center", fontsize=font_size)

    # add widgest to pie chart with pie descr
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))*pie_size_scale
        x = np.cos(np.deg2rad(ang))*pie_size_scale
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(pie_descr[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)
    ax.set_title(title)
    



    


# Function, ............................................................................ 
def annotated_barplot(*, 
    # input data
    data_examples, 
    top_val_perc,                                                                        
    df_filter, 
    
    # plot aestetics
    plot_title="", 
    fig_size=(12,12), 
    fontsize_scale=1,
    examples_fontsize_scale=1,
    group_size=5,
    barplot_cmap="tab10",
    cmap_from=0, 
    cmap_to=0.5,
    adjust_top=0.8
):
    
    
    '''
        Generates bar plot used to get fast information on data 
        in different column in large df
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        
        . Input .
        * data_examples     DataFrame with large Dataframe summary, 
                            generated with  summarize_data_and_give_examples()
        * top_val_perc      DataFrame with % of the top three or most frequence records in each column 
                            in large dataframe that was summarized with summarize_data_and_give_examples()
        
        .
        * df_filter         list, with True/False for each row in data_examples & top_val_perc 
                            if True, the row will be displayed on barplot
                            
        * fig_size          tuple, (row lenght, col lenght), in inches
        * font_size         int, size of all fonts used on the plot
        * group_size        int, how many rows will be diplayes as group on y axis on horizonal barplot.
                            groups are divided by space == to one bar.
                            
        Returns             
        _________________   _______________________________________________________________________________
        
        * plt.figure 
    '''
    
    # basic fontsize:
    font_size=8
    
    # helper,
    def stacked_barh_one_level(*, f_ax, bar_pos, top, bottom, colors, edgecolor, labels):
        f_ax.barh(bar_pos, top,left=bottom, color=colors, edgecolor=edgecolor, label=labels, linewidth=0.5, height=0.6)
        return f_ax    
    
    
    # ............................................................
    # Names and group filtering

    # group names,
    group_names       = list(data_examples.name.loc[df_filter])

    # data for plot,
    data_completness  = 100-np.array(data_examples.NaN_perc[df_filter]).flatten()
    tick_description  = data_examples.name[df_filter]
    top_values        = top_val_perc.values[df_filter, :]
    top_data_examples = data_examples.examples[df_filter]
    group_description = data_examples.summary[df_filter]

    # rescale top values,so they are part of non-missing data
    for i in range(top_values.shape[1]):
        v = top_values[:,i]
        top_values[:,i] = (v*data_completness)/100
    all_remaining_values = data_completness-top_values.sum(axis=1)

    # join the data in one array, I had some problems here, 
    data_for_plot = np.round(np.c_[(np.round(top_values,1), all_remaining_values)],1)

    

    # ............................................................
    # order the bars,

    # find order of the bars, based on data completness,
    bar_order = pd.DataFrame(np.c_[(data_completness, np.arange(data_completness.size))]).sort_values(0, ascending=True)
    bar_order.reset_index(inplace=True, drop=True)
    bar_order = pd.concat([bar_order, pd.Series(list(bar_order.index))], axis=1)
    bar_order = bar_order.sort_values(1,ascending=True)
    bar_order = np.array(list(bar_order.index))

    # add spaces between everyx n-th bar, 
    add_spacers = True
    if add_spacers==True:
        # add spaces between everyx 5th bar, 
        space_between_groups = 1
        new_br = bar_order.copy().flatten()
        group_top  = []
        group_bottom = []

        for i, j in enumerate(sorted(list(bar_order))):

            if i==0: 
                add_to_list, counter = 0, 0
                group_bottom.append(j)

            if i>0 and counter<group_size: 
                counter +=1       

            if counter==group_size:
                group_bottom.append(j+add_to_list+1)
                counter=0
                add_to_list +=space_between_groups; 

            new_br[bar_order==j]=j+add_to_list

        group_top = [x+group_size-1 for x in group_bottom]    
        group_top[-1] = np.max(bar_order)+add_to_list
        bar_order = new_br.copy()

        
        
    # ............................................................
    # barplot parameters; this was just to help me in long function !
    numeric_data_for_plot = data_for_plot # np array, 
    top_data_examples     = top_data_examples
    bar_position          = bar_order + 1
    group_description     = group_description
    bar_related_fontsize  = font_size
    
    # bar_names, (ytick labels),
    # if len(list(data_examples.dtype[df_filter].unique()))>1:
    df_bar_names       = pd.DataFrame({"col_1":group_names, "col_2":list(data_examples.dtype[df_filter])}) 
    df_bar_names.col_2 = df_bar_names.col_2.str.pad(width=20, side="left", fillchar=".")
    bar_names          = list(df_bar_names.col_1.str.cat([", "]*df_bar_names.shape[0]).str.cat(df_bar_names.col_2))
    #else:
    #    bar_names          = group_names # list, old script, now chnaged as in below
        
  

    # ............................................................   
    # barplot,

    #### prepare data and figure, 
    
    # Set style and colors,
    plt.style.use("classic")
    bar_colors = plt.get_cmap(barplot_cmap)(np.linspace(cmap_from, cmap_to, data_for_plot.shape[1])) # different nr of colors,
    edge_colors = bar_colors.copy()
    bar_colors[-1,0:3] = colors.to_rgb("lightgrey")
    edge_colors[-1,0:3] = colors.to_rgb("grey")

    # fig
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size, facecolor="white")
    fig.suptitle(plot_title)
    plot_top_value = np.max(bar_position)+15
    ax.set_ylim(0,plot_top_value)
    ax.set_xlim(0,300)

    # add top values as % od data completness,
    counter =[]
    add_top_values=True
    if add_top_values==True:
        counter = 0
        for i in list(range(data_for_plot.shape[1]))[::-1]:
            if counter == 0:
                bar_start = [0]*data_for_plot.shape[0]    
                bar_end   = data_for_plot[:,i]
            else:
                bar_start = bar_start+bar_end           
                bar_end   = data_for_plot[:,i] # bar end is hot tall is an individual bar, segment, not top point on a graph
            counter+=1

            # plot level on stacked plot
            ax = stacked_barh_one_level(
                 f_ax=ax, 
                 bar_pos=bar_position, top=bar_end, bottom=bar_start, 
                 colors=bar_colors[i], edgecolor=bar_colors[i], labels="test",
                 )


            
    #### ad legend on y axis        

    # Add ticks on y axis, and names for each bar,
    ax.set_yticks(bar_position)
    ax.set_yticklabels(bar_names, fontsize=bar_related_fontsize*fontsize_scale, color="black")
    ax.set_xticks([0, 25,50,75,100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=bar_related_fontsize*fontsize_scale, color="black")

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

    # Add vertical lines from grid,
    ax.xaxis.grid(color='grey', linestyle='--', linewidth=1) # horizontal lines

    # add patch on top to remove surplus gridlines
    x_left      = -100
    rect_width  = 500
    y_bottom    = np.max(bar_position)+1.4 # add a bit to nut cut text opr boxes,
    rect_height = 500
    rect = mpatches.Rectangle(
        xy=(x_left,y_bottom),
        width=rect_width,
        height=rect_height,
        linewidth=0,
        edgecolor='white',
        facecolor='white',
        alpha=1, 
        zorder=10
    )
    ax.add_patch(rect)
    plt.ylim(top=np.max(bar_position)+1.4)

    # axes desciption    
    ax.set_xlabel(f"Percentage of non-missing data, rows in total={data_examples.nr_of_all_rows_in_original_df.iloc[0]}                           ", ha="right") # I intentionally, added these spaces here!
    ax.set_ylabel("Column name, datatype", ha="center")


    
    #### add, numbers and examples on a left side of the barplot, 

    # add rectagles arrnoud examples
    for i, j in zip(group_bottom, group_top):
        x_left      = 113
        rect_width  = 186
        y_bottom    = i+0.2
        rect_height = j-i+1.5
        rect = mpatches.Rectangle(
            xy=(x_left,y_bottom),
            width=rect_width,
            height=rect_height,
            linewidth=1,
            edgecolor="darkgreen",
            facecolor='yellow',
            alpha=0.3
        )
        ax.add_patch(rect)

    # add text with data completness above each bar,
    add_text_wiht_data_completness_above_each_bar=True
    if add_text_wiht_data_completness_above_each_bar==True:
        for i in range(numeric_data_for_plot.shape[0]):
            text_y_position = bar_position[i]-0.3
            text_x_position = numeric_data_for_plot.sum(axis=1).tolist()[i]+2

            # text,
            text_to_display = "".join([str(int(np.round(numeric_data_for_plot.sum(axis=1).tolist()[i],0))),"%"])
            t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=bar_related_fontsize*fontsize_scale, color="darkred")
            #t.set_bbox(dict(facecolor="white", alpha=0.3, edgecolor="white"))
    else: 
        pass

    # Add table, with data to plot,
    for i in range(numeric_data_for_plot.shape[0]):
        text_y_position = bar_position[i]-0.3
        text_x_position = 115

        # text,
        text_to_display = list(group_description)[i]
        t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=bar_related_fontsize*fontsize_scale, color="black")

    # add examples,   
    for i in range(numeric_data_for_plot.shape[0]):
        text_y_position = bar_position[i]-0.3
        text_x_position = 170

        # text,
        if re.search("all nonnull",str(list(group_description)[i])):
            text_to_display = "".join(["- - - > ",list(top_data_examples)[i]])
        else: text_to_display = list(top_data_examples)[i]
        t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=(bar_related_fontsize)*fontsize_scale*examples_fontsize_scale, color="black")

        
        
    #### add plot legend  
    box_color       = "yellowgreen"
    box_edge_color  = "darkgreen"
    text_color      = "black" 
    text_size       = bar_related_fontsize

    text_x_position = 3
    text_y_position = np.max(bar_position)+2.5
    text_to_display = '''BAR DESCRIPTION\n- each bar shows % of non-missing data in a given columns\n- Colour bars on top, shows the % of the most frequent classes'''   
    t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=text_size*fontsize_scale, color=text_color, ha="left")
    t.set_bbox(dict(facecolor=box_color, alpha=1, edgecolor=box_edge_color))

    text_x_position = 115
    text_y_position = np.max(bar_position)+2.5
    text_to_display = '''FEATURE SUMMARY \n - numeric.: min; mean; max \n - string/time: nr of classes'''
    t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=text_size*fontsize_scale, color=text_color, ha="left")
    t.set_bbox(dict(facecolor=box_color, alpha=1, edgecolor=box_edge_color))

    text_x_position = 175
    text_y_position = np.max(bar_position)+2.5
    text_to_display = '''EXAMPLES of the most Frequent Non-Missing Values:\n - first: %of rows, with a given class, \n - second: class value, or the first 15-th characters'''
    t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=text_size*fontsize_scale, color=text_color, ha="left")
    t.set_bbox(dict(facecolor=box_color, alpha=1, edgecolor=box_edge_color))
    
    fig.subplots_adjust(top=adjust_top)
    plt.show();
    


 

# Function, ............................................................................
def df_summary_table(*, 
    df, 
    fig_size=(8,8),
    title="",
    fontsize_scale=1
):
    
    """
        Plots image of a table with basic statitics on the amount 
        of missing and non-missing data in large dataFrame
        
        Parameters/Input              
        _________________   _______________________________________________________________________________        
        * df.               DataFrame with large Dataframe summary, 
                            generated with  summarize_data_and_give_examples()
        * fig_size          tuple, (int, int)
        * fig_numer         fiugure number that will be added to that plot
        
        Returns               
        _________________   _______________________________________________________________________________
        
        * plt figure
    """    
        
    # data preparatio
    col_nr = int(df.shape[0])
    nr_of_columns_with_no_NAN = (df.NaN_perc==0).sum()
    nr_of_columns_with_any_NAN = df.loc[(df["NaN_perc"]>0)].shape[0]
    nr_of_columns_with_over_50per_NAN = df.loc[(df["NaN_perc"]>50)].shape[0]
    nr_of_columns_with_over_90per_NAN = df.loc[(df["NaN_perc"]>90)].shape[0]
    ###
    mean_nr_of_noNaN_per_row = str((df.nr_of_non_null_values.sum()/df.nr_of_all_rows_in_original_df[0]).round(3))
    mean_perc_of_noNaN_per_row = str(((df.nr_of_non_null_values.sum()/df.nr_of_all_rows_in_original_df[0])/col_nr*100).round(3))
    ###
    mean_nr_of_NaN_per_row = str((col_nr-df.nr_of_non_null_values.sum()/df.nr_of_all_rows_in_original_df[0]).round(3))
    mean_perc_of_NaN_per_row = str(((col_nr-df.nr_of_non_null_values.sum()/df.nr_of_all_rows_in_original_df[0])/col_nr*100).round(3))


    # collect all data for the table:
    dct_table_numbers={
    "Row number": str(int(df.nr_of_all_rows_in_original_df.mean().round(0))),
    "Column number": str(df.shape[0]),
    ".    " : str(""),  
    "non-missing data per Row": mean_nr_of_noNaN_per_row,
    "missing data per Row":  mean_nr_of_NaN_per_row,
    "....  " : str(""),        
    "Mean %/nr of non-missing data per column": str(int(df.nr_of_non_null_values.mean().round(0))),
    "Mean %/nr of unique values per column" : str(int(df.nr_of_unique_values.mean().round(0))),
    "    " : str(""),       
    "Columns with no missing data": nr_of_columns_with_no_NAN,
    "Columns with any missing data":  nr_of_columns_with_any_NAN,
    "Columns with >50% of missing data": nr_of_columns_with_over_50per_NAN,    
    "Columns with >90% of missing data": nr_of_columns_with_over_90per_NAN
    }
    
    dct_table_perc={
    "Row number": "100%",
    "Column number": "100%",
    " " : str(""),
    "Non-missing data per Row": "".join([mean_perc_of_noNaN_per_row,"%"]),
    "Missing data per Row": "".join([mean_perc_of_NaN_per_row,"%"]),
    "....  " : str(""),    
    "Non-missing data per column": "".join([str(((df.nr_of_non_null_values/df.nr_of_all_rows_in_original_df).mean()*100).round(1)),"%"]),
    "Unique values per column" : "".join([str(((df.nr_of_unique_values/df.nr_of_all_rows_in_original_df).mean()*100).round(1)),"%"]),        
    "     " : str(""),
    "Columns with no missing data": "".join([str(np.round((int(nr_of_columns_with_no_NAN)/col_nr)*100,1)), "%"]),
    "Columns with any missing data": "".join([str(np.round((int(nr_of_columns_with_any_NAN)/col_nr)*100,1)), "%"]),
    "Columns with >50% of missing data": "".join([str(np.round((int(nr_of_columns_with_over_50per_NAN)/col_nr)*100,1)), "%"]),   
    "Columns with >90% of missing data": "".join([str(np.round((int(nr_of_columns_with_over_90per_NAN)/col_nr)*100,1)), "%"])
    } 
    
    # np.array with data so they can be displayed on plot
    arr = np.zeros([len(dct_table_numbers.keys()), 3], dtype="object"); arr
    arr[:,0] = list(dct_table_numbers.keys())
    arr[:,1] = list(dct_table_perc.values())
    arr[:,2] = list(dct_table_numbers.values())
    arr[(2,5,8),0] = [".", ".","."]
     
    # figure
    sns.set_context("notebook")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size, facecolor="white")
    fig.suptitle(title, fontsize=12*fontsize_scale)
    
    # add table to ax
    table = plt.table(
        cellText=arr, 
        colLabels=['Category',"%", "Number" ], 
        loc='center', 
        cellLoc='left', 
        colColours=['lightgrey']*3, 
        fontsize=10*fontsize_scale
    )
    table.auto_set_column_width((-1, 0, 1, 2, 3))
    table.scale(1, 3)
    #table.auto_set_font_size(False) # numbers look better with autmatic fontsize setting and sns.set_style()
    table.set_fontsize(10*fontsize_scale)
    
    # remove all plot spines, ticks and labels:
    ax.spines['top'].set_visible(False) # remove axis ...
    ax.spines['right'].set_visible(False) # ...
    ax.spines['left'].set_visible(False) #  ...
    ax.spines['bottom'].set_visible(False) #  ...
    ax.xaxis.set_ticks_position('none')# remove ticks ...
    ax.yaxis.set_ticks_position('none')# ...
    ax.set_xticklabels("")
    ax.set_yticklabels("")
    
    # adjust
    fig.subplots_adjust(top=0.65)# to adjust for the titl
    
    
    
    
    
    
# Function, ............................................................................    
def df_summary_plot(*, 
    # input data (either df or data_examples & top_values_perc)
    df=None,
    df_top_n=3,
    data_examples=None, 
    top_values_perc=None, 
    
    # options on dat to display, 
    groups_to_display=None, 
    pieChart=True, 
    showTable=True,
    barPlot=True,
               
                    
    # settings, 
    barPlot_figsize=None,
    barPlot_groupSize=None,
    barPlot_dct=dict(),
    pieChart_dct=dict(),
    table_dct=dict(),
               
    verbose=False      
               
):
 
    """
        Plots Pie chart, table and barplot summarizing data in large dataFrame
        
        Parameters/Input              
        _________________   _______________________________________________________________________________  
        
        . Input .
        
        * data_examples     DataFrame with large Dataframe summary, 
                            generated with  summarize_data_and_give_examples()
        * top_val_perc      DataFrame with % of the top three or most frequence records in each column 
                            in large dataframe that was summarized with summarize_data_and_give_examples()
        * groups_to_display str, or list with strings, {"all", "text", "numeric", "datetime"}
                            "all", (default), or one of the dtypes, in data_examples.dtype, 
                            or list with different dtypes that will be ploted on rseraprate barplots
                            Columns only with missing data are not included in groups, these are plotted
                            only with "all" default option
                
        . Parameters . 
        
        * start_figure_numbers_at 
                            >=1, how nto start numeration of the figures with plots
        * pieChart          if True (default), display Pie chart with dtypes detected in data_examples
                            with number start_figure_numbers_at 
        * showTable         if True (default), display image of a summary table
                            with number start_figure_numbers_at + 1
        * barPlot.          if True (default), displays
                            with number start_figure_numbers_at  + 2, 3,4 and so on for each dtype
                            
                            
        Returns               
        _________________   _______________________________________________________________________________
        
        * Pie chart.        by :   pie_chart_with_perc_of_classes_df_column()
        * Tamble image.     by :   table_with_stats_on_missing_non_missing_data_in_df()
        * BarPlot           by :   barplot_with_data_completness_class_description_and_top_value_examples()
        
    """    

    # set up groups to be selected 
    if groups_to_display==None:
        groups_to_display="all"
    elif isinstance(groups_to_display, str):
        if groups_to_display=="all":
            pass
        else:
            groups_to_display=[groups_to_display]
    else:
        pass
    
    
    
    
    # prepare data directly from datafram, or use summary df elements caluated separately, 
    if data_examples is None or top_values_perc is None: 
        data_examples, _, top_values_perc = summarize_df(  
            df = df, 
            nr_of_examples_per_category = df_top_n,
            csv_file_name = None, 
            save_dir = None,
            verbose=verbose
        )
    else:
        pass

 
    # Pie chart with nr of col with different dtypes in data df,
    if pieChart==True:
        
        '''
            # old code:
            sns.set_context("notebook")
            fig, axs = plt.subplots(nrows=1, ncols=1,
                                    figsize=(4, 5), facecolor="white",
                                   subplot_kw=dict(aspect="equal"))
            fig.suptitle(f"DATA TYPES", fontsize=18)
            annotated_pie_chart(ax=axs, s=data_examples["dtype"], title="")
            plt.subplots_adjust(top=0.9)
            plt.show() 
        
        '''        
        annotated_pie_chart_with_class_and_group(
            classnames        = data_examples["dtype"], 
            **pieChart_dct
        )
        plt.show();
        
    # table with basic info on missiing data in entire dataframe    
    if showTable==True:
        df_summary_table(df=data_examples, **table_dct)
        

    if barPlot==True:
        
        # .. barplot for each column with any non-missing data, eacg dtype is plotted separately,
        if groups_to_display=="all": 
            groups_to_display=["all"]; add_all_groups=True
        else: 
            add_all_groups=False
        
        
        for i, group_name in enumerate(groups_to_display): 
            
            # filter the data, and plot title, 
            if add_all_groups:
                df_filter         = pd.Series([True]*data_examples.shape[0])
            else:
                df_filter         = data_examples['dtype']==group_name

                
            # test, if the given group was present:  
            if df_filter.sum()==0:
                if verbose==True:
                    print("- - -THERE WERE NO COLUMNS WITH THAT DATA TYPE IN SEARCHED DataFrame - - -", end="\n\n")
                else:
                    pass
            else:
                # set size for the figure with barplot, 
                if df_filter.sum()>0 and df_filter.sum()<=10:
                    figSize = (12,5)
                    groupSize = df_filter.sum()

                elif df_filter.sum()>10:
                    figSize = (16,16)
                    groupSize = 5  
                 
                elif df_filter.sum()>50:
                    figSize = (12,22)
                    groupSize = 8  
                    
                # replace values in case 
                if barPlot_figsize!=None:
                    figSize= barPlot_figsize
                else:
                    pass
                if barPlot_groupSize!=None:
                    groupSize = barPlot_groupSize
                else:
                    pass
                
                    
                ##    
                annotated_barplot(
                    data_examples = data_examples, 
                    top_val_perc  = top_values_perc, 
                    df_filter     = df_filter, 
                    fig_size      = figSize,
                    group_size    = groupSize,
                    **barPlot_dct
                )

            # example
            #groups_to_display = ['text', 'numeric', 'datetime']
            # plot_summary_pie_chart_and_bar_plots(data_examples=data_examples, top_val_perc=top_val_perc, start_figure_numbers_at=4, pieChart=True, showTable=True, groups_to_display = ['text', 'numeric', 'datetime'])