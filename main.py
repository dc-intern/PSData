import enum
from lib2to3.pgen2.pgen import DFAState
from logging import NullHandler
import streamlit as st
import numpy as np
import pandas as pd
import xlsxwriter
import io
import plotly.express as px

import PSData as ps
import data_analytics

# get x_axis and y_axis value from the scan name string
def get_exp_data(experiment: str):
    x_data = input.data[f'{experiment} xdata']
    y_data = input.data[f'{experiment} ydata']
    titles = input.data[f'{experiment} titles']

    # a hash table for converting scan title to int 
    titles_index = {}
    for i, title in enumerate(titles.values()):
        titles[i] = title.split(' [', 1)[0]
        titles_index[titles[i]] = i

    # find the number of channnels
    no_channels = set([])
    for title in titles.values():
        if 'Channel' in title:
            no_channels.add(int(title.split('Channel ', 1)[1][0]) )
    channel_list = [f'Channel {i}' for i in no_channels]


    # create a scan list that store scan title without channel number
    scan_list = {}
    for title in titles.values():
        if 'Scan' in title:
            current_scan = int(title.split('Scan ', 1)[1][0])
            if f'Scan {current_scan}' not in scan_list:
                scan_list[f'Scan {current_scan}'] = [title]
            else:
                scan_list[f'Scan {current_scan}'].append(title)
        else:
            if 'Other' not in scan_list:
                scan_list['Other'] = [title]
            else:
                scan_list['Other'].append(title)
    
    return x_data, y_data, titles, titles_index, channel_list, scan_list

# get data from all experiments
def get_data(experiments: list):
    x_data = {}
    y_data = {}
    titles = {}
    title_index = {}
    channel_list = {}
    scan_list = {}
    for experiment in experiments:
        x_data[experiment], y_data[experiment], titles[experiment], title_index[experiment], channel_list[experiment], scan_list[experiment] = get_exp_data(experiment)

    return x_data, y_data, titles, title_index, channel_list, scan_list

# create input element for user to select which scan to put on the graph
def select_data(group: int, experiment: str, titles: list, channel_list: list, scan_list: dict) -> list:
    # generate a select all button
    col1, col2 = st.columns([1,1])
    with col1:
        select_all = st.button('Select All', key=f'{group}{experiment}')
    if select_all:
        with col2:  
            if (st.button('Drop All', key=f'{group}{experiment}drop')):
                select_all = False

    if select_all:
        selected_scan = titles.values()
    else:
        selected_scan = []
        titles_copy = list(titles.values()) 

        # select channel 
        selected_channels = []
        with st.expander(f'Select all scan in channel'):
            for channel in channel_list:
                if (st.checkbox(channel, key=f'{channel}{experiment}{group}')):
                    selected_channels.append(channel)
                
            # covernt selected channel to scans
            for channel in selected_channels:
                for i, title in enumerate(titles.values()):
                    if channel in title:
                        selected_scan.append(title)
                        titles_copy.remove(title)
                    elif (i > 0):
                        if (not('Channel' in title) and channel in titles[i-1]):
                            selected_scan.append(title)
                            titles_copy.remove(title)

        # select scan from all channel
        with st.expander(f'Select scan from all channel'):
            for scans in scan_list:
                if st.checkbox(scans, key=f'{scans}{experiment}{group}'):
                    selected_scan += scan_list[scans] 
                    for scan in scan_list[scans]:
                        if scan in titles_copy:
                            titles_copy.remove(scan)

        # select scan
        with st.expander(f'Select scan'):
            for i, scan in enumerate(titles_copy):
                if (st.checkbox(scan, key=f'{scan}{i}{experiment}{group}')):
                    selected_scan.append(scan)

    return selected_scan 

def handle_split(scan_in_group: list, experiments: list):
    max_size = 0
    for exp in experiments:
        max_size = max(max_size, len(scan_in_group[exp]))
    split_list = []
    for i in range(max_size):
        split_list.append({exp: [] for exp in experiments}) 

     
    for exp in experiments: 
        for i, scan in enumerate(scan_in_group[exp]):
            split_list[i][exp].append(scan)

    return split_list


# get x,y axis value from the selected scan title
def get_axis(scan_in_group: list, title_index: dict, x_data: dict, y_data: dict, experiments: list):
    # convert scan title to titel_index
    name_to_index = []
    for i in range(len(scan_in_group)):
        name_to_index.append({})
        for exp in experiments:
            name_to_index[i][exp] = [] 
            for scan in scan_in_group[i][exp]:
                name_to_index[i][exp].append(title_index[exp][scan]) 

    # collect x,y axis values
    x_values = [] 
    y_values = []
    for i in range(len(name_to_index)):
        x_values.append({})
        y_values.append({})
        for exp in experiments:
            x_values[i][exp] = []
            y_values[i][exp] = []
            for scan in name_to_index[i][exp]:
                x_values[i][exp].append(x_data[exp][scan])
                y_values[i][exp].append(y_data[exp][scan])
    
    
    return x_values, y_values

# plot graph
def plot_graph(x_values: list, y_values: list, 
        titles: list, experiments: list, xlabel: str, 
        ylabel: str, color: list, group_name: list, outlier = []
) -> dict:

    dict = {'Scan':[], 'Group':[], xlabel:[], ylabel:[]}
     
    for x_group, y_group, gp_name, title_group in zip(x_values, y_values, group_name, titles):
        for exp in experiments:
            for x_scan, y_scan, title in zip(x_group[exp], y_group[exp], title_group[exp]):   
                if title in outlier:
                    continue
                for x, y in zip(x_scan.values(), y_scan.values()):
                    dict['Group'].append(gp_name)
                    dict['Scan'].append(f'{exp} {title}') 
                    dict[xlabel].append(x)
                    dict[ylabel].append(y)

    df = pd.DataFrame(dict)
    # st.table(df)
    # config for download graph as png
    config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'custom_image',
            'height': 500,
            'width': 700,
            'scale':5 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }

    # Apply default color if there no custom color input
    for i, c in enumerate(color):
        color[i] = px.colors.qualitative.Plotly[i] if c == '#000000' else c
    
    # Create and ploy graph
    fig = px.line(df, y=ylabel, x=xlabel, color='Group', line_group='Scan', color_discrete_sequence=color)
    st.plotly_chart(fig, config=config)

    return dict

# genaerate a excel file with x_axis and y_axis data and a download button to download the file
def axis_data_download(x_values: list, y_values: list, name: list, experiment: str, group_names: list):

    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    for i in range(len(y_values)):
        df = pd.DataFrame()
        if (len(x_values[i][experiment])) == 0: continue 
        df['x'] = x_values[i][experiment][0].values()
        for j in range(len(y_values[i][experiment])):
            df[name[i][experiment][j]] = y_values[i][experiment][j].values()
        df.to_excel(writer, sheet_name=group_names[i], index=False)
    
    writer.save()
    st.download_button(
        label=f"Download {experiment} Value",
        data=output.getvalue(),
        file_name=f"{experiment}_value.xlsx",
        mime="application/vnd.ms-excel"
    )

# App Start ######################
st.title('PSData')
# st.markdown("""---""")
pssession = st.file_uploader('.pssession')

if not pssession:
    st.stop()

input = ps.jparse(pssession)

with st.sidebar:
    # get x_data, y_data and titles of scan in .pssession
    experiments = st.multiselect('Experiment', input.experimentList)
    if (len(experiments) == 0):
        st.stop()
    x_default = 'Time (secs)' if 'AD' in experiments[0] else 'Voltage (V)' 
    col1, col2 = st.columns([1,1])
    with col1:
        xlabel = st.text_input('X Axis Title', x_default)
    with col2:
        ylabel = st.text_input('Y Axis Title', 'Current (uA)')

    x_data, y_data, titles, title_index, channel_list, scan_list = get_data(experiments)

    # ask for the number of groups 
    num_of_group = st.slider('Number of Groups', 1, 10)

    scan_in_group = []
    group_names = []
    color = []

    # generate two level of tabs for seleting group and experiments
    tab = st.tabs([f'Group {i+1}' for i in range(num_of_group)])
    for i in range(num_of_group):
        scan_in_group.append({})
        with tab[i]:
            inner_taps = st.tabs(experiments)
            for j, exp in enumerate(experiments):
                scan_in_group[i][exp] = []
                with inner_taps[j]:
                    scan_in_group[i][exp] += select_data(i, exp, titles[exp], channel_list[exp], scan_list[exp])

            split = st.checkbox('Split', value=False, key=f's{i}')
            if split:
                scan_in_group = handle_split(scan_in_group[i], experiments)
                group_names = []
                color = []
                split_taps = st.tabs([f'Group {j+1}' for j, s in enumerate(scan_in_group)])
                for j, s in enumerate(scan_in_group):
                    with split_taps[j]:
                        col1, col2 = st.columns([1,1.5]) 
                        with col1:
                            color.append(st.color_picker('Custom Color', key=f'sc{j}', help='Select #000000 (bottom left hand corner) to reset the color'))        
                        with col2:
                            gp_name = st.text_input('Group Name', value = f'Group {j+1}', key=f'sgp{j}')
                            group_names.append(gp_name)
                break
                
            col1, col2 = st.columns([1,1.5]) 
            with col1:
                color.append(st.color_picker('Custom Color', key=i, help='Select #000000 (bottom left hand corner) to reset the color'))
            with col2:
                gp_name = st.text_input('Group Name', value = f'Group {i+1}', key=f'gpname{i}')
                group_names.append(gp_name)

x_values, y_values = get_axis(scan_in_group, title_index, x_data, y_data, experiments)
graph_data = plot_graph(x_values, y_values, scan_in_group, experiments, xlabel, ylabel, color, group_names)
for experiment in experiments:
    axis_data_download(x_values, y_values, scan_in_group, experiment, group_names)


# Data Analytics
st.title('Data Analytics')

exp_type = st.selectbox('Select Experiment Type For Data Storage', [' ', 'CV1', 'CV2', 'CV3', 'AD1'])

x_df, y_df = data_analytics.get_xy_dataframe(x_values, y_values, experiments, scan_in_group, xlabel)

# get previous data from aws
old_avg_df, old_std_df, old_numof_scan = data_analytics.get_data_from_aws(exp_type)

# detect outlier with pass data
outlier, y_df_no_outlier_mean, y_df_no_outlier = data_analytics.detect_outlier(y_df, old_avg_df, old_std_df, old_numof_scan, ylabel)

# store to aws
if exp_type != ' ':
    data_analytics.store_to_aws(y_df_no_outlier, old_avg_df, old_std_df, old_numof_scan, exp_type, pssession.name)

avg_df = data_analytics.get_avg_df(x_df, y_df_no_outlier_mean)
top_half, bottom_half = data_analytics.get_two_half(avg_df, xlabel, ylabel)
tab1, tab2, tab3 = st.tabs(['Outlier Detection', 'Polynomial Regression',  'Peak Calucation'])

#display outlier
with tab1:
    with st.expander('Outliers List'):
        st.table(pd.Series(outlier, name='Outlier', dtype=pd.StringDtype()).to_frame())
    st.subheader('Graph without Outliers')
    graph_data = plot_graph(x_values, y_values, scan_in_group, experiments, xlabel, ylabel, color, group_names, outlier)
    
# Polynomial Regression
with tab2:
    top_intercept, top_coef, top_pred = data_analytics.polynomial_regression(top_half)
    bottom_intercept, bottom_coef, bottom_pred = data_analytics.polynomial_regression(bottom_half)
    top_formula, top_text_formula = data_analytics.get_formula(top_coef, top_intercept)
    bottom_formula, bottom_text_formula = data_analytics.get_formula(bottom_coef, bottom_intercept)
    st.write('Top Half Formula:')
    st.latex(f'y = {top_formula}')
    with st.expander("Pain Text Formula:"):
        st.write(f'y = {top_text_formula}')
    st.write('Bottom Half Formula:')
    st.latex(f'y = {bottom_formula}')
    with st.expander("Pain Text Formula:"):
        st.write(f'y = {bottom_text_formula}')

# Peak Calucation
with tab3:
    top_r_sq_limit = st.number_input('Limit of top half r^2', value=0.999, format='%.3f')
    bottom_r_sq_limit = st.number_input('Limit of bottom half r^2', value=0.999, format='%.3f')

    if not st.button('Calucate Peak'):
        st.stop() 
    max_length = 0
    
    # test every all value from 0 to 5000 for limit of d2A/d2(f(x))
    for i in range(0, 5000, 5):
        try:
            df, e_pa, e_pa_x, intercept_pt = data_analytics.get_peak(avg_df, top_half, top_coef, i, top_r_sq_limit, 'top') 
        except ValueError:
            continue
        if len(df.index) > max_length:
            top_df = df
            E_pa = e_pa
            E_pa_x = e_pa_x
            top_intercept_pt = intercept_pt
            max_length = len(df.index)
            top_d2_limit = i

    for i in range(0, 5000, 5):
        try:
            df, e_pc, e_pc_x, intercept_pt = data_analytics.get_peak(top_df, bottom_half, bottom_coef, i, bottom_r_sq_limit, 'bottom') 
        except ValueError:
            continue
        if len(df.index) > max_length:
            result_df = df
            E_pc = e_pc
            E_pc_x = e_pc_x
            bottom_intercept_pt = intercept_pt
            max_length = len(result_df.index)
            bottom_d2_limit = i

    fig = px.line(result_df, x=xlabel, y=ylabel, color='Line Group', line_dash='Dash')
    fig.add_shape(type='line', x0=E_pa_x, y0=top_intercept_pt, x1=E_pa_x, y1=E_pa, line_dash='dash')
    fig.add_shape(type='line', x0=E_pc_x, y0=bottom_intercept_pt, x1=E_pc_x, y1=E_pc, line_dash='dash')

    # round the result to 2 decimals pt for display
    E_pa = round(E_pa, 2) 
    I_pa = round(E_pa - top_intercept_pt, 2)
    E_pc = round(E_pc, 2)
    I_pc = round(bottom_intercept_pt - E_pc, 2)

    # display the result with number on the graph
    fig.add_annotation(text=f'E PA = {E_pa}', x=E_pa_x, y=E_pa*1.05, showarrow=False)
    fig.add_annotation(text=f'I PA = {I_pa}', x=E_pa_x+0.07, y=(E_pa+top_intercept_pt)/2 , showarrow=False)
    fig.add_annotation(text=f'E PC = {E_pc}', x=E_pc_x, y=E_pc*1.05, showarrow=False)
    fig.add_annotation(text=f'I PC = {I_pc}', x=E_pc_x+0.07, y=(E_pc+bottom_intercept_pt)/2, showarrow=False)

    # plot graph
    config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'custom_image',
            'height': 500,
            'width': 700,
            'scale':5 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, config=config)

    st.write(f'E PA = {E_pa}')
    st.write(f'I PA = {I_pa}')
    st.write(f'E PC = {E_pc}')
    st.write(f'I PC = {I_pc}')
    st.write(f'Optimum limit for top half d2a/dv2: {top_d2_limit}')
    st.write(f'Optimum limit for bottom half d2a/dv2: {bottom_d2_limit}')
    

