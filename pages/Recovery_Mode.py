import streamlit as st
import pandas as pd
import plotly.express as px
import s3fs


def get_data(excels:list, xlabel:str, ylabel:str) -> pd.DataFrame:
    dict = {'Scan':[], 'Group':[], xlabel:[], ylabel:[]}
    group_count = set([])
    for excel in excels: 
        sheets = pd.read_excel(excel, sheet_name=None) 
        for sheet in sheets:
            group_count.add(sheet)
            df = pd.read_excel(excel, sheet_name=sheet)
            for col in df:
                if col == 'x':
                    continue
                dict['Scan'] += [col] * len(df['x'])
                dict['Group'] += [sheet] * len(df['x'])
                dict[xlabel] += list(df['x'])
                dict[ylabel] += list(df[col])
    ret = pd.DataFrame(dict)

    return ret, len(group_count) 

def plot_graph(df, color):
    config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'custom_image',
            'height': 500,
            'width': 700,
            'scale':5 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }
    for i, c in enumerate(color):
        color[i] = px.colors.qualitative.Plotly[i] if c == '#000000' else c

    fig = px.line(df, y=ylabel, x=xlabel, color='Group', line_group='Scan', color_discrete_sequence=color)
    st.plotly_chart(fig, config=config)

def update_summary_csv():
    exp_type = st.selectbox('Select Experiment Type For Data Storage', ['CV1', 'CV2', 'CV3', 'AD1'])
    if (st.button(f'Updata Summary.csv in {exp_type}')):
        fs = s3fs.S3FileSystem(anon=False, client_kwargs={
                            'endpoint_url':'https://s3.ap-east-1.amazonaws.com'
                            })
        bucket = f'decode-cure-psdata/{exp_type}'
        file_list = fs.ls(bucket)

        # combine all data into one df to calculate mean and std 
        df_list = []
        for file in file_list:
            if file == f'{bucket}/' or file == f'{bucket}/Summary.csv':
                continue
            df = pd.read_csv(f's3://{file}')
            df_list.append(df)
        
        combined_df = pd.concat(df_list, axis=1)

        avg_df = combined_df.mean(axis=1)
        avg_df.name = 'Avg'
        std_df = combined_df.std(axis=1)
        std_df.name = 'Std'
        scan_count_df = pd.DataFrame({'Scan Count':[len(combined_df.columns)]})

        summary = pd.concat([avg_df, std_df, scan_count_df], axis=1)
        with fs.open(f'{bucket}/Summary.csv', 'w') as f:
            summary.to_csv(f, index=False)

        st.success(f'Summary.csv in {exp_type} is updated')

### Page Start ####################
st.title('Recovery Mode')
st.markdown("""---""")
st.subheader('Update Summary.csv in AWS')
update_summary_csv()

st.markdown("""---""")
excel = st.file_uploader('Input excel files to recover graph', accept_multiple_files=True)
if len(excel) == 0:
    st.stop()

with st.sidebar:
    col1, col2 = st.columns([1,1])
    with col1:
        xlabel = st.text_input('X Axis Title', 'Voltage (V)')
    with col2:
        ylabel = st.text_input('Y Axis Title', 'Current (uA)')

data, group_no = get_data(excel, xlabel, ylabel)

color = [''] * group_no 
with st.sidebar:
    tab = st.tabs([f'Group {i+1}' for i in range(group_no)])
    for i in range(group_no):
        with tab[i]:
            color[i] = st.color_picker('Custom Color', key=i, help='Select #000000 (bottom left hand corner) to reset the color')

plot_graph(data, color)
 
