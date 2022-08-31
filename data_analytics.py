import enum
from types import NoneType
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import s3fs


def get_xy_dataframe(x_value:list, y_value:list, experiment:list, title:list, xlabel:str):
    if len(experiment) > 1:
        st.warning("Data analytics cannot be performed when more than one experiments are selected")
        st.stop()

    exp = experiment[0]
    x_dict = {}
    y_dict = {}
    for x_group, y_group, title_group in zip(x_value, y_value, title):
        for scan_x, scan_y, title in zip(x_group[exp], y_group[exp], title_group[exp]):
            x_dict[title] = list(scan_x.values())
            y_dict[title] = list(scan_y.values())

    x_df = pd.DataFrame(x_dict)
    y_df = pd.DataFrame(y_dict)

    try:
        x_df = x_df[x_df.columns[0]]
    except IndexError:
        st.warning("No data for data analytics")
        st.stop()

    x_df.name = xlabel

    return x_df, y_df

def get_data_from_aws(type:str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    fs = s3fs.S3FileSystem(anon=False,client_kwargs={
                           'endpoint_url':'https://s3.ap-east-1.amazonaws.com'
                           })
    bucket = f'decode-cure-psdata/{type}'

    if f'{bucket}/Summary.csv' not in fs.ls(bucket, refresh=True):
        return pd.DataFrame, pd.DataFrame, 0 

    df = pd.read_csv(f's3://{bucket}/Summary.csv')
    latest_avg = df['Avg']
    latest_std = df['Std']
    total_num_of_scan = df['Scan Count'][0]

    
    return latest_avg, latest_std, total_num_of_scan
    
def detect_outlier(y_df:pd.DataFrame, avg_df:pd.DataFrame, std_df:pd.DataFrame, total_num_of_scan:int, ylabel:str) -> tuple[list, pd.DataFrame]:
    
    # calculate new avg and std if there are data in AWS S3
    if total_num_of_scan != 0:
        N = total_num_of_scan
        last_avg_df = avg_df
        variance_df = std_df**(2)
        for col in y_df:
            N += 1
            variance_df = variance_df.multiply(N-2)
            
            # curr_avg_df = Mean(Xn)
            curr_avg_df = last_avg_df.multiply(N-1).add(y_df[col]).divide(N)

            # product = (Xn - Mean(Xn)(Xn - Mean(Xn-1))
            product = y_df[col].sub(curr_avg_df)
            product = product.multiply(y_df[col].sub(last_avg_df))
            variance_df = variance_df.add(product).divide(N-1)

            # update for next iteration (last_avg_df = Mean(Xn-1))
            last_avg_df = curr_avg_df
    
        y_avg = curr_avg_df 
        y_std = variance_df**(1/2) 
        y_3std = y_std.multiply(3)
    # if there are no data in AWS S3, calculate the avg and std with the inputed data alone
    else:
        y_avg = y_df.mean(axis=1)
        y_std = y_df.std(axis=1)
        y_3std = y_std.multiply(3)

    y_ceil = y_avg.add(y_3std)
    y_floor = y_avg.sub(y_3std)
    # remove out lier
    y_df_no_outlier = y_df
    outlier = []
    for scan in y_df:
        for y, ceil, floor in zip(y_df[scan], list(y_ceil), list(y_floor)):
            if y > ceil or y < floor:
                y_df_no_outlier = y_df_no_outlier.drop(scan, axis=1)
                outlier.append(scan)
                break
                
    y_df_no_outlier_mean = y_df_no_outlier.mean(axis=1)
    y_df_no_outlier_mean.name = ylabel

    return outlier, y_df_no_outlier_mean, y_df_no_outlier

def get_avg_df(x_df:pd.DataFrame, y_df:pd.DataFrame) -> pd.DataFrame:
    line_group = pd.DataFrame({'Line Group': ['main'] * len(y_df)}) 
    dash = pd.DataFrame({'Dash': ['false'] * len(y_df)}) 
    df = pd.concat([x_df, y_df, line_group, dash], axis=1)
    return df

def store_to_aws(y_df:pd.DataFrame, avg_df:pd.DataFrame, std_df:pd.DataFrame, total_num_of_scan:int, type:str, file_name:str):
     
    # update Summary.csv
    # if no previous data, create Summary.csv from inputed data
    if total_num_of_scan == 0:
        y_avg = y_df.mean(axis=1)
        y_std = y_df.std(axis=1)   
        updated_num_of_scan = len(y_df.columns)
    # update Summary.csv with previous data
    else:
        N = total_num_of_scan
        last_avg_df = avg_df
        variance_df = std_df**(2)
        for col in y_df:
            N += 1
            variance_df = variance_df.multiply(N-2)
            
            # curr_avg_df = Mean(Xn)
            curr_avg_df = last_avg_df.multiply(N-1).add(y_df[col]).divide(N)

            # product = (Xn - Mean(Xn)(Xn - Mean(Xn-1))
            product = y_df[col].sub(curr_avg_df)
            product = product.multiply(y_df[col].sub(last_avg_df))
            variance_df = variance_df.add(product).divide(N-1)

            # update for next iteration (last_avg_df = Mean(Xn-1))
            last_avg_df = curr_avg_df
        
        y_avg = curr_avg_df 
        y_std = variance_df**(1/2) 
        updated_num_of_scan = N

    y_avg.name = 'Avg'
    y_std.name = 'Std'
    num_of_scan_df = pd.DataFrame({'Scan Count':[updated_num_of_scan]})

    fs = s3fs.S3FileSystem(anon=False, client_kwargs={
                           'endpoint_url':'https://s3.ap-east-1.amazonaws.com'
                           })
    # store Summary.csv
    updated_summary = pd.concat([y_avg, y_std, num_of_scan_df], axis=1)
    with fs.open(f'decode-cure-psdata/{type}/Summary.csv', 'w') as f:
        updated_summary.to_csv(f, index=False)

    # Store input data without out liers
    # if there are file with then same name in S3, generate new file name 
    file_name = file_name.split('.pssession', 1)[0]
    output_path = f'decode-cure-psdata/{type}/{file_name}.csv'
    exist_file_list = fs.ls(f'decode-cure-psdata/{type}/', refresh=True)
    i=1 
    while output_path in exist_file_list:
        output_path = f'decode-cure-psdata/{type}/{file_name} ({i}).csv'
        i += 1
    
    with fs.open(output_path, 'w') as f:
        y_df.to_csv(f, index=False) 

def get_two_half(avg_df:pd.DataFrame, xlabel:str, ylabel:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # divide the data into two half
    start = avg_df[xlabel].idxmin()
    end = avg_df[xlabel].idxmax()
    change_pt = max(start, end) + 1

    # determine which is top half and which is second half
    first_half = avg_df.iloc[:change_pt]
    first_half_mean = first_half[ylabel].mean(0)
    second_half = avg_df.iloc[change_pt:]
    second_half_mean = second_half[ylabel].mean(0)

    top_half = first_half if first_half_mean > second_half_mean else second_half
    bottom_half = second_half if first_half_mean > second_half_mean else first_half 

    top_half = top_half.reset_index(drop=True)
    bottom_half = bottom_half.reset_index(drop=True)

    return top_half, bottom_half

def polynomial_regression(df:pd.DataFrame) -> tuple[np.float64, np.array, np.array]:

    regression_x = np.array(df[df.columns[0]].tolist()).reshape(-1,1)
    regression_y = np.array(df[df.columns[1]].tolist()).reshape(-1,1)

    r_sq_max = 0
    i = 2 
    while True:
        regression_x_ = PolynomialFeatures(degree=i, include_bias=False).fit_transform(regression_x)  
        model_i = LinearRegression().fit(regression_x_, regression_y)
        r_sq = model_i.score(regression_x_, regression_y)

        # end loop if degree = i result in lower r^2
        if r_sq <= r_sq_max:
            break

        r_sq_max = r_sq
        i += 1

    # use max degreee, which is i - 1
    regression_x_ = PolynomialFeatures(degree=i-1, include_bias=False).fit_transform(regression_x) 
    model_i = LinearRegression().fit(regression_x_, regression_y)
    y_pred = model_i.predict(regression_x_)

    return model_i.intercept_[0], model_i.coef_[0], y_pred

def get_formula(coefs:np.array, coef_x0:np.float64) -> tuple[str,str]:
    formula = f'{np.format_float_positional(coef_x0, precision=2)} '
    pain_text_formula = f'{np.format_float_positional(coef_x0, precision=2)}' 
    for i, num in enumerate(coefs):
        formula += str(np.format_float_positional(num, precision=2, sign=True))
        formula += 'x^{' + str(i+1) + '} '
        if (i+1)%4 == 0:
            formula += '\\\\'
        pain_text_formula += str(np.format_float_positional(num, precision=2, sign=True))
        pain_text_formula += f'x^{i+1}'
    return formula, pain_text_formula

def d2A_dV2(coefs:np.array) -> list:
    dA_dV = [float(coef*(i+1)) for i, coef in enumerate(coefs)]
    result = [float(coef*(i+1)) for i, coef in enumerate(dA_dV)]    
    return result[1:]

def sub_x(coefs:list, x:float) -> float:
    concave = 0
    for j, coef in enumerate(coefs):
        curvature_change = coef * pow(x, j)
        concave += curvature_change
    return concave 
    
def get_peak(df, half, coef, d2_limit, r_sq_limit, position):

    xlabel = df.columns[0]
    ylabel = df.columns[1]

    idx = half[ylabel].idxmax() if position == 'top' else half[ylabel].idxmin()
    E_pa = half[ylabel][idx]
    E_pa_x = half[xlabel][idx]

    d2_coef = d2A_dV2(coef)

    regressionX = np.array([])
    regressionY = np.array([])
    for x, y in zip(half[xlabel].iloc[:idx], half[ylabel].iloc[:idx]):
        regressionX = np.append(regressionX, x)
        regressionY = np.append(regressionY, y)

        if abs(sub_x(d2_coef, x)) > d2_limit:
            regressionX = np.array([])
            regressionY = np.array([])
            continue
            
        if (len(regressionX) == 1 or len(regressionX) == 2):
            continue

        fit_regressionX = regressionX.reshape(-1,1)
        fit_regressionY = regressionY.reshape(-1,1)
        model = LinearRegression().fit(fit_regressionX, fit_regressionY)
        r_sq = model.score(fit_regressionX, fit_regressionY)
        if r_sq < r_sq_limit:
            regressionX = np.delete(regressionX, -1)
            regressionY = np.delete(regressionY, -1)
            break
    
    fit_regressionX = regressionX.reshape(-1,1)
    fit_regressionY = regressionY.reshape(-1,1)
    model = LinearRegression().fit(fit_regressionX, fit_regressionY)
    y_pred = model.predict(fit_regressionX)
    c = model.intercept_[0]
    m = model.coef_[0][0]
    first_linear_reg_end = m * regressionX[-1] + c
    intercept_pt = m*E_pa_x+c

    label = 'i_pa' if position == 'top' else 'i_pc'
    new_rows = pd.DataFrame({xlabel:regressionX, ylabel: y_pred.flatten(), 'Line Group': [label] * len(regressionX), 'Dash': ['false'] * len(regressionX)})
    df = pd.concat([df,new_rows])
    new_rows = pd.DataFrame({xlabel:[regressionX[-1], E_pa_x], ylabel:[first_linear_reg_end, intercept_pt], 'Line Group':[f'{label} extend']*2, 'Dash': ['true','true']})
    df = pd.concat([df,new_rows])

    return df, E_pa, E_pa_x,intercept_pt 



    
