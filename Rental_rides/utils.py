import numpy as np
import pandas as pd
import time
import ast
import matplotlib.pyplot as plt
import datetime
import pyodbc
import os
import scipy.stats as st
plt.style.use('fivethirtyeight')

def ttest_2(a,b):
    return st.ttest_ind(a,b, equal_var=False)

def read_from_redshift(sql_query):
    """
        Reads data from redshift
    """
    conn = pyodbc.connect('DSN=Redshift_live')
    
    print("query started at: {0}".format(datetime.datetime.now().time()))
    df = pd.read_sql(sql_query, conn)
    print("query ended at: {0}".format(datetime.datetime.now().time()))
    
    conn.close()
    return df


def get_cohorts(conf, data_path, overwrite_file = False):
    """
        Loads rider cohorts as of the Monday of the start_date week
    """
    sql_cohorts = open('get_cohorts.sql', 'r').read()
    filename = 'de_cohorts_{}_{}.csv{}'.format(conf['city'], conf['de_discount_start_date'], '.gz')
    if filename not in os.listdir(data_path) or overwrite_file:
        print("Querying cohort data from Redshift...")
        df = read_from_redshift(sql_cohorts.format(
            country_code=conf['country'],
            start_date=conf['de_discount_start_date'],
            city_name=conf['city']))
        df.to_csv('{}/{}'.format(data_path, filename), compression='gzip', index=False, header=True)
    else:
        print("Reading cohort data from csv...")
        df = pd.read_csv('{}/{}'.format(data_path, filename), compression='gzip')
    return df

def get_cohorts_etl(conf, data_path, overwrite_file = False):
    """
        Loads rider cohorts from ETL as of the Monday of the start_date week
    """
    sql = open('get_cohorts_etl.sql', 'r').read()
    filename = 'de_cohorts_etl_{}_{}.csv{}'.format(conf['city'], conf['de_discount_start_date'], '.gz')
    if filename not in os.listdir(data_path) or overwrite_file:
        print("Querying cohort ETL data from Redshift...")
        df = read_from_redshift(sql.format(
            country_code=conf['country'],
            start_date=conf['de_discount_start_date'],
            city_name=conf['city']))
        df.to_csv('{}/{}'.format(data_path, filename), compression='gzip', index=False, header=True)
    else:
        print("Reading cohort ETL data from csv...")
        df = pd.read_csv('{}/{}'.format(data_path, filename), compression='gzip')
    return df

def get_cohorts_rfmv1(conf, data_path, overwrite_file = False):
    """
        Loads rider RFMv1 cohorts as of the Monday of the start_date week
    """
    sql_cohorts = open('get_cohorts_rfmv1.sql', 'r').read()
    filename = 'de_cohorts_rfmv1_{}_{}.csv{}'.format(conf['city'], conf['de_discount_start_date'], '.gz')
    if filename not in os.listdir(data_path) or overwrite_file:
        print("Querying cohort rfmv1 data from Redshift...")
        df = read_from_redshift(sql_cohorts.format(
            country_code=conf['country'],
            start_date=conf['de_discount_start_date'],
            city_name=conf['city']))
        df.to_csv('{}/{}'.format(data_path, filename), compression='gzip', index=False, header=True)
    else:
        print("Reading cohort rfmv1 data from csv...")
        df = pd.read_csv('{}/{}'.format(data_path, filename), compression='gzip')
    return df

def get_cohorts_boda(conf, data_path, overwrite_file = False):
    """
        Loads rider RFMv2 cohorts for boda rides as of the Monday of the start_date week
    """
    sql_cohorts = open('get_cohorts_boda.sql', 'r').read()
    filename = 'de_cohorts_boda_{}_{}.csv{}'.format(conf['city'], conf['de_discount_start_date'], '.gz')
    if filename not in os.listdir(data_path) or overwrite_file:
        print("Querying cohort boda data from Redshift...")
        df = read_from_redshift(sql_cohorts.format(
            country_code=conf['country'],
            start_date=conf['de_discount_start_date'],
            city_name=conf['city']))
        df.to_csv('{}/{}'.format(data_path, filename), compression='gzip', index=False, header=True)
    else:
        print("Reading cohort boda data from csv...")
        df = pd.read_csv('{}/{}'.format(data_path, filename), compression='gzip')
    return df

def get_cohorts_sessions(conf, data_path, overwrite_file = False):
    """
        Loads rider RFMv2 cohorts from sessions as of the Monday of the start_date week
    """
    sql_cohorts = open('get_cohorts_sessions.sql', 'r').read()
    filename = 'de_cohorts_sessions_{}_{}.csv{}'.format(conf['city'], conf['de_discount_start_date'], '.gz')
    if filename not in os.listdir(data_path) or overwrite_file:
        print("Querying cohort sessions data from Redshift...")
        df = read_from_redshift(sql_cohorts.format(
            country_code=conf['country'],
            start_date=conf['de_discount_start_date'],
            city_name=conf['city']))
        df.to_csv('{}/{}'.format(data_path, filename), compression='gzip', index=False, header=True)
    else:
        print("Reading cohort sessions data from csv...")
        df = pd.read_csv('{}/{}'.format(data_path, filename), compression='gzip')
    return df

def get_lifetime_activity(conf, data_path, overwrite_file = False):
    """
        Loads rider lifetime activity from the day before campaign start
    """
    sql_cohorts = open('get_lifetime_activity.sql', 'r').read()
    filename = 'de_lifetime_activity_{}_{}.csv{}'.format(conf['city'], conf['de_discount_start_date'], '.gz')
    if filename not in os.listdir(data_path) or overwrite_file:
        print("Querying lifetime activity data from Redshift...")
        df = read_from_redshift(sql_cohorts.format(
            country_code=conf['country'],
            start_date=conf['de_discount_start_date'],
            city_name=conf['city']))
        df.to_csv('{}/{}'.format(data_path, filename), compression='gzip', index=False, header=True)
    else:
        print("Reading lifetime activity data from csv...")
        df = pd.read_csv('{}/{}'.format(data_path, filename), compression='gzip')
    return df


def get_alive_probability(conf, data_path, overwrite_file = False):
    """
        Loads alive probability from previous day
    """
    sql = open('get_alive_probability.sql', 'r').read()
    filename = 'de_alive_probability_{}_{}.csv{}'.format(conf['city'], conf['de_discount_start_date'], '.gz')
    if filename not in os.listdir(data_path) or overwrite_file:
        print("Querying alive probability data from Redshift...")
        df = read_from_redshift(sql.format(
            country_code=conf['country'],
            start_date=conf['de_discount_start_date'],
            city_name=conf['city']))
        df.to_csv('{}/{}'.format(data_path, filename), compression='gzip', index=False, header=True)
    else:
        print("Reading alive probability data from csv...")
        df = pd.read_csv('{}/{}'.format(data_path, filename), compression='gzip')
    return df


def get_cherrypickers(conf, data_path, overwrite_file = False):
    """
        Loads cherrypickers' status from based on the activity before dex start
    """
    sql = open('get_cherrypickers.sql', 'r').read()
    filename = 'de_cherrypickers_{}_{}.csv{}'.format(conf['city'], conf['de_discount_start_date'], '.gz')
    if filename not in os.listdir(data_path) or overwrite_file:
        print("Querying cherrypickers data from Redshift...")
        df = read_from_redshift(sql.format(
            start_date=conf['de_discount_start_date'],
            city_name=conf['city']))
        df.to_csv('{}/{}'.format(data_path, filename), compression='gzip', index=False, header=True)
    else:
        print("Reading cherrypickers data from csv...")
        df = pd.read_csv('{}/{}'.format(data_path, filename), compression='gzip')
    return df

def get_profit(conf, data_path, overwrite_file = False):
    """
        Loads user profit from it's lifetime
    """
    sql = open('get_profit.sql', 'r').read()
    filename = 'de_profit_{}_{}.csv{}'.format(conf['city'], conf['de_discount_start_date'], '.gz')
    if filename not in os.listdir(data_path) or overwrite_file:
        print("Querying profit data from Redshift...")
        df = read_from_redshift(sql.format(
            start_date=conf['de_discount_start_date'],
            city_name=conf['city']))
        df.to_csv('{}/{}'.format(data_path, filename), compression='gzip', index=False, header=True)
    else:
        print("Reading profit data from csv...")
        df = pd.read_csv('{}/{}'.format(data_path, filename), compression='gzip')
    return df

def get_enrollments(conf, data_path, overwrite_file = False):
    """
        Loads test enrollments data for riders
    """
    sql_enrollments = open('get_enrollments.sql', 'r').read()
    filename = 'de_enrollments_{}_{}.csv{}'.format(conf['city'], conf['de_discount_start_date'], '.gz')
    if filename not in os.listdir(data_path) or overwrite_file:
        print("Querying enrollments data from Redshift...")

        df = read_from_redshift(sql_enrollments.format(
            city_test_ids=list_to_string(conf['city_test_ids']),
            city_treatment_share=conf['city_treatment_share'],
            additional_sql_enrollment_selection=conf['additional_sql_enrollment_selection']))

        # Check for holdout targeting
        if conf['use_holdout_targeting'] == 1:
            df = df[df['in_holdout'] == 1]

        # Convert discount size -1 to 0 for control group
        df["discount_percentage"] = np.where(df["discount_percentage"]==-1, 0, df["discount_percentage"])

        df.to_csv('{}/{}'.format(data_path, filename), compression='gzip', index=False, header=True)
    else:
        print("Reading enrollments data from csv...")
        df = pd.read_csv('{}/{}'.format(data_path, filename), compression='gzip')
    return df


def get_activity(conf, data_path, overwrite_file = False):
    """
        Loads rider activity data in the discount exploration
    """
    sql_activity = open('get_activity.sql', 'r').read()
    filename = 'de_activity_{}_{}.csv{}'.format(conf['city'], conf['de_discount_start_date'], '.gz')
    if filename not in os.listdir(data_path) or overwrite_file:
        print("Querying activity data from Redshift...")
        df = read_from_redshift(sql_activity.format(
            city=conf['city'],
            discount_start_date=conf['de_discount_start_date'], 
            discount_end_date=conf['de_discount_end_date'], 
            holdout_start_date=conf['de_holdout_start_date'], 
            holdout_end_date=conf['de_holdout_end_date']))
        df.to_csv('{}/{}'.format(data_path, filename), compression='gzip', index=False, header=True)
    else:
        print("Reading activity data from csv...")
        df = pd.read_csv('{}/{}'.format(data_path, filename), compression='gzip')
    return df


def get_city_dfi(df, 
                 test_ids = [6102], 
                 control_treatment_ids = [11098], 
                 n_treatments = 8, # how many different discounts were given
                 print_overview = True,
                 random_sample = False):
    """
        Creates dataframe consisting of all users in the experiment (treatment and control) 
        and adds one treatment group size worth of randomly sampled users from control group
    """    
    # Select users from treatment and control groups
    dfi = df[df["test_id"].isin(test_ids)]
    
    # Name treatment group
    dfi["treatment_type"] = dfi.apply(lambda x: 'control' if x.treatment_id in control_treatment_ids else 'treatment', axis=1)
    dfi["discount_percentage"] = dfi["discount_percentage"].fillna(-1)
    
    # As some combinations have 0% discount for treatment groups, sample 1 treatment group worth of users from control
    # Sample without replacement -> assuming no treatment is bigger than control group
    # do only in case control had different share than treatments
    if dfi[(dfi["treatment_id"].isin(control_treatment_ids))].distribution.unique()[0] != dfi[(~dfi["treatment_id"].isin(control_treatment_ids))].distribution.unique()[0]:
        treatment_size = int(len(dfi[~dfi.treatment_id.isin(control_treatment_ids)]) / n_treatments)
        tmp = df[(df["treatment_id"].isin(control_treatment_ids))].sample(n=treatment_size, replace=True, random_state=1)
    else:
        tmp = df[(df["treatment_id"].isin(control_treatment_ids))]
    
    # Fill columns for sampled control group 
    tmp["discount_percentage"] = tmp["discount_percentage"].fillna(-1)
    tmp["discount_percentage"] = np.where(tmp["discount_percentage"]==-1.0,0,tmp["discount_percentage"])
    tmp["treatment_type"] = 'treatment'
    
    # Merge sampled users with treatment and control group users
    dfi = pd.concat([dfi,tmp]).reset_index(drop=True)

    # Add column
    dfi["discount"] = dfi["discount_percentage"]
        
    if random_sample:
        # Sample dataset from population
        dfi = dfi.sample(len(dfi), replace=True)
    
    print(f"Dataset size: {len(dfi)}")

    # Overview of enrolled users
    if print_overview:
        fields = ["treatment_type","test_id","treatment_id","distribution","campaign_id","discount_percentage"]
        simulation_data_overview = dfi.fillna(-1).groupby(fields)["user_id"].count()
        print(simulation_data_overview)

    return dfi




def aggregate_metrics(df, conf, calc_cohorts = [["cohort","rfmv2","rfm"]], calc_efficiency = False, calc_p = False):
    # Calculate aggregated activity metrics
    m = pd.DataFrame()
    
    # Define cohorts to use
    # [field name in dfi, treatment name in m, rfm for numbered cluster]
    #calc_cohorts = [["cohort","rfmv1","rfm"], 
    #               ["cohort_max","rfmv2","rfm"]]
    #calc_cohorts = [["cohort","rfmv1","rfm"]]
 
    # Get m for all discounts
    for c in calc_cohorts:
        conf['slice_cols'] = [c[0]][0]

        slices = [conf['slice_cols'], conf['experiment_group_column']]
        cols = [
            'dp_sessions',
            'dp_rides',
            'dp_gmv_eur',
            'dp_discount_eur',
            'dp_nmv_eur',
            'dp_retained_visitor',
            'dp_retained_rider',
            'hp_sessions',
            'hp_rides',
            'hp_gmv_eur',
            'hp_discount_eur',
            'hp_nmv_eur',
            'hp_retained_visitor',
            'hp_retained_rider',
            'gmv_eur',
            'nmv_eur'
        ] + slices

        # Apply analysis
        m_tmp = df[cols].groupby(slices).agg(['count','mean','sum'])
        m_tmp = m_tmp.rename(columns={c[0]:"treatment_group"})
        m_tmp["treatment_method"] = c[1]
        m_tmp["treatment_method_column"] = c[0]
        m = m.append(m_tmp)

    # Rename columns and filter out columns that are not used
    g = m.copy()
    m = pd.DataFrame(list(g.index), columns=slices)

    # Total riders
    m['riders_in_experiment'] = g['dp_rides']['count'].values
    # Sessions
    m['dp_sessions'] = np.round(g['dp_sessions']['sum'].values,2)
    m['hp_sessions'] = np.round(g['hp_sessions']['sum'].values,2)
    # rides
    m['dp_rides'] = np.round(g['dp_rides']['sum'].values,2)
    m['hp_rides'] = np.round(g['hp_rides']['sum'].values,2)
    # Total discount
    m['dp_discount_eur'] = np.round(g['dp_discount_eur']['sum'].values,2)
    m['hp_discount_eur'] = np.round(g['hp_discount_eur']['sum'].values,2)
    # Retained visitors
    m['dp_retained_visitors'] = g['dp_retained_visitor']['sum'].values
    m['hp_retained_visitors'] = g['hp_retained_visitor']['sum'].values
    # Visitor retention
    m['dp_retention_visitor'] = np.round(g['dp_retained_visitor']['mean'].values,4)
    m['hp_retention_visitor'] = np.round(g['hp_retained_visitor']['mean'].values,4)
    # Retained riders
    m['dp_retained_riders'] = g['dp_retained_rider']['sum'].values
    m['hp_retained_riders'] = g['hp_retained_rider']['sum'].values
    # Rider retention
    m['dp_retention_rider'] = np.round(g['dp_retained_rider']['mean'].values,4)
    m['hp_retention_rider'] = np.round(g['hp_retained_rider']['mean'].values,4)
    # Summed GMV per rider
    m['dp_gmv_eur'] = np.round(g['dp_gmv_eur']['sum'].values,2)
    m['hp_gmv_eur'] = np.round(g['hp_gmv_eur']['sum'].values,2)
    # Avg GMV per rider (ARPU)
    m['dp_gmv_eur_avg'] = np.round(g['dp_gmv_eur']['mean'].values,4)
    m['hp_gmv_eur_avg'] = np.round(g['hp_gmv_eur']['mean'].values,4)
    m['gmv_eur_avg'] = np.round(g['gmv_eur']['mean'].values,4)
    # Summed NMV per rider
    m['dp_nmv_eur'] = np.round(g['dp_nmv_eur']['sum'].values,2)
    m['hp_nmv_eur'] = np.round(g['hp_nmv_eur']['sum'].values,2)
    # Avg NMV per rider (Net ARPU)
    m['dp_nmv_eur_avg'] = np.round(g['dp_nmv_eur']['mean'].values,4)
    m['hp_nmv_eur_avg'] = np.round(g['hp_nmv_eur']['mean'].values,4)
    m['nmv_eur_avg'] = np.round(g['nmv_eur']['mean'].values,4)

    m['treatment_method'] = g["treatment_method"].values
    m['treatment_method_column'] = g["treatment_method_column"].values

    # Sanity check for proportion of discounts
    m['dp_discount_of_gmv_pct'] = np.round(m['dp_discount_eur'] / m['dp_gmv_eur'],4)
    m['hp_discount_of_gmv_pct'] = np.round(m['hp_discount_eur'] / m['hp_gmv_eur'],4)

    #rename "cohort"/"cluster" to "treatment_group"
    m = m.rename(columns={m.columns[0]:'treatment_group'})
    
    m["dp_discount_eur_avg"] = m["dp_discount_eur"] / m["riders_in_experiment"]
    m["treatment_type"] = np.where(m["discount"] == -1,'control', 'treatment')

    #add individual treatment efficiency to the df
    if calc_efficiency:
        
        fields = [["treatment_method","treatment_group"],"treatment_type"]
        control_name= 'control' #name of the control group in groupby_1 array
    

        #add control metrics to df
        m_control = m[m[fields[1]]==control_name][fields[0]+["dp_gmv_eur_avg","hp_gmv_eur_avg","dp_discount_eur_avg"]]
        m_control = m_control.rename(columns = {"dp_gmv_eur_avg":"control_dp_gmv_eur_avg",
                                                "hp_gmv_eur_avg":"control_hp_gmv_eur_avg",
                                                "dp_discount_eur_avg":"control_dp_discount_eur_avg"
                                                   })
        m = m.merge(m_control, how = 'left', on = fields[0]) 


        #calculate efficiency
        m["dp_gmv_uplift"]= np.where(m[fields[1]]==control_name,
                                np.nan,
                                m["dp_gmv_eur_avg"]-m["control_dp_gmv_eur_avg"])

        m["dp_gmv_uplift_ratio"]=np.where(m[fields[1]]==control_name,
                                    np.nan,
                                    m["dp_gmv_uplift"]/m["dp_discount_eur_avg"])
        m["hp_gmv_uplift"]= np.where(m[fields[1]]==control_name,
                                np.nan,
                                m["hp_gmv_eur_avg"]-m["control_hp_gmv_eur_avg"])

        m["hp_gmv_uplift_ratio"]=np.where(m[fields[1]]==control_name,
                                    np.nan,
                                    m["hp_gmv_uplift"]/m["dp_discount_eur_avg"])
    if calc_p:

        #set names for aggregations and calcs
        groupby_treatment = df["discount"].unique() #this feature includes control group and different discounts
        control_name= -1 #name of the control group in groupby_treatment
        
        #p-value for AB
        #fields in user level dataset
        fields_p = ["dp_gmv_eur","hp_gmv_eur"]

        for c in calc_cohorts: #go through all the treatment methods we test - each will get it's own p-values
            groupby_2 = df[c[0]].unique() #go through each cohort in treatment method
            for group in groupby_2: 
                for treatment in groupby_treatment: 
                    for field in fields_p: #go through the metrics for which we'd like to have p-values
                        p=ttest_2(df[(df["discount"]==control_name)&(df[c[0]] == group)][field],
                                  df[(df["discount"]==treatment)&(df[c[0]] == group)][field])[1]
                        p = round(p,4)
                        #rename some fields
                        if field == "ride_price_sum": field = "gmv_sum"
                        if field == "price_sum": field = "nmv_sum"
                        #print(country + ' ' + treatment + ' ' + field + ' {}'.format(p))
                        m.loc[(m["discount"]==treatment)&(m["treatment_group"]==group)&(m["treatment_method_column"]==c[0]), str(field) + ' p-val'] = p

    return m 


def do_simulations(m, combinations, clusters):
    """
        Calculates simulation metrics for all combinations
    """ 
    t0 = time.time()

    print("Will run {} combinations times {} treatment methods".format(len(combinations),m.treatment_method.nunique()))

    fields_extra = ["treatment_method","treatment_method_column","treatment_group","discount"]
    fields_groupby = ["treatment_method","treatment_method_column"]
    fields_sum = [
        "dp_sessions","dp_rides","dp_discount_eur","dp_gmv_eur","dp_nmv_eur","dp_retained_riders","dp_retained_visitors",
        "hp_sessions","hp_rides","hp_discount_eur","hp_gmv_eur","hp_nmv_eur","hp_retained_riders","hp_retained_visitors",
        "riders_in_experiment"
    ]

    # Create discount combinations  dataframe
    flat_list = [item for sublist in combinations for item in sublist]
    treatment_group = [i[0] for i in flat_list]
    discount = [i[1] for i in flat_list]
    row_nr = list(np.sort(list(np.arange(0,len(combinations),1))*len(clusters)))
    g = pd.DataFrame({'treatment_group': treatment_group, 'discount': discount, 'row_nr': row_nr})
    
    discounts_dict = {}
    discounts = []
    for i,c in enumerate(combinations):
        discounts.append(str(dict(c)))
    discounts_dict['row_nr'] = list(np.arange(0,len(combinations),1))
    discounts_dict['discounts'] = discounts
    discounts_df = pd.DataFrame(discounts_dict)
    g = g.merge(discounts_df, on='row_nr')

    # Merge discount combination with corresponding data
    g = g.merge(m, on=['treatment_group','discount'], how='left')
    
    # Aggregate metrics to get one row per each discount combination 
    g = g.groupby(fields_groupby+['discounts'])[fields_sum].sum().reset_index()
    
    # Parse cohort discounts into readable form
    g["method_type"]=np.where(g["discounts"].str[:2]=="{'","cohort","cluster")
    g["discounts"] = g["discounts"].apply(lambda x: ast.literal_eval(x))

    cohort_fields = ["c1","c2","c3","c4","c5","c6"]
    cohort = ['not_active_last_4_month', 'low_activity', 'med_activity', 'high_recency', 'high_frequency', 'highest_activity']
    cluster = [0, 1, 2, 3, 4, 5]

    i=0
    for c in cohort_fields:
        g[c] = np.where(g["method_type"]=="cohort",
                        g["discounts"].apply(lambda x: x.get(cohort[i])),
                        g["discounts"].apply(lambda x: x.get(cluster[i])))
        i = i+1
    
    # Add control groups to calculations
    df_control = m[(m["treatment_method"]==m.treatment_method.unique()[0])&(m["discount"]==-1)][fields_extra+fields_sum]
    df_control = df_control.groupby(fields_groupby)[fields_sum].sum().reset_index()
    df_control.treatment_method = 'control'
    df_control.treatment_method_column = 'control'
    df_control["method_type"] = 'control'
    g = g.append(df_control)
    #print(f"Time taken: {round((time.time() - t0)/60,2)} min.")
    return g


def calculate_efficiency(df):
    """
        Calculates efficiency metrics by comparing treatment to control
    """
    
    #TODO eemaldada kontrollgrupi kulu treatment grupist rida 404 ja 408 ja 410 (arvestada grupi suurusega, proportsionaalselt eemaldada)
    
    df["dp_gmv_per_enrolled_rider"] = df["dp_gmv_eur"] / df["riders_in_experiment"]
    df["hp_gmv_per_enrolled_rider"] = df["hp_gmv_eur"] / df["riders_in_experiment"]

    df["dp_discount_per_enrolled_rider"] = df["dp_discount_eur"] / df["riders_in_experiment"]
    df["hp_discount_per_enrolled_rider"] = df["hp_discount_eur"] / df["riders_in_experiment"]

    df["dp_gmv_uplift_per_enrolled_rider"] = np.where(df["treatment_method"]=="control",
        -1, df["dp_gmv_per_enrolled_rider"] - df[df["treatment_method"]=="control"]["dp_gmv_per_enrolled_rider"].iloc[0])

    df["hp_gmv_uplift_per_enrolled_rider"] = np.where(df["treatment_method"]=="control",
        -1, df["hp_gmv_per_enrolled_rider"] - df[df["treatment_method"]=="control"]["hp_gmv_per_enrolled_rider"].iloc[0])

    df["dp_gmv_uplift_ratio_per_enrolled_rider"] = np.where(df["treatment_method"]=="control",
        -1, df["dp_gmv_uplift_per_enrolled_rider"] / (df["dp_discount_per_enrolled_rider"]))

    df["hp_gmv_uplift_ratio_per_enrolled_rider"] = np.where(df["treatment_method"]=="control",
        -1, df["hp_gmv_uplift_per_enrolled_rider"] / df["dp_discount_per_enrolled_rider"])

    df["dp_discount_cost_pct"] =  100 * df["dp_discount_eur"] / df["dp_gmv_eur"]

    df["dp_hp_sum_gmv_uplift_ratio_per_enrolled_rider"] = df["dp_gmv_uplift_ratio_per_enrolled_rider"] + df["hp_gmv_uplift_ratio_per_enrolled_rider"]

    df["dp_hp_delta_gmv_uplift_ratio_per_enrolled_rider"] = df["dp_gmv_uplift_ratio_per_enrolled_rider"] - df["hp_gmv_uplift_ratio_per_enrolled_rider"]
    
    return df


def apply_cost_estimation(df_riders, combinations, params, caps, rides, n_rides):
    """
        Applies cost estimation on all (cohort, discount) combinations based
        on riders activity in a specified week.
    """
    t0 = time.time()
    # Creates a list of ride ranks
    n_rides_list = []
    for i in range(n_rides):
        n_rides_list.append(i+1)

    # Remove riders without cohort
    df_riders = df_riders.dropna(subset=[params['cohort_name']])

    # Define discounted dataset - remove signed up riders and take only up to allowed number of rides
    dfd = df_riders[(df_riders.rk.isin(n_rides_list)) & (df_riders.signups != '1')]

    # Create (cohort, discount) combinations
    flat_list = [item for sublist in combinations for item in sublist]
    treatment_group = [i[0] for i in flat_list]
    discount = [i[1]/100 for i in flat_list]
    row_nr = list(np.sort(list(np.arange(0,len(combinations),1))*len(combinations[0])))
    g = pd.DataFrame({params['cohort_name']: treatment_group, 'apply_discount': discount, 'row_nr': row_nr})
    discounts_dict = {}
    discounts = []

    for i,c in enumerate(combinations):
        discounts.append(str(dict(c)))

    discounts_dict['row_nr'] = list(np.arange(0, len(combinations), 1))
    discounts_dict['discounts'] = discounts
    discounts_df = pd.DataFrame(discounts_dict)
    g = g.merge(discounts_df, on='row_nr')

    # Caps dataframe
    caps_df = pd.DataFrame.from_dict(caps, orient='index').reset_index()
    caps_df.columns=[params['cohort_name'], 'apply_cap']
    caps_df['apply_cap'] = caps_df['apply_cap'].apply(lambda x: float('Inf') if x == 0 else x)

    # Max discounted rides dataframe
    n_rides_df = pd.DataFrame.from_dict(rides, orient='index').reset_index()
    n_rides_df.columns=[params['cohort_name'], 'n_rides']

    # Merge riders data with maximum allowed rides
    dfd = dfd[[params['cohort_name'],'rk','ride_price']].merge(n_rides_df, on=params['cohort_name'], how='left')

    # Merge riders data with caps
    dfd = dfd.merge(caps_df, on=params['cohort_name'], how='left')

    # Remove ride ranks that are not discounted
    dfd = dfd[dfd.rk <= dfd.n_rides]

    # Merge riders data with (cohort, discount) combination pairs
    dfd = g[[params['cohort_name'],'apply_discount','row_nr','discounts']].\
        merge(dfd[[params['cohort_name'],'ride_price','apply_cap']], how='left', on=params['cohort_name'])

    # Apply minimum of estimated cost and cap
    dfd['est_cost'] = np.minimum(dfd.ride_price * dfd.apply_discount, dfd.apply_cap)

    # Calculate costs per cohort and merge with riders gmv
    dfrg = df_riders.groupby(params['cohort_name'])[["ride_price"]].sum().reset_index()
    dfdg = dfd.groupby(['discounts', params['cohort_name']])[["est_cost"]].sum().reset_index()
    c_cohorts = dfdg.merge(dfrg, how="left", on=params['cohort_name'])

    # Calculate total gmv
    total_gmv = df_riders.ride_price.sum()

    # Calculate cost proportion
    c_cohorts['cost_pct'] = 100 * (c_cohorts.est_cost / total_gmv) * params['control_share']

    # Group all costs together
    c_total = c_cohorts.groupby('discounts')[['est_cost','ride_price','cost_pct']].agg('sum').reset_index()
    print(f"Time taken: {round((time.time()-t0)/60,2)} mins")
    return c_total, c_cohorts


def apply_filters(df, min_spend_eur, min_enrollments):
    """
        Applies filters on simulations such as minimum budget
    """
    # Minimum allowed spend in EUR
    print(f"Total simulations: {len(df)}")
    df = df[df['dp_discount_eur'] >= min_spend_eur]
    print(f"Simulations after min_spend_eur filter: {len(df)}")

    # Minimum allowed number of enrolled riders
    df = df[df['riders_in_experiment'] >= min_enrollments]
    print(f"Simulations after min_enrollments filter: {len(df)}\n")
    return df


def prep_data(df, metric, budget_step):
    """
        Prepare dataset
    """  
    # Remove control
    df = df[df.treatment_method != 'control']

    # Rename columns to more readable format
    df = df.rename(columns={
        'dp_gmv_uplift_ratio_per_enrolled_rider': 'dp_efficiency',
        'hp_gmv_uplift_ratio_per_enrolled_rider': 'hp_efficiency',
        'dp_hp_sum_gmv_uplift_ratio_per_enrolled_rider': 'dp_hp_efficiency',
        'dp_discount_cost_pct': 'dp_cost'})

    # Define relevant columns 
    cols = ['treatment_method','c1','c2','c3','c4','c5','c6',
            'dp_cost','dp_efficiency','hp_efficiency','dp_hp_efficiency']
    df = df[cols]

    # Create budget bins in steps of 5pct
    df['budget_bin'] = (np.floor(df['dp_cost']/budget_step) * budget_step).astype(int)

    # Sort combinations by budget bin and efficiency ratio
    df = df.sort_values(by=['treatment_method','budget_bin', metric], ascending=False)
    return df


def plot_average_efficiency(df, budget, budget_step, metric, city):
    """
        Plots average efficiencies over all simulations
    """
    plt.figure(figsize=(13,6))
    colors = ['blue','red','green','orange','black']
    for i, m in enumerate(df.treatment_method.unique()):
        rm = df[df.treatment_method == m]
        rmg = rm.groupby('budget_bin')[[metric,'dp_cost']].agg(['count','mean','std']).reset_index()
        rmg.columns = ['_'.join(col).strip() for col in rmg.columns.values]
        plt.plot(rmg.budget_bin_, rmg[metric+'_mean'], label=f"{m} (mean)", color=colors[i])
        plt.plot(rmg.budget_bin_, rmg[metric+'_mean'] + rmg[metric+'_std'], 
                 linewidth=2, linestyle='--', color=colors[i], label=f"{m} (1 stddev)")
        plt.plot(rmg.budget_bin_, rmg[metric+'_mean'] - rmg[metric+'_std'], 
                 linewidth=2, linestyle='--', color=colors[i])
    plt.axvline(budget, linewidth=3, alpha=0.5, linestyle='--', color='black', label='chosen budget range')
    plt.axvline(budget+budget_step, linewidth=3, alpha=0.5, linestyle='--', color='black')
    plt.legend()
    #plt.ylim(-2,5)
    plt.ylabel(f'Average {metric}')
    plt.xlabel(f'Budget as % of GMV (steps of {budget_step} %)')
    plt.title(f"({city}) Budget vs Efficiency")
    plt.show()
    
    
def plot_average_discounts(df, budget_step, percentile_step, metric, city):
    """
        Plots average discounts for different efficiency quantiles
    """
    for m in df.method.unique():
        print(f"\nCohorting method: {m}")
        for b in np.sort(df.budget.unique()):
            sp = df[(df.method==m) & (df.budget==b)]
            
            plt.figure(figsize=(13,6))

            percentile_list = list(np.arange(0,1,percentile_step/100)[:3]) + list(np.arange(0,1,percentile_step/100)[-3:])
            colors = plt.cm.Reds(np.linspace(1, 0.1, len(percentile_list)))
            for k, i in enumerate(percentile_list):
                q = sp[sp['quantile'] == i+percentile_step/100].iloc[0]
                plt.plot(['c1','c2','c3','c4','c5','c6'], 
                         [q['c1_mean'],q['c2_mean'],q['c3_mean'],q['c4_mean'],q['c5_mean'],q['c6_mean']],
                         linewidth=3, color=colors[k], label=f"%tile: {int(100*i)}-{int(100*(i+percentile_step/100))}% (Avg eff.: {q[metric+'_mean']})")
            plt.xlabel('Cohorts')
            plt.ylabel('Average discount level %')
            plt.title(f"({city}) Method: {m}, Budget level: {int(b)}-{int(b+budget_step)}% of GMV\nColors correspond to most (dark) to least (light) efficient in holdout period\nMetric: {metric}",
                     fontsize=18)
            #plt.ylim(-2,50)
            plt.legend(fontsize=11)
            plt.show()


def plot_scatter(df, metric, city):
    """
        Plots scatterplot of efficiencies
    """
    colors = ['blue','red','green','black','orange']
    for i, m in enumerate(df['treatment_method'].unique()):
        print(f"\nCohorting method: {m}")
        sp = df[df['treatment_method']==m]
        plt.figure(figsize=(6,4))
        plt.scatter(sp['dp_cost'], sp[metric], alpha=0.1, s=10, color=colors[i])
        plt.xlabel('Cost as percentage of GMV')
        plt.ylabel(metric)
        plt.title(f"({city}) Method: {m}, Metric: {metric}", fontsize=14)
        plt.legend(fontsize=11)
        plt.show()
            

def pick_combination(df, budget_step, budget, metric, conf, percentile_step, plots=True):
    """
        Picks best cohort discount combination for each clustering method
        with closest match to average discount in top 10th percentile of all combinations.
    """
    df = prep_data(df, metric, budget_step)

    if plots:
        plot_average_efficiency(df, budget, budget_step, metric, conf['city'])

    methods = df.treatment_method.unique()
    print(f"Clustering methods: {methods}")
    budgets = np.sort(df.budget_bin.unique())

    summary_cols = ['method','budget','quantile',metric+'_mean',metric+'_std',
        'c1_mean','c2_mean','c3_mean','c4_mean','c5_mean','c6_mean',
        'c1_std','c2_std','c3_std','c4_std','c5_std','c6_std']

    s = pd.DataFrame(columns=summary_cols)

    for m in methods:
        # Iterate over budget bins
        a = df[(df.budget_bin == budget) & (df.treatment_method == m)]
        a['row_nr'] = np.arange(len(a))+1

        for i in np.arange(0,1,percentile_step/100):
            # Iterate over quantiles
            aq = a[(a.row_nr >= a.row_nr.quantile(i)) & (a.row_nr < a.row_nr.quantile(i+percentile_step/100))]
            s1 = {}
            s1['method'] = m
            s1['budget'] = budget
            s1['quantile'] = i+percentile_step/100
            s1[metric+'_mean'] = round(aq[metric].mean(),2)
            s1[metric+'_std'] = round(aq[metric].std(),2)
            s1['c1_mean'] = aq.c1.mean()
            s1['c2_mean'] = aq.c2.mean()
            s1['c3_mean'] = aq.c3.mean()
            s1['c4_mean'] = aq.c4.mean()
            s1['c5_mean'] = aq.c5.mean()
            s1['c6_mean'] = aq.c6.mean()
            s1['c1_std'] = aq.c1.std()
            s1['c2_std'] = aq.c2.std()
            s1['c3_std'] = aq.c3.std()
            s1['c4_std'] = aq.c4.std()
            s1['c5_std'] = aq.c5.std()
            s1['c6_std'] = aq.c6.std()
            s = s.append(pd.DataFrame([s1]))

    if plots:
        plot_average_discounts(s, budget_step, percentile_step, metric, conf['city'])
        plot_scatter(df, metric, conf['city'])

    discounts = list(np.sort(df.c1.unique()))
    print(f"Allowed discount levels: {discounts}")

    # Pick closest match to average discount levels in the best performing quantile
    s['c1_discount'] = s.apply(lambda x: min(discounts, key=lambda y:abs(y-x['c1_mean'])), axis=1)
    s['c2_discount'] = s.apply(lambda x: min(discounts, key=lambda y:abs(y-x['c2_mean'])), axis=1)
    s['c3_discount'] = s.apply(lambda x: min(discounts, key=lambda y:abs(y-x['c3_mean'])), axis=1)
    s['c4_discount'] = s.apply(lambda x: min(discounts, key=lambda y:abs(y-x['c4_mean'])), axis=1)
    s['c5_discount'] = s.apply(lambda x: min(discounts, key=lambda y:abs(y-x['c5_mean'])), axis=1)
    s['c6_discount'] = s.apply(lambda x: min(discounts, key=lambda y:abs(y-x['c6_mean'])), axis=1)

    best_matches = pd.DataFrame()
    best_simulations = pd.DataFrame()
    for m in methods:
        # Best simulations for each clustering method
        bs1 = df[(df.treatment_method==m) & (df.budget_bin==budget)].sort_values(by=metric, ascending=False).head(1)
        best_simulations = best_simulations.append(bs1)

        # Best simulations based on closest match by average discount levels for each clustering method
        sb = s[(s.method==m) & (s.budget==budget)].sort_values(by=metric+'_mean', ascending=False)
        bm1 = df[(df.treatment_method==m) & (df.c1==sb.c1_discount.iloc[0]) & (df.c2==sb.c2_discount.iloc[0]) 
            & (df.c3==sb.c3_discount.iloc[0]) & (df.c4==sb.c4_discount.iloc[0]) & (df.c5==sb.c5_discount.iloc[0]) 
            & (df.c6==sb.c6_discount.iloc[0])]
        best_matches = best_matches.append(bm1)
    best_simulations = best_simulations.sort_values(by=metric, ascending=False).reset_index(drop=True)
    best_matches = best_matches.sort_values(by=metric, ascending=False).reset_index(drop=True)
    return s[s.budget==budget], best_simulations, best_matches

def list_to_string(lst):
    s = ",".join(map(str, lst))   
    return s

