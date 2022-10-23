from django.shortcuts import render_to_response, redirect, render
from django.http import HttpResponse
from django.contrib import auth
from django.contrib.auth import authenticate, login, logout
from django.template import loader, RequestContext
from django.http import HttpResponseRedirect
import hashlib
import pymssql
from common import config as c
from common.config import temp_files
from common.functions import execute_db, get_client_ip, get_meta, save_event
from common import sql_requests as sr
from common import sql_requests_mr as srmr
from django.http import JsonResponse
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
import os
import json
import re
from scipy import stats
from sklearn.metrics import roc_auc_score
import logging
import pickle
# disable warnings
import warnings
warnings.filterwarnings("ignore")
from scorecard.models import ClientType, BusinessType, Scorecard
from development.models import Gantt, DataSource
from limits.models import SegmentInfo
from activity.models import Activity
from django.contrib.auth.models import User
from django.db.models.expressions import RawSQL
from django.contrib.auth.decorators import login_required
from collections import OrderedDict


# init logging
logger = logging.getLogger("log")
logger.setLevel(logging.INFO)


@login_required
# calculate gini locally
def scorecard_monitoring_new(request):

    save_event(request)

    if request.POST: 
        cur_scorecard = request.POST['scorecard']
        cur_business_type = request.POST['business_type']
        cur_client_type = request.POST['client_type'].rsplit('_', 1)[1]

        print(request.POST['client_type'], cur_client_type)
    else:

        cur_scorecard_obj = Scorecard.objects.filter(name='MainScore_201709_Core_New_EquifaxHit', business_type='Core')[0] # MainScore_201709_Core_New_EquifaxHit dn_pd_eq_1217
        cur_scorecard = str(cur_scorecard_obj.name)
        cur_business_type = str(cur_scorecard_obj.business_type)
        cur_client_type = str(cur_scorecard_obj.client_type)

        # print(type(cur_client_type))
        # print('Новый', cur_client_type, 'Новый' == cur_client_type) #, len('Новый'), len(cur_client_type))

    now = datetime.now()

    table_name = '[TestForRisk].[dbo].[shorokh_scorecard_variables_%s]'
    table_date = str(now.year) + '_' + '0' * (2 - len(str(now.month))) + str(now.month) + '_01'

    # init logging
    logger = logging.getLogger("log")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=temp_files + '/scorecard_monitoring/log/' + datetime.strftime(now, '%Y_%m_%d') + ".log")  # create the logging file handler
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add handler to logger object
    # logger.info()
    # logger.removeHandler(fh)

    cur_log = datetime.strftime(now, '%H_%M_%S')

    logger.info('current filters: [' + cur_business_type + '] [' + cur_client_type + '] [' + cur_scorecard + ']')
    print('------------ start -------------', cur_business_type, cur_client_type, cur_scorecard)

    # cur_week_start = now - timedelta(days=(now.weekday() + 1))
    # cur_week_end = cur_week_start + timedelta(days=6)

    prev_week_end = now - timedelta(days=(now.weekday() + 1))
    # print(prev_week_end)

    ################################################################
    #                          INIT DIRS                           #
    ################################################################

    sm_base = "/scorecard_monitoring/"

    ################################################################
    #                       SCORECARDS LIST                        #
    ################################################################
    
    min_appl_count = 100
    cur_file = 'scorecards_' + datetime.strftime(prev_week_end, "%Y_%m_%d") + '.csv'

    if cur_file in os.listdir(temp_files + sm_base + 'scorecards_list/'):
        scorecards = pd.read_csv(temp_files + sm_base + 'scorecards_list/' + cur_file, encoding='cp1251', sep='\t')
        logger.info(cur_log + ' scorecards list loaded from ' + cur_file)
        print(cur_log + ' scorecards list loaded from ' + cur_file)
   
    else:

        # take 4 weeks ago
        weeks_to_past = 4  # weeks count for option selection
        dt = datetime.now()
        cur_week_start = dt - timedelta(days=(dt.weekday() + 7))
        cur_week_end = cur_week_start + timedelta(days=6)
        weeks_ago = []
        for i in range(weeks_to_past):
            week_start = cur_week_start - timedelta(days=7)
            week_end = week_start + timedelta(days=6)
            cur_week_start = week_start
            weeks_ago.append("'"+ datetime.strftime(week_start, "%Y-%m-%d") + ' --- ' + datetime.strftime(week_end, "%Y-%m-%d") + "'")
        
        req = sr.scorecards_list_per_weeks
        req = req.replace("WEEKS1", ','.join(weeks_ago)).replace("COUNT1", str(min_appl_count))

        res = execute_db(c.db_p, req, pr=True)
        logger.info(cur_log + '\n' + req + '\n')
        scorecards = pd.DataFrame(res[1:], columns=res[0])

        # scorecards_db = Scorecard.objects.filter(working=True)
        if not scorecards.empty:
            scorecards.to_csv(temp_files + sm_base + 'scorecards_list/' + cur_file, encoding='cp1251', sep='\t', index=False)
    
    
        ################################################################
        #                      UPDATE SCORECARDS                       #
        ################################################################

        for i in range(len(scorecards)):
            check_scorecard = Scorecard.objects.filter(name=scorecards.loc[i, 'SCORECARD'], business_type=scorecards.loc[i, 'BUSINESS_TYPE'], 
               client_type=scorecards.loc[i, 'CLIENT_TYPE'])
            if check_scorecard:
                pass
            else:
                new_scorecard = Scorecard(name=scorecards.loc[i, 'SCORECARD'], business_type=scorecards.loc[i, 'BUSINESS_TYPE'], 
                client_type=scorecards.loc[i, 'CLIENT_TYPE'])
                new_scorecard.working = True
                new_scorecard.save()

            # set-off scorecards
            for scorecard_db in Scorecard.objects.all().order_by('business_type', 'client_type'):
                if scorecards[(scorecards['SCORECARD'] == str(scorecard_db.name)) & (scorecards['CLIENT_TYPE'] == str(scorecard_db.client_type)) 
                & (scorecards['BUSINESS_TYPE'] == str(scorecard_db.business_type))].empty:
                    scorecard_db.working = False
                    scorecard_db.save()

    # print(scorecards)

    # scorecards by python
    cur_file = 'scorecards_by_python_' + datetime.strftime(prev_week_end, "%Y_%m_%d") + '.pkl'

    if cur_file in os.listdir(temp_files + sm_base + 'scorecards_list/'):
        
        with open(temp_files + sm_base + 'scorecards_list/' + cur_file, 'rb') as file:
            scorecards_by_python = pickle.load(file)   
        logger.info(cur_log + ' scorecards by python list loaded from ' + cur_file)
        print(cur_log + ' scorecards by python list loaded from ' + cur_file)

    else:
        req = sr.scorecards_by_python
        res = execute_db(c.db_p, req, pr=True)[1:]
        logger.info(cur_log + '\n' + req + '\n')
        if res:
            scorecards_by_python = [el[0] for el in res]
        else:
            scorecards_by_python = []

        if scorecards_by_python:
            with open(temp_files + sm_base + 'scorecards_list/' + cur_file, 'wb') as file:
                pickle.dump(scorecards_by_python, file)

    # print(scorecards_by_python)

    ################################################################
    #                        REQUESTS DICT                         #
    ################################################################

    cur_file = 'requests.pkl'
    if cur_file in os.listdir(temp_files + sm_base + '/'):
        with open(temp_files + sm_base + '/' + cur_file, 'rb') as file:
            requests_out = pickle.load(file)
    else:
        requests_out = {} # dict for all requests

    ################################################################
    #                           GINI                               #
    ################################################################

    min_appl_count = 100
    months_ago_gini = 4

    month1 = now.month - months_ago_gini
    year1 = now.year
    if month1 <= 0:
        months_last_year = months_ago_gini - now.month
        month1 = 12 - months_last_year
        year1 -= 1  # previous year

    date_for_indicator = str(year1) + '-' +  '0'*(2 - len(str(month1))) + str(month1)

    month2 = now.month - months_ago_gini - 1
    year2 = now.year
    if month2 <= 0:
        months_last_year = months_ago_gini + 1 - now.month
        month2 = 12 - months_last_year
        year2 -= 1  # previous year

    date_prev = str(year2) + '-' +  '0'*(2 - len(str(month2))) + str(month2)
    
    print('date_for_indicator', date_for_indicator)
    print('date_prev', date_prev)
    
    cur_file = 'gini_basis_' + date_for_indicator + '.csv'
    if cur_file not in os.listdir(temp_files + sm_base + '/scorecards_gini/'):
        
        # calculate all ginis

        last_file = max([el for el in os.listdir(temp_files + sm_base + '/scorecards_gini/') if el != 'log'])
        print(last_file)
        p = re.compile(r'\d_(.*)\.csv')
        print(cur_file, p.findall(cur_file))
        # last_date = p.findall(cur_file)[-1]
        last_date = cur_file.split('_')[-1].split('.')[0]
        print(last_date)

        req = sr.scorecards_list_per_month
        req = req.replace("MONTH1", date_for_indicator).replace("COUNT1", str(min_appl_count))
        data = execute_db(c.db_p, req, pr=True)

        logger.info(cur_log + '\n' + req + '\n')
        
        df = pd.DataFrame(data[1:], columns=data[0])
        df['gini'] = -1
        df['ks'] = -1

        print(df)

        if not df.empty:

            df.insert(0, 'MONTH', date_for_indicator)
            df.insert(3, 'BUREAU', np.nan)
            df.insert(4, 'SEGMENT', np.nan)

            #df = df[df['SCORECARD'].isin(['dn_pd_eq_1217', 'cr_no_upsale_0318'])]
            # df.index = range(len(df))
            # print(df)

            for i in range(len(df)):
                CUR_SCORECARD = df.loc[i, 'SCORECARD']
                CUR_BUSINESS_TYPE = df.loc[i, 'BUSINESS_TYPE']
                CUR_CLIENT_TYPE = df.loc[i, 'CLIENT_TYPE']
                bureau = ''

                # Init business type
                if CUR_BUSINESS_TYPE == 'Core':
                    CUR_BUSINESS_TYPE1 = "Core', 'from_SPR7_to_SPR4', 'from_SPR14_to_SPR4"
                else:
                    CUR_BUSINESS_TYPE1 = CUR_BUSINESS_TYPE

                # Init bureau
                cond1 = "1=1"
                if 'equifaxhit' in CUR_SCORECARD.lower() or 'eq' in CUR_SCORECARD.lower():
                    cond1 = "Bureau = 'Equifax hit'"
                    bureau = 'EquifaxHit'

                if 'nobureauhit' in CUR_SCORECARD.lower():
                    bureau = 'NoBureauHit'

                if 'nbch' in CUR_SCORECARD.lower():
                    bureau = 'NbchHit'

                df.loc[i, 'BUREAU'] = bureau

                # Init score field
                if CUR_SCORECARD in scorecards_by_python:
                    cur_score = 'MainPROBABballs as MainSCOREballs'  # MainSCOREballs, MainPROBABballs,
                else:
                    cur_score = 'MainSCOREballs' 

                req = sr.data_for_gini_badrate
                req = req.replace("MONTH1", date_for_indicator).replace("SCORECARD1", CUR_SCORECARD).replace("BUSINESS_TYPE1", CUR_BUSINESS_TYPE1).replace("CLIENT_TYPE1", CUR_CLIENT_TYPE).replace("COND1", cond1).replace("SCORE1", cur_score)
               
                label = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + 'gini'
                if label not in requests_out.keys():
                    requests_out[label] = req

                res = execute_db(c.db_p, req, pr=True)

                logger.info(cur_log + '\n' + req + '\n')

                temp = pd.DataFrame(res[1:], columns=res[0])
                temp['MainSCOREballs'] = temp['MainSCOREballs'].astype(float)
                print(temp.head())

                try:
                    gini = 100 * abs(2 * roc_auc_score(temp[temp['BadRate'].notnull()]['BadRate'], temp[temp['BadRate'].notnull()]['MainSCOREballs']) - 1)
                    ks = 100 * stats.ks_2samp(temp[temp['BadRate'] == 0]['MainSCOREballs'], temp[temp['BadRate'] == 1]['MainSCOREballs']).statistic   
                    df.loc[i, 'gini'] = gini
                    df.loc[i, 'ks'] = ks     
                except:
                    pass

                ################################################################
                #           Scorecard variables effectiveness part             #
                ################################################################

                try:
                    
                    report_date = str(year1) + '-' + '0' * (2 - len(str(month1))) + str(month1)

                    share_date_month = month1 + 4   
                    share_date_year = year1
                    if share_date_month > 12:
                        share_date_month = share_date_month % 12
                        share_date_year += 1  # previous year
                    share_date = str(share_date_year) + '_' + '0' * (2 - len(str(share_date_month))) + str(share_date_month) + '_01'

                    prev_share_date_month = month1 + 3   
                    prev_share_date_year = year1
                    if prev_share_date_month > 12:
                        prev_share_date_month = prev_share_date_month % 12
                        prev_share_date_year += 1  # previous year
                    prev_share_date = str(prev_share_date_year) + '_' + '0' * (2 - len(str(prev_share_date_month))) + str(prev_share_date_month) + '_01'

                    print('report_date', report_date)
                    print('share_date', share_date)
                    print('prev_share_date', prev_share_date)


                    print()

                    # Create new table

                    req = '''
                    IF OBJECT_ID('TABLE1', 'U') IS NULL begin

                    CREATE table TABLE1 (
                        MONTH NVARCHAR(255),
                        BUSINESS_TYPE NVARCHAR(255),
                        CLIENT_TYPE NVARCHAR(255),
                        BUREAU NVARCHAR(255),
                        SEGMENT NVARCHAR(255),
                        SCORECARD NVARCHAR(255),
                        Variable NVARCHAR(255),  
                        Point FLOAT,
                        AllAppl FLOAT,
                        FinAppl FLOAT,
                        AllShare FLOAT,
                        FinShare FLOAT,
                        FinRate FLOAT,
                        Indicator FLOAT,
                        IndicatorType NVARCHAR(255),
                    )

                    insert
                    into TABLE1
                    select * from TABLE1
                    
                    end

                    '''.replace('TABLE1', table_name) % (share_date, share_date, share_date, prev_share_date)

                    print('Create new table')
                    print(req)

                    conn = pymssql.connect(c.db_d)
                    cur = conn.cursor()
                    cur.execute(req)
                    conn.commit()
                    conn.close()

                    # Get new report data

                    table = '[DSS].[spr].[SPR_ALL_DATA]'

                    scorecard = CUR_SCORECARD
                    client_type = CUR_CLIENT_TYPE
                    business_type = CUR_BUSINESS_TYPE
                    
                    date1 = report_date + '-01'
                    
                    date2_month = month1 + 1
                    date2_year = year1
                    if date2_month > 12:
                        date2_month = date2_month % 12
                        date2_year += 1

                    date2 = str(date2_year) + '-' + '0' * (2 - len(str(date2_month))) + str(date2_month) + '-01'
                    
                    cond2 = '1=1'

                    segment = '%s_%s' % (business_type, 'New' if 'Новый' in client_type else 'Repeat')
                    if bureau:
                        segment += '_%s' % bureau

                    print(date1, date2)

                    req_ve = sr.scorecard_variables_effectiveness.replace("TABLE1", table).replace("SCORECARD1", scorecard).replace("CLIENT_TYPE1", client_type).replace("BUSINESS_TYPE1", CUR_BUSINESS_TYPE1)\
                                 .replace("DATE1", date1).replace("DATE2", date2).replace("COND2", cond2)

                    print('Get new report data')
                    print(req_ve)

                    conn = pymssql.connect(c.db_p)
                    df3 = pd.read_sql(req_ve, conn)
                    conn.close()
                    df3.rename(columns={'CHARACT': 'Variable'}, inplace=True)

                    # df3['POINT'] = df3['POINT'].astype(float)

                    # check 0 group
                    for var in list(df3['Variable'].unique()):
                        if 0.0 not in list(df3[df3['Variable'] == var]['Point']):
                            print('no null here')
                            df3 = df3.append(pd.DataFrame([[var, 0, 0, 0, 0, 0, 0, 0]], columns=list(df3.columns)))

                    df3 = df3.sort_values(by=['Variable', 'Point'])
                    df3.index = range(len(df3))

                    df3.insert(0, 'MONTH', report_date)
                    df3.insert(1, 'BUSINESS_TYPE', business_type)
                    df3.insert(2, 'CLIENT_TYPE', 'New' if 'Новый' in client_type else 'Repeat')
                    df3.insert(3, 'BUREAU', bureau)
                    df3.insert(4, 'SEGMENT', segment)
                    df3.insert(5, 'SCORECARD', scorecard)

                    df3.fillna(0, inplace=True)

                    indicator_type = list(df3.columns)[-1]
                    df3.rename(columns={indicator_type: 'Indicator'}, inplace=True)
                    df3['IndicatorType'] = indicator_type

                    df3 = df3.sort_values(by=['Variable', 'Point'], ascending=[True, True])

                    df3.to_excel(temp_files + sm_base + '/scorecards_gini/eff/%s_%s_%s_%s.xlsx' % (report_date, business_type, client_type, scorecard), index=False)

                    print(df3.head(10))

                    # Delete old report data

                    conn = pymssql.connect(c.db_d)
                    cur = conn.cursor()

                    req = '''
                    delete from TABLE1 where month = '%s' 
                    and scorecard = '%s' 
                    and business_type = '%s'
                    and client_type = '%s'
                    '''.replace('TABLE1', table_name) % (table_date, report_date, scorecard, business_type,  'New' if 'Новый' in client_type else 'Repeat')

                    print('Delete old report data')
                    print(req)

                    cur.execute(req)
                    conn.commit()
                    conn.close()

                    # Insert new report data

                    conn = pymssql.connect(c.db_d)
                    cur = conn.cursor()
                    query = "INSERT INTO TABLE1 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, " \
                            "%s, %s, %s, %s, %s, %s, %s)"
                    query = query.replace('TABLE1', table_name % table_date)

                    print('Insert new report data')
                    print(query)
                    
                    sql_data = tuple(map(tuple, df3.values))
                    print(sql_data)
                    cur.executemany(query, sql_data)
                    conn.commit()
                    cur.close()
                    conn.close()


                    # Get new share data
                    date1 = prev_share_date.replace('_', '-')

                    date2_month = month1 + 4
                    date2_year = year1
                    if date2_month > 12:
                        date2_month = date2_month % 12
                        date2_year += 1  # previous year
                    date2 = str(date2_year) + '-' + '0' * (2 - len(str(date2_month))) + str(date2_month) + '-01'
                    month_new = str(prev_share_date_year) + '-' + '0' * (2 - len(str(prev_share_date_month))) + str(prev_share_date_month)

                    print(date1, date2)

                    req_ve = sr.scorecard_variables_effectiveness.replace("TABLE1", table).replace("SCORECARD1", scorecard).replace("CLIENT_TYPE1", client_type).replace("BUSINESS_TYPE1", CUR_BUSINESS_TYPE1)\
                                                 .replace("DATE1", date1).replace("DATE2", date2).replace("COND2", cond2)

                    print('Get new share data')
                    print(req_ve)

                    conn = pymssql.connect('mck-p-dwh')
                    df_new = pd.read_sql(req_ve, conn)
                    conn.close()
                    df_new.rename(columns={'CHARACT': 'Variable'}, inplace=True)


                    # check 0 group
                    for var in list(df_new['Variable'].unique()):
                        if 0.0 not in list(df3[df3['Variable'] == var]['Point']):
                            print('no null here')
                            df_new = df_new.append(pd.DataFrame([[var, 0, 0, 0, 0, 0, 0, 0]], columns=list(df3.columns)))

                    df_new = df_new.sort_values(by=['Variable', 'Point'])
                    df_new.index = range(len(df_new))

                    df_new.insert(0, 'MONTH', month_new)
                    df_new.insert(1, 'BUSINESS_TYPE', business_type)
                    df_new.insert(2, 'CLIENT_TYPE', 'New' if 'Новый' in client_type else 'Repeat')
                    df_new.insert(3, 'BUREAU', bureau)
                    df_new.insert(4, 'SEGMENT', segment)
                    df_new.insert(5, 'SCORECARD', scorecard)

                    df_new.fillna(0, inplace=True)

                    df_new.rename(columns={indicator_type: 'Indicator'}, inplace=True)
                    df_new['IndicatorType'] = indicator_type

                    df_new = df_new.sort_values(by=['Variable', 'Point'], ascending=[True, True])

                    print(df_new.head(10))

                    df_new.to_excel(temp_files + sm_base + '/scorecards_gini/eff/%s_%s_%s_%s.xlsx' % (month_new, business_type, client_type, scorecard), index=False)

                    # Insert new share data

                    conn = pymssql.connect(c.db_d)
                    cur = conn.cursor()
                    query = "INSERT INTO TABLE1 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, " \
                            "%s, %s, %s, %s, %s, %s, %s)"
                    query = query.replace('TABLE1', table_name % table_date)
                    print(query)
                    sql_data = tuple(map(tuple, df_new.values))
                    print(sql_data)
                    cur.executemany(query, sql_data)
                    conn.commit()
                    cur.close()
                    conn.close()

                    # End
                except Exception as e:
                    print(e)

                ################################################################
                #         Scorecard variables effectiveness part end           #
                ################################################################


            df = df.replace({'CLIENT_TYPE': {'Новый': 'New', 'Повторный': 'Repeat'}})

            print(df)
            logger.info(cur_log + '\n' + df.to_string() + '\n')

            df = df[df['gini'] != -1]  # filter by not null values
            # save all data in excel
            # data = pd.read_excel(temp_files + '/scorecard_gini/gini_basis.xlsx', encoding='cp1251')  # checkpoint from 2018-03
            gini_df = pd.read_csv(temp_files + sm_base + '/scorecards_gini/gini_basis_' + date_prev + '.csv', encoding='cp1251', sep='\t')
            gini_df = pd.concat([gini_df, df], axis=0)
            gini_df.index = range(len(gini_df))
            print(gini_df)
            gini_df.to_csv(temp_files + sm_base + '/scorecards_gini/gini_basis_' + date_for_indicator + '.csv', encoding='cp1251', sep='\t', index=False)
     
    
    # If file exists
    else:
        
        gini_df = pd.read_csv(temp_files + sm_base + '/scorecards_gini/gini_basis_' + date_for_indicator + '.csv', encoding='cp1251', sep='\t')
        print(cur_log + ' gini loaded from ' + 'gini_basis_' + date_for_indicator + '.csv')

    d = {'Новый': 'New', 'Повторный': 'Repeat'}
    gini_df = gini_df[(gini_df['BUSINESS_TYPE'] == cur_business_type) & (gini_df['SCORECARD'] == cur_scorecard) & (gini_df['CLIENT_TYPE'] == d[cur_client_type])]

    gini_df.index = range(len(gini_df))

    # print(gini)
    json_gini = []

    for i in range(len(gini_df)):
        json_gini.append({ 'date': gini_df.loc[i, 'MONTH'], 'value': round(gini_df.loc[i, 'gini'], 3)})
    
    # print(json_gini)

    ################################################################
    #                        RISK GROUPS                           #
    ################################################################

    cur_file = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + datetime.strftime(prev_week_end, "%Y_%m_%d") + '.csv'

    if cur_file in os.listdir(temp_files + sm_base + 'scorecards_data/'):

        file_size = os.stat(temp_files + sm_base + 'scorecards_data/' + cur_file).st_size / (1024 ** 2)  # Get file size
        if file_size > 50:  # If file size more than threshold
            data_main = pd.DataFrame()  # Create result data frame
            chunks = pd.read_table(temp_files + sm_base + 'scorecards_data/' + cur_file, chunksize=1000, iterator=True, encoding='cp1251', sep='\t')
            for chunk in chunks:  # For each part in parts
                data_main = pd.concat([data_main, chunk], axis=0)  # Join file parts

        else:
            data_main = pd.read_csv(temp_files + sm_base + 'scorecards_data/' + cur_file, encoding='cp1251', sep='\t')
        logger.info(cur_log + ' data for ' + cur_scorecard +' loaded from ' + cur_file)
        print(cur_log + ' data for ' + cur_scorecard +' loaded from ' + cur_file)
   
    else:

        # cur_business_type = scorecards[scorecards['SCORECARD'] == cur_scorecard]['BUSINESS_TYPE'].values[0]
        # cur_client_type = scorecards[scorecards['SCORECARD'] == cur_scorecard]['CLIENT_TYPE'].values[0]

        # Init business type
        if str(cur_business_type) == 'Core':
            cur_business_type_f = "Core', 'from_SPR7_to_SPR4', 'from_SPR15_to_SPR4"
        else:
            cur_business_type_f = str(cur_business_type)

        # Init score field
        if cur_scorecard in scorecards_by_python:
            cur_score = 'MainPROBABballs as MainSCOREballs'  # MainSCOREballs, MainPROBABballs,
        else:
            cur_score = 'MainSCOREballs' 

        cond1 = '1=1'  # special condition

        req = sr.scorecards_rg_indicators
        req = req.replace("SCORECARD1", str(cur_scorecard)).replace("BUSINESS_TYPE1", cur_business_type_f).replace("CLIENT_TYPE1", str(cur_client_type)).replace("COND1", cond1).replace("SCORE1", cur_score)
        res = execute_db(c.db_p, req, pr=True)
        
        label = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + 'financed'
        if label not in requests_out.keys():
            requests_out[label] = req.replace('1=1', 'Financed = 1')

        logger.info(cur_log + '\n' + req + '\n')
        data_main = pd.DataFrame(res[1:], columns=res[0])
        if not data_main.empty:
            data_main.to_csv(temp_files + sm_base + 'scorecards_data/' + cur_file, encoding='cp1251', sep='\t', index=False)

    
    # print(data.head())
    # try:
    #     data_main['MainSCOREballs'] = data_main['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
    #     data_main = data_main.replace({'MainSCOREballs': {'NULL': -1}})
    #     data_main['MainSCOREballs'] = data_main['MainSCOREballs'].astype(float)
    # except Exception as e:
    #     print(e)

    # data_main['MainSCOREballs'].fillna(-1, inplace=True)
    # data_main = data_main.replace({'MainSCOREballs': {'NULL': -1}})

    # data_main['MainSCOREballs'] = data_main['MainSCOREballs'].astype(float)

    ################################################################
    #                       FINANCED PART                          #
    ################################################################

    data_f = data_main[data_main['Financed'] == 1].copy()
    data_f_g = data_f.groupby(['CRM_CREATION_Month', 'RISK_GROUP']).agg({'UCDB_ID': ['count']})
    data_f = data_f_g.transpose().reset_index(level=0, drop=True).transpose().reset_index()
    data_f.rename(columns={'CRM_CREATION_Month': 'date'}, inplace=True)

    # print(data_f)
    
    # print(data_f)

    # json_data = [
    #       { 'date': "2018-01", 'A': 60, 'B': 15, 'C': 9, 'D': 6},
    # ]

    # json_gini = [
    #       { 'date': "2018-01", 'value': 0.65 },
    # ]

    json_data_f = []

    for date in sorted(data_f['date'].unique()):
        temp = data_f[data_f['date'] == date]
        lst = {'date': date}
        for el in zip(temp['RISK_GROUP'], temp['count']):
            lst[el[0]] = el[1]
            for rg in list('ABCD'):
                if rg not in lst.keys():
                    lst[rg] = 0
        json_data_f.append(lst)

    # print(json_data_f)

    ################################################################
    #                      ALL APPLICATIONS                        #
    ################################################################

    data_all = data_main.copy()
    data_all_g = data_all.groupby(['CRM_CREATION_Month', 'RISK_GROUP']).agg({'UCDB_ID': ['count']})
    data_all = data_all_g.transpose().reset_index(level=0, drop=True).transpose().reset_index()
    data_all.rename(columns={'CRM_CREATION_Month': 'date'}, inplace=True)
    
    # print(data_all.head())
    json_data_all = []

    for date in sorted(data_all['date'].unique()):
        temp = data_all[data_all['date'] == date]
        lst = {'date': date}
        for el in zip(temp['RISK_GROUP'], temp['count']):
            lst[el[0]] = el[1]
            for rg in list('ABCD'):
                if rg not in lst.keys():
                    lst[rg] = 0
        json_data_all.append(lst)

    # print(json_data_f)

    ################################################################
    #                     SCORE STABILITY GROUP                    #
    ################################################################

    #  -------------------- v.1 (training qcut) --------------------

    # cur_file = cur_scorecard + '_bins.pkl'

    # # load bins
    # if cur_file in os.listdir(temp_files + sm_base + 'scorecards_dev_bins/'):
    #     with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'rb') as file:
    #         bins = pickle.load(file)
    #     logger.info(cur_log + ' bins for ' + cur_scorecard +' loaded from ' + cur_file)

    # else:
    #     scores_file = cur_scorecard + '_scores.xlsx'
    #     if scores_file in os.listdir(temp_files + sm_base + 'scorecards_dev_scores/'):
            
    #         df = pd.read_excel(temp_files + sm_base + 'scorecards_dev_scores/' + scores_file)
    #         print(df.head())
    #         _, bins = pd.qcut(df['Prop'], 10, retbins=True)
    #         bins = list(bins)
    #         bins[0] = -np.inf
    #         bins[-1] = np.inf
    #         df['dev'] = pd.cut(df['Prop'], bins=bins)
    #         # get dev scores
    #         data = df['dev'].value_counts(normalize=True).sort_index().to_frame()  
    #         data.index.rename('group', inplace=True)
    #         data.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_dev_scores.csv', sep='\t', encoding='cp1251')

    #         with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'wb') as file:
    #             pickle.dump(bins, file) 
    #     else:
    #         bins = []

    # if bins:
    #     bins[0] = -np.inf
    #     bins[-1] = np.inf

    # data_bins = pd.read_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_dev_scores.csv', sep='\t', encoding='cp1251', index_col='group')
    
    # # print(data_bins)

    # dates_ss = []

    # # data_f = data_main[data_main['Financed'] == 1].copy()
    # data_f = data_main.copy()

    # for date in sorted(list(data_f['CRM_CREATION_Month'].unique())):

    #     temp = data_f[data_f['CRM_CREATION_Month'] == date]
    #     temp['N'] = pd.cut(temp['MainSCOREballs'], bins=bins)
    #     # get dev scores
    #     data_bins[date] = temp['N'].value_counts(normalize=True).sort_index().values
    #     dates_ss.append(date)

    # print(data_bins)
    # if not data_bins.empty:
    #     data_bins.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + max(dates_ss) + '_scores.csv', sep='\t', encoding='cp1251')

    # json_scores = []
    # for date in dates_ss:
    #     data_bins[date+'_d'] = data_bins[date]-data_bins['dev']
    #     data_bins[date+'_w'] = np.log(data_bins[date]/data_bins['dev'])
    #     data_bins = data_bins.replace({date+'_w': {np.inf: 0, -np.inf: 0}})
    #     data_bins[date+'_s'] = data_bins[date+'_d']*data_bins[date+'_w'] 
    #     ssi = data_bins[date+'_s'].sum()
    #     json_scores.append({'date': date, 'value': round(ssi, 5)})

    # print(json_scores)
    # print(data_bins)
    # data_bins.to_excel(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + max(dates_ss) + '_scores_calc.xlsx')


    #  ----------------------- v.2 (nth week qcut)  ---------------

    from_month_number = 2

    # data_f = data_main[data_main['Financed'] == 1].copy()
    data_ss = data_main.copy()
    # data_ss = data_ss[(~data_ss['MainSCOREballs'].isin(['NULL']))]
    data_ss = data_ss[(data_ss['MainSCOREballs'].notnull()) & (~data_ss['MainSCOREballs'].isin(['NULL']))]

    try:
        data_ss['MainSCOREballs'] = data_ss['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
        data_ss['MainSCOREballs'] = data_ss['MainSCOREballs'].astype(float)
    except Exception as e:
        print(e)

    all_dates = sorted(list(data_ss['CRM_CREATION_Month'].unique()))

    # check scorecard's work 
    if len(all_dates) > 0:

        try:
            date_base = all_dates[from_month_number - 1]
        except:
            date_base = all_dates[0]

        
        # cur_file = cur_scorecard + '_from_'+ str(date_base) +'_month_bins.pkl'
        cur_file = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_from_'+ str(date_base) +'_month_bins.pkl'

        # load bins
        if cur_file in os.listdir(temp_files + sm_base + 'scorecards_dev_bins/'):
            with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'rb') as file:
                bins = pickle.load(file)
            logger.info(cur_log + ' bins for ' + cur_scorecard +' loaded from ' + cur_file)

        else:

            df = data_ss[(data_ss['CRM_CREATION_Month'] == date_base)]
        
            _, bins = pd.qcut(df['MainSCOREballs'], 10, retbins=True, duplicates='drop')
            bins = list(bins)
            bins[0] = -np.inf
            bins[-1] = np.inf
            df[date_base + '_dev'] = pd.cut(df['MainSCOREballs'], bins=bins)
            # get dev scores
            data = df[date_base + '_dev'].value_counts(normalize=True).sort_index().to_frame()  
            data.index.rename('group', inplace=True)
            data.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_from_'+ str(date_base) +'_month_dev_scores.csv', sep='\t', encoding='cp1251')

            with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'wb') as file:
                pickle.dump(bins, file) 

        if bins:
            bins[0] = -np.inf
            bins[-1] = np.inf

        data_bins = pd.read_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_from_'+ str(date_base) +'_month_dev_scores.csv', sep='\t', encoding='cp1251', index_col='group')
        
        # print(data_bins)

        dates_ss = all_dates[(all_dates.index(date_base) + 1):]
        # print('dates_ss', dates_ss)

        if dates_ss:

            for date in dates_ss:

                temp = data_ss[data_ss['CRM_CREATION_Month'] == date]
                temp['N'] = pd.cut(temp['MainSCOREballs'], bins=bins)
                # get dev scores
                data_bins[date] = temp['N'].value_counts(normalize=True).sort_index().values

            # print(data_bins)
            if not data_bins.empty:
                data_bins.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + max(dates_ss) + '_scores.csv', sep='\t', encoding='cp1251')

            json_scores = []
            for date in dates_ss:
                data_bins[date+'_d'] = data_bins[date]-data_bins[date_base + '_dev']
                data_bins[date+'_w'] = np.log(data_bins[date]/data_bins[date_base + '_dev'])
                data_bins = data_bins.replace({date+'_w': {np.inf: 0, -np.inf: 0}})
                data_bins[date+'_s'] = data_bins[date+'_d']*data_bins[date+'_w'] 
                ssi = data_bins[date+'_s'].sum()
                json_scores.append({'date': date, 'value': round(ssi, 5)})

            # print(json_scores)
            # print(data_bins)
            data_bins.to_excel(temp_files + sm_base + 'scorecards_dev_bins/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + max(dates_ss) + '_scores_calc.xlsx')
        else:
            json_scores = []
    # scorecard too new
    else:
        json_scores = []
        date_base = ''


    ################################################################
    #                          MEAN VALUES                         #
    ################################################################

    json_mean_scores = []
    json_mean_indicators = []

    indicators = ['3+ 4 *2WoB', '30+ 3MoB']

    min_date = data_main['CRM_CREATION_Month'].min()  # for filtering in the end

    for date in sorted(list(data_main['CRM_CREATION_Month'].unique())):

        temp = data_main[data_main['CRM_CREATION_Month'] == date].copy()
        temp = temp[(temp['MainSCOREballs'].notnull()) & (~temp['MainSCOREballs'].isin(['NULL']))]

        try:
            temp['MainSCOREballs'] = temp['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
            temp['MainSCOREballs'] = temp['MainSCOREballs'].astype(float)
        except Exception as e:
            print(e)


        score_mean = temp['MainSCOREballs'].mean() if len(temp) > 0 else 0.0
        json_mean_scores.append({"date": date, "score_mean": score_mean})
        
        # FPD
        fpd_mean = temp[temp['FPD'].notnull()]['FPD'].mean() if len(temp[temp['FPD'].notnull()]) > 0 else 0
        json_mean_indicators.append({"date": date, "label": 'fpd', "value": round(fpd_mean, 5)})


        for indicator in indicators:
            temp = temp[temp[indicator].notnull()]
            temp['ind_' + indicator] = temp[indicator] / (temp['LOAN_AMOUNT'] + temp['LOAN_COMISSION'])
            indicator_value = temp[temp["ind_" + indicator].notnull()]["ind_" + indicator].mean() if len(temp[temp["ind_" + indicator].notnull()]) > 0 else 0
            json_mean_indicators.append({"date": date, "label": indicator, "value": round(indicator_value, 5)})

        del temp

    ################################################################
    #                        SHARE OF SCORES                       #
    ################################################################

    temp_ind = data_main[(data_main['Financed'] == 1) & (data_main['CRM_CREATION_Month'] == date_for_indicator)].copy()
    temp_ind = temp_ind[(temp_ind['MainSCOREballs'].notnull()) & (~temp_ind['MainSCOREballs'].isin(['NULL']))]
    
    temp = data_main[data_main['CRM_CREATION_Month'] == date_for_indicator].copy()
    temp = temp[(temp['MainSCOREballs'].notnull()) & (~temp['MainSCOREballs'].isin(['NULL']))]

    try:
        temp_ind['MainSCOREballs'] = temp_ind['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
        temp_ind['MainSCOREballs'] = temp_ind['MainSCOREballs'].astype(float)
    except Exception as e:
        print(e)

    try:
        temp['MainSCOREballs'] = temp['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
        temp['MainSCOREballs'] = temp['MainSCOREballs'].astype(float)
    except Exception as e:
        print(e)

    if cur_scorecard in scorecards_by_python:
        temp_ind['MainSCOREballs'] = temp_ind['MainSCOREballs'].map(lambda x: round(x, 2))
    else:
        temp_ind['MainSCOREballs'] = temp_ind['MainSCOREballs'].map(lambda x: round(x, -1))

    if cur_scorecard in scorecards_by_python:
        temp['MainSCOREballs'] = temp['MainSCOREballs'].map(lambda x: round(x, 2))
    else:
        temp['MainSCOREballs'] = temp['MainSCOREballs'].map(lambda x: round(x, -1))
    
    indicator = '30+ 3MoB'
    temp_ind['indicator'] = temp_ind[indicator].map(lambda x: 1 if x > 0 else 0)

    temp_1 = temp_ind[temp_ind['indicator'] == 1]
    temp_0 = temp_ind[temp_ind['indicator'] == 0]

    json_scores_share = []

    for dfi in zip([temp, temp_1, temp_0], ['score_all_applications', 'score_indicator_1', 'score_indicator_0']):
    # for dfi in zip([temp_1, temp_0], ['score_indicator_1', 'score_indicator_0']):

        label = dfi[1]
        temp = dfi[0].groupby(['MainSCOREballs']).agg({'MainSCOREballs': 'count'})
        temp['MainSCOREballs'] = temp['MainSCOREballs'] / temp['MainSCOREballs'].sum()

        print(temp)

        for index in temp.index:
            json_scores_share.append({'score': index, 'key': label, 'share': temp.loc[index, 'MainSCOREballs']})

        del temp
    
    # print(json_scores_share) 
    # print(temp.head())
    # print(temp_1.head())
    # print(temp_0.head())
    # print(temp['indicator'].value_counts())


    ################################################################
    #                    VARIABLES  EFFECTIVENESS                  #
    ################################################################

    n_months = 100

    scorecard_variables = OrderedDict()

    req = sr.scorecard_variables.replace('TABLE1', table_name % table_date)

    CUR_CLIENT_TYPE = 'New' if cur_client_type == 'Новый' else 'Repeat'
    req = req.replace("SCORECARD1", cur_scorecard).replace("BUSINESS_TYPE1", cur_business_type).replace("CLIENT_TYPE1", CUR_CLIENT_TYPE)
    try:
        res = execute_db(c.db_d, req, pr=True)
        logger.info(cur_log + '\n' + req + '\n')
        variables = pd.DataFrame(res[1:], columns=res[0])
        variables.columns = [x.lower() for x in variables.columns]

        variables = variables.loc[:, ['month', 'variable', 'point', 'allappl', 'finappl', 
            'allshare', 'finshare', 'finrate', 'indicator', 'indicatortype']]

        if not variables.empty:

            # check coef=0 group
            for month in list(variables['month'].unique()):
                temp = variables[variables['month'] == month]
                for var in list(temp['variable'].unique()):
                    if 0.0 not in list(temp[temp['variable'] == var]['point']):
                        print(var, 'no null here')
                        variables = variables.append(pd.DataFrame([[month, var] + [0]*7 + [temp['indicatortype'].values[0]]], columns=list(variables.columns)))

            variables = variables.sort_values(by=['variable', 'point'])
            variables.index = range(len(variables))


            variables.fillna(0, inplace=True)
            for col in ['allappl', 'finappl']:
                variables[col] = variables[col].astype(int)
            variables = variables.round({k:3 for k in ['allshare', 'finshare', 'finrate', 'indicator']})
            
            dates = sorted(list(variables['month'].unique()), reverse=True)
            dates = dates[:min(n_months, variables['month'].nunique())]

            for date in dates:
                # scorecard_variables[date] = variables[variables['month'] == date].loc[:, 'variable':].to_html(index=False)
                indicator_name = variables[variables['month'] == date]['indicatortype'].values[0]
                temp = variables[variables['month'] == date].loc[:, 'variable':]
                temp.rename(columns={'indicator': indicator_name}, inplace=True)
                temp.drop(['indicatortype'], axis=1, inplace=True)
                scorecard_variables[date] = temp.to_html(index=False)
                del temp

            print(variables.head(10))
        else:
            # scorecard_variables = ''
            pass
    except Exception as e:
        print(e)
        # scorecard_variables = ''
        pass
    
    ################################################################
    #                     CURRENT SCORECARDS                       #
    ################################################################

    scorecards_db = Scorecard.objects.filter(working=True).order_by('business_type', 'client_type')
    # scorecards_db = Scorecard.objects.all()
    scorecards_db_list = []
    for el in scorecards_db:
        scorecards_db_list.append([str(el.business_type), str(el.client_type), str(el.name)])
        # print(str(el.business_type), str(el.client_type), str(el.name))

    ################################################################
    #                     DELETE OLD FILES                         #
    ################################################################

    old_file = temp_files + sm_base + 'scorecards_data/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + datetime.strftime(prev_week_end - timedelta(days=7), "%Y_%m_%d") + '.csv'
    try:
        os.remove(old_file)  # delete main data for risk groups
    except: pass

    old_file = temp_files + sm_base + 'scorecards_list/' + 'scorecards_' + datetime.strftime(prev_week_end - timedelta(days=7), "%Y_%m_%d") + '.csv'
    try:
        os.remove(old_file)  # delete csv scorecards list
    except: pass

    old_file = temp_files + sm_base + 'scorecards_list/' + 'scorecards_by_python_' + datetime.strftime(prev_week_end - timedelta(days=7), "%Y_%m_%d") + '.pkl'
    try:
        os.remove(old_file)  # delete pkl scorecards by python list
    except: pass

    months_ago = 1
    old_file_1 = temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + str(now.year) + '-' + '0'*(2 - len(str(now.month - months_ago))) + str(now.month - months_ago) + '_scores.csv'
    old_file_2 = temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + str(now.year) + '-' + '0'*(2 - len(str(now.month - months_ago))) + str(now.month - months_ago) + '_scores_calc.xlsx'
    for file in [old_file_1, old_file_2]:
        try:
            os.remove(file)  # delete old scorecard's scores bins
        except: pass

    # delete old gini (keep only two last files)
    date_old = str(now.year) + '-' +  '0'*(2 - len(str(now.month - (months_ago_gini + 1) - 1))) + str(now.month - (months_ago_gini + 1) - 1)
    old_file = temp_files + sm_base + 'scorecards_gini/' + 'gini_basis_' + date_old + '.csv'
    # print(old_file)
    try:
        os.remove(old_file)  # delete
    except: pass

    # delete logs
    all_log_files = os.listdir(temp_files + sm_base + 'log/')
    sec_in_day = 86400
    life_time = 14  # days

    for path in all_log_files:
        f_time = os.stat(os.path.join(temp_files + sm_base + 'log/', path)).st_mtime
        now = time.time()
        if f_time < now - life_time * sec_in_day:
            try:
                os.remove(temp_files + sm_base + 'log/' + path)
            except: pass

    # dirty delete
    for root_path in ['scorecards_data', 'scorecards_list', 'scorecards_dev_bins']:
        all_log_files = os.listdir(temp_files + sm_base + root_path + '/')
        sec_in_day = 86400
        life_time = 45  # days

        for path in all_log_files:
            f_time = os.stat(os.path.join(temp_files + sm_base + root_path + '/', path)).st_mtime
            now = time.time()
            if f_time < now - life_time * sec_in_day:
                try:
                    os.remove(temp_files + sm_base + root_path + '/' + path)
                except: pass


    # --------------------------------------------------------------

    logger.removeHandler(fh)

    print('------------- end --------------', cur_business_type, cur_client_type, cur_scorecard)

    if requests_out:
        with open(temp_files + sm_base + '/requests.pkl', 'wb') as file:
            pickle.dump(requests_out, file)

    ################################################################
    #                         FILTER DATA                          #
    ################################################################

    json_mean_indicators = [x for x in json_mean_indicators if x['value'] > 0 or x['date'] == min_date]


    # n_last = 1
    # json_gini = json_gini[:-n_last]
    # json_data_f = json_data_f[:-n_last]
    # json_data_all = json_data_all[:-n_last]
    # json_mean_scores = json_mean_scores[:-n_last]
    # json_mean_indicators = json_mean_indicators[:-n_last]
    # json_scores_share = json_scores_share[:-n_last]
    # json_scores = json_scores[:-n_last]
    

    # --------------------------------------------------------------

    return render(request, 'scorecard_monitoring.html', {
        'requests_out': requests_out,

        'json_data_f': json.dumps(json_data_f), 
        'json_gini': json.dumps(json_gini), 
        'json_data_all': json.dumps(json_data_all),
        'json_scores': json.dumps(json_scores), 
        'json_mean_scores': json.dumps(json_mean_scores), 
        'json_mean_indicators': json.dumps(json_mean_indicators),

        'json_scores_share': json.dumps(json_scores_share),

        'cur_scorecard': cur_scorecard, 
        'cur_business_type': cur_business_type, 
        'cur_client_type': cur_client_type, 
        'scorecards_db': scorecards_db, 
        'scorecards_db_list': scorecards_db_list, 
        'date_base': date_base,

        'scorecard_variables': scorecard_variables
        })


@login_required
def scorecard_monitoring(request):

    save_event(request)

    if request.POST:
        cur_scorecard = request.POST['scorecard']
        cur_business_type = request.POST['business_type']
        cur_client_type = request.POST['client_type'].rsplit('_', 1)[1]

        print(request.POST['client_type'], cur_client_type)
    else:

        cur_scorecard_obj = Scorecard.objects.filter(name='dn_pd_eq_1217')[0] # MainScore_201709_Core_New_EquifaxHit dn_pd_eq_1217
        cur_scorecard = str(cur_scorecard_obj.name)
        cur_business_type = str(cur_scorecard_obj.business_type)
        cur_client_type = str(cur_scorecard_obj.client_type)

        # print(type(cur_client_type))
        # print('Новый', cur_client_type, 'Новый' == cur_client_type) #, len('Новый'), len(cur_client_type))

    print('---------------------', cur_business_type, cur_client_type, cur_scorecard)

    now = datetime.now()
    # init logging
    logger = logging.getLogger("log")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=temp_files + '/scorecard_monitoring/log/' + datetime.strftime(now, '%Y_%m_%d') + ".log")  # create the logging file handler
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add handler to logger object
    # logger.info()
    # logger.removeHandler(fh)

    cur_log = datetime.strftime(now, '%H_%M_%S')

    # cur_week_start = now - timedelta(days=(now.weekday() + 1))
    # cur_week_end = cur_week_start + timedelta(days=6)

    prev_week_end = now - timedelta(days=(now.weekday() + 1))
    # print(prev_week_end)

    ################################################################
    #                          INIT DIRS                           #
    ################################################################

    sm_base = "/scorecard_monitoring/"

    ################################################################
    #                       SCORECARDS LIST                        #
    ################################################################
    
    min_appl_count = 100
    cur_file = 'scorecards_' + datetime.strftime(prev_week_end, "%Y_%m_%d") + '.csv'

    if cur_file in os.listdir(temp_files + sm_base + 'scorecards_list/'):
        scorecards = pd.read_csv(temp_files + sm_base + 'scorecards_list/' + cur_file, encoding='cp1251', sep='\t')
        logger.info(cur_log + ' scorecards list loaded from ' + cur_file)
   
    else:

        # take 4 weeks ago
        weeks_to_past = 4  # weeks count for option selection
        dt = datetime.now()
        cur_week_start = dt - timedelta(days=(dt.weekday() + 7))
        cur_week_end = cur_week_start + timedelta(days=6)
        weeks_ago = []
        for i in range(weeks_to_past):
            week_start = cur_week_start - timedelta(days=7)
            week_end = week_start + timedelta(days=6)
            cur_week_start = week_start
            weeks_ago.append("'"+ datetime.strftime(week_start, "%Y-%m-%d") + ' --- ' + datetime.strftime(week_end, "%Y-%m-%d") + "'")
        
        req = sr.scorecards_list_per_weeks
        req = req.replace("WEEKS1", ','.join(weeks_ago)).replace("COUNT1", str(min_appl_count))
        res = execute_db(c.db_p, req, pr=True)
        logger.info(cur_log + '\n' + req + '\n')
        scorecards = pd.DataFrame(res[1:], columns=res[0])

        # scorecards_db = Scorecard.objects.filter(working=True)
        if not scorecards.empty:
            scorecards.to_csv(temp_files + sm_base + 'scorecards_list/' + cur_file, encoding='cp1251', sep='\t', index=False)
    
    print(scorecards)

    ################################################################
    #                      UPDATE SCORECARDS                       #
    ################################################################

    for i in range(len(scorecards)):
        check_scorecard = Scorecard.objects.filter(name=scorecards.loc[i, 'SCORECARD'], business_type=scorecards.loc[i, 'BUSINESS_TYPE'], 
           client_type=scorecards.loc[i, 'CLIENT_TYPE'])
        if check_scorecard:
            pass
        else:
            new_scorecard = Scorecard(name=scorecards.loc[i, 'SCORECARD'], business_type=scorecards.loc[i, 'BUSINESS_TYPE'], 
            client_type=scorecards.loc[i, 'CLIENT_TYPE'])
            new_scorecard.working = True
            new_scorecard.save()

        # set-off scorecards
        for scorecard_db in Scorecard.objects.all().order_by('business_type', 'client_type'):
            if scorecards[(scorecards['SCORECARD'] == str(scorecard_db.name)) & (scorecards['CLIENT_TYPE'] == str(scorecard_db.client_type)) 
            & (scorecards['BUSINESS_TYPE'] == str(scorecard_db.business_type))].empty:
                scorecard_db.working = False
                scorecard_db.save()


    # scorecards by python
    cur_file = 'scorecards_by_python_' + datetime.strftime(prev_week_end, "%Y_%m_%d") + '.pkl'

    if cur_file in os.listdir(temp_files + sm_base + 'scorecards_list/'):
        
        with open(temp_files + sm_base + 'scorecards_list/' + cur_file, 'rb') as file:
            scorecards_by_python = pickle.load(file)   
        logger.info(cur_log + ' scorecards by python list loaded from ' + cur_file)

    else:
        req = sr.scorecards_by_python
        res = execute_db(c.db_p, req, pr=True)[1:]
        logger.info(cur_log + '\n' + req + '\n')
        if res:
            scorecards_by_python = [el[0] for el in res]
        else:
            scorecards_by_python = []

        if scorecards_by_python:
            with open(temp_files + sm_base + 'scorecards_list/' + cur_file, 'wb') as file:
                pickle.dump(scorecards_by_python, file)
    
    # print(scorecards_by_python)

    ################################################################
    #                           GINI                               #
    ################################################################
    
    min_appl_count = 100
    months_ago = 4
    date_for_indicator = str(now.year) + '-' +  '0'*(2 - len(str(now.month - months_ago))) + str(now.month - months_ago)

    cur_file = 'gini_basis_' + date_for_indicator + '.csv'
    if cur_file not in os.listdir(temp_files + '/scorecard_gini/'):
        return redirect('/gini_badrate/')  # Let's calculate all gini
    else:
        gini = pd.read_csv(temp_files + '/scorecard_gini/gini_basis_' + date_for_indicator + '.csv', encoding='cp1251', sep='\t')

    d = {'Новый': 'New', 'Повторный': 'Repeat'}
    gini = gini[(gini['BUSINESS_TYPE'] == cur_business_type) & (gini['SCORECARD'] == cur_scorecard) & (gini['CLIENT_TYPE'] == d[cur_client_type])]

    gini.index = range(len(gini))

    # print(gini)
    json_gini = []

    for i in range(len(gini)):
        json_gini.append({ 'date': gini.loc[i, 'MONTH'], 'value': round(gini.loc[i, 'gini'], 3)})
    
    # print(json_gini)

    ################################################################
    #                        RISK GROUPS                           #
    ################################################################

    cur_file = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + datetime.strftime(prev_week_end, "%Y_%m_%d") + '.csv'

    if cur_file in os.listdir(temp_files + sm_base + 'scorecards_data/'):

        file_size = os.stat(temp_files + sm_base + 'scorecards_data/' + cur_file).st_size / (1024 ** 2)  # Get file size
        if file_size > 50:  # If file size more than threshold
            data_main = pd.DataFrame()  # Create result data frame
            chunks = pd.read_table(temp_files + sm_base + 'scorecards_data/' + cur_file, chunksize=1000, iterator=True, encoding='cp1251', sep='\t')
            for chunk in chunks:  # For each part in parts
                data_main = pd.concat([data_main, chunk], axis=0)  # Join file parts

        else:
            data_main = pd.read_csv(temp_files + sm_base + 'scorecards_data/' + cur_file, encoding='cp1251', sep='\t')
        logger.info(cur_log + ' data for ' + cur_scorecard +' loaded from ' + cur_file)
   
    else:

        # cur_business_type = scorecards[scorecards['SCORECARD'] == cur_scorecard]['BUSINESS_TYPE'].values[0]
        # cur_client_type = scorecards[scorecards['SCORECARD'] == cur_scorecard]['CLIENT_TYPE'].values[0]

        # Init business type
        if str(cur_business_type) == 'Core':
            cur_business_type_f = "Core', 'from_SPR7_to_SPR4', 'from_SPR15_to_SPR4"
        else:
            cur_business_type_f = str(cur_business_type)

        # Init score field
        if cur_scorecard in scorecards_by_python:
            cur_score = 'MainPROBABballs as MainSCOREballs'  # MainSCOREballs, MainPROBABballs,
        else:
            cur_score = 'MainSCOREballs' 

        cond1 = '1=1'  # special condition

        req = sr.scorecards_rg_indicators
        req = req.replace("SCORECARD1", str(cur_scorecard)).replace("BUSINESS_TYPE1", cur_business_type_f).replace("CLIENT_TYPE1", str(cur_client_type)).replace("COND1", cond1).replace("SCORE1", cur_score)
        res = execute_db(c.db_p, req, pr=True)
        logger.info(cur_log + '\n' + req + '\n')
        data_main = pd.DataFrame(res[1:], columns=res[0])
        if not data_main.empty:
            data_main.to_csv(temp_files + sm_base + 'scorecards_data/' + cur_file, encoding='cp1251', sep='\t', index=False)

    
    # print(data.head())
    # try:
    #     data_main['MainSCOREballs'] = data_main['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
    #     data_main = data_main.replace({'MainSCOREballs': {'NULL': -1}})
    #     data_main['MainSCOREballs'] = data_main['MainSCOREballs'].astype(float)
    # except Exception as e:
    #     print(e)

    # data_main['MainSCOREballs'].fillna(-1, inplace=True)
    # data_main = data_main.replace({'MainSCOREballs': {'NULL': -1}})

    # data_main['MainSCOREballs'] = data_main['MainSCOREballs'].astype(float)

    ################################################################
    #                       FINANCED PART                          #
    ################################################################

    data_f = data_main[data_main['Financed'] == 1].copy()
    data_f_g = data_f.groupby(['CRM_CREATION_Month', 'RISK_GROUP']).agg({'UCDB_ID': ['count']})
    data_f = data_f_g.transpose().reset_index(level=0, drop=True).transpose().reset_index()
    data_f.rename(columns={'CRM_CREATION_Month': 'date'}, inplace=True)
    
    # print(data_f)

    # json_data = [
    #       { 'date': "2018-01", 'A': 60, 'B': 15, 'C': 9, 'D': 6},
    # ]

    # json_gini = [
    #       { 'date': "2018-01", 'value': 0.65 },
    # ]

    json_data_f = []

    for date in sorted(data_f['date'].unique()):
        temp = data_f[data_f['date'] == date]
        lst = {'date': date}
        for el in zip(temp['RISK_GROUP'], temp['count']):
            lst[el[0]] = el[1]
            for rg in list('ABCD'):
                if rg not in lst.keys():
                    lst[rg] = 0
        json_data_f.append(lst)

    # print(json_data_f)

    ################################################################
    #                      ALL APPLICATIONS                        #
    ################################################################

    data_all = data_main.copy()
    data_all_g = data_all.groupby(['CRM_CREATION_Month', 'RISK_GROUP']).agg({'UCDB_ID': ['count']})
    data_all = data_all_g.transpose().reset_index(level=0, drop=True).transpose().reset_index()
    data_all.rename(columns={'CRM_CREATION_Month': 'date'}, inplace=True)
    
    # print(data_all.head())
    json_data_all = []

    for date in sorted(data_all['date'].unique()):
        temp = data_all[data_all['date'] == date]
        lst = {'date': date}
        for el in zip(temp['RISK_GROUP'], temp['count']):
            lst[el[0]] = el[1]
            for rg in list('ABCD'):
                if rg not in lst.keys():
                    lst[rg] = 0
        json_data_all.append(lst)

    # print(json_data_f)

    ################################################################
    #                     SCORE STABILITY GROUP                    #
    ################################################################

    #  -------------------- v.1 (training qcut) --------------------

    # cur_file = cur_scorecard + '_bins.pkl'

    # # load bins
    # if cur_file in os.listdir(temp_files + sm_base + 'scorecards_dev_bins/'):
    #     with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'rb') as file:
    #         bins = pickle.load(file)
    #     logger.info(cur_log + ' bins for ' + cur_scorecard +' loaded from ' + cur_file)

    # else:
    #     scores_file = cur_scorecard + '_scores.xlsx'
    #     if scores_file in os.listdir(temp_files + sm_base + 'scorecards_dev_scores/'):
            
    #         df = pd.read_excel(temp_files + sm_base + 'scorecards_dev_scores/' + scores_file)
    #         print(df.head())
    #         _, bins = pd.qcut(df['Prop'], 10, retbins=True)
    #         bins = list(bins)
    #         bins[0] = -np.inf
    #         bins[-1] = np.inf
    #         df['dev'] = pd.cut(df['Prop'], bins=bins)
    #         # get dev scores
    #         data = df['dev'].value_counts(normalize=True).sort_index().to_frame()  
    #         data.index.rename('group', inplace=True)
    #         data.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_dev_scores.csv', sep='\t', encoding='cp1251')

    #         with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'wb') as file:
    #             pickle.dump(bins, file) 
    #     else:
    #         bins = []

    # if bins:
    #     bins[0] = -np.inf
    #     bins[-1] = np.inf

    # data_bins = pd.read_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_dev_scores.csv', sep='\t', encoding='cp1251', index_col='group')
    
    # # print(data_bins)

    # dates_ss = []

    # # data_f = data_main[data_main['Financed'] == 1].copy()
    # data_f = data_main.copy()

    # for date in sorted(list(data_f['CRM_CREATION_Month'].unique())):

    #     temp = data_f[data_f['CRM_CREATION_Month'] == date]
    #     temp['N'] = pd.cut(temp['MainSCOREballs'], bins=bins)
    #     # get dev scores
    #     data_bins[date] = temp['N'].value_counts(normalize=True).sort_index().values
    #     dates_ss.append(date)

    # print(data_bins)
    # if not data_bins.empty:
    #     data_bins.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + max(dates_ss) + '_scores.csv', sep='\t', encoding='cp1251')

    # json_scores = []
    # for date in dates_ss:
    #     data_bins[date+'_d'] = data_bins[date]-data_bins['dev']
    #     data_bins[date+'_w'] = np.log(data_bins[date]/data_bins['dev'])
    #     data_bins = data_bins.replace({date+'_w': {np.inf: 0, -np.inf: 0}})
    #     data_bins[date+'_s'] = data_bins[date+'_d']*data_bins[date+'_w'] 
    #     ssi = data_bins[date+'_s'].sum()
    #     json_scores.append({'date': date, 'value': round(ssi, 5)})

    # print(json_scores)
    # print(data_bins)
    # data_bins.to_excel(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + max(dates_ss) + '_scores_calc.xlsx')


    #  ----------------------- v.2 (nth week qcut)  ---------------

    from_month_number = 2

    # data_f = data_main[data_main['Financed'] == 1].copy()
    data_ss = data_main.copy()
    # data_ss = data_ss[(~data_ss['MainSCOREballs'].isin(['NULL']))]
    data_ss = data_ss[(data_ss['MainSCOREballs'].notnull()) & (~data_ss['MainSCOREballs'].isin(['NULL']))]

    try:
        data_ss['MainSCOREballs'] = data_ss['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
        data_ss['MainSCOREballs'] = data_ss['MainSCOREballs'].astype(float)
    except Exception as e:
        print(e)

    all_dates = sorted(list(data_ss['CRM_CREATION_Month'].unique()))

    # check scorecard's work 
    if len(all_dates) > 0:

        try:
            date_base = all_dates[from_month_number - 1]
        except:
            date_base = all_dates[0]

        
        cur_file = cur_scorecard + '_from_'+ str(date_base) +'_month_bins.pkl'

        # load bins
        if cur_file in os.listdir(temp_files + sm_base + 'scorecards_dev_bins/'):
            with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'rb') as file:
                bins = pickle.load(file)
            logger.info(cur_log + ' bins for ' + cur_scorecard +' loaded from ' + cur_file)

        else:

            df = data_ss[(data_ss['CRM_CREATION_Month'] == date_base)]
        
            _, bins = pd.qcut(df['MainSCOREballs'], 10, retbins=True, duplicates='drop')
            bins = list(bins)
            bins[0] = -np.inf
            bins[-1] = np.inf
            df[date_base + '_dev'] = pd.cut(df['MainSCOREballs'], bins=bins)
            # get dev scores
            data = df[date_base + '_dev'].value_counts(normalize=True).sort_index().to_frame()  
            data.index.rename('group', inplace=True)
            data.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_from_'+ str(date_base) +'_month_dev_scores.csv', sep='\t', encoding='cp1251')

            with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'wb') as file:
                pickle.dump(bins, file) 

        if bins:
            bins[0] = -np.inf
            bins[-1] = np.inf

        data_bins = pd.read_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_from_'+ str(date_base) +'_month_dev_scores.csv', sep='\t', encoding='cp1251', index_col='group')
        
        # print(data_bins)

        dates_ss = all_dates[(all_dates.index(date_base) + 1):]

        for date in dates_ss:

            temp = data_ss[data_ss['CRM_CREATION_Month'] == date]
            temp['N'] = pd.cut(temp['MainSCOREballs'], bins=bins)
            # get dev scores
            data_bins[date] = temp['N'].value_counts(normalize=True).sort_index().values

        print(data_bins)
        if not data_bins.empty:
            data_bins.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + max(dates_ss) + '_scores.csv', sep='\t', encoding='cp1251')

        json_scores = []
        for date in dates_ss:
            data_bins[date+'_d'] = data_bins[date]-data_bins[date_base + '_dev']
            data_bins[date+'_w'] = np.log(data_bins[date]/data_bins[date_base + '_dev'])
            data_bins = data_bins.replace({date+'_w': {np.inf: 0, -np.inf: 0}})
            data_bins[date+'_s'] = data_bins[date+'_d']*data_bins[date+'_w'] 
            ssi = data_bins[date+'_s'].sum()
            json_scores.append({'date': date, 'value': round(ssi, 5)})

        print(json_scores)
        print(data_bins)
        data_bins.to_excel(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + max(dates_ss) + '_scores_calc.xlsx')

    # scorecard too new
    else:
        json_scores = []


    ################################################################
    #                          MEAN VALUES                         #
    ################################################################

    json_mean_scores = []
    json_mean_indicators = []

    indicators = ['3+ 4 *2WoB', '30+ 3MoB']

    min_date = data_main['CRM_CREATION_Month'].min()  # for filtering in the end

    for date in sorted(list(data_main['CRM_CREATION_Month'].unique())):

        temp = data_main[data_main['CRM_CREATION_Month'] == date].copy()
        temp = temp[(temp['MainSCOREballs'].notnull()) & (~temp['MainSCOREballs'].isin(['NULL']))]

        try:
            temp['MainSCOREballs'] = temp['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
            temp['MainSCOREballs'] = temp['MainSCOREballs'].astype(float)
        except Exception as e:
            print(e)


        score_mean = temp['MainSCOREballs'].mean() if len(temp) > 0 else 0.0
        json_mean_scores.append({"date": date, "score_mean": score_mean})
        
        # FPD
        fpd_mean = temp[temp['FPD'].notnull()]['FPD'].mean() if len(temp[temp['FPD'].notnull()]) > 0 else 0
        json_mean_indicators.append({"date": date, "label": 'fpd', "value": round(fpd_mean, 5)})


        for indicator in indicators:
            temp = temp[temp[indicator].notnull()]
            temp['ind_' + indicator] = temp[indicator] / (temp['LOAN_AMOUNT'] + temp['LOAN_COMISSION'])
            indicator_value = temp[temp["ind_" + indicator].notnull()]["ind_" + indicator].mean() if len(temp[temp["ind_" + indicator].notnull()]) > 0 else 0
            json_mean_indicators.append({"date": date, "label": indicator, "value": round(indicator_value, 5)})

        del temp
    print(json_mean_scores)

    ################################################################
    #                     CURRENT SCORECARDS                       #
    ################################################################

    scorecards_db = Scorecard.objects.filter(working=True).order_by('business_type', 'client_type')
    # scorecards_db = Scorecard.objects.all()
    scorecards_db_list = []
    for el in scorecards_db:
        scorecards_db_list.append([str(el.business_type), str(el.client_type), str(el.name)])
        print(str(el.business_type), str(el.client_type), str(el.name))

    ################################################################
    #                     DELETE OLD FILES                         #
    ################################################################

    old_file = temp_files + sm_base + 'scorecards_data/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + datetime.strftime(prev_week_end - timedelta(days=7), "%Y_%m_%d") + '.csv'
    try:
        os.remove(old_file)  # delete main data for risk groups
    except: pass

    old_file = temp_files + sm_base + 'scorecards_list/' + 'scorecards_' + datetime.strftime(prev_week_end - timedelta(days=7), "%Y_%m_%d") + '.csv'
    try:
        os.remove(old_file)  # delete csv scorecards list
    except: pass

    old_file = temp_files + sm_base + 'scorecards_list/' + 'scorecards_by_python_' + datetime.strftime(prev_week_end - timedelta(days=7), "%Y_%m_%d") + '.pkl'
    try:
        os.remove(old_file)  # delete pkl scorecards by python list
    except: pass

    months_ago = 1
    old_file_1 = temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + str(now.year) + '-' + '0'*(2 - len(str(now.month - months_ago))) + str(now.month - months_ago) + '_scores.csv'
    old_file_2 = temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + str(now.year) + '-' + '0'*(2 - len(str(now.month - months_ago))) + str(now.month - months_ago) + '_scores_calc.xlsx'
    for file in [old_file_1, old_file_2]:
        try:
            os.remove(file)  # delete old scorecard's scores bins
        except: pass

    # delete logs
    all_log_files = os.listdir(temp_files + sm_base + 'log/')
    sec_in_day = 86400
    life_time = 14  # days

    for path in all_log_files:
        f_time = os.stat(os.path.join(temp_files + sm_base + 'log/', path)).st_mtime
        now = time.time()
        if f_time < now - life_time * sec_in_day:
            try:
                os.remove(temp_files + sm_base + 'log/' + path)
            except: pass


    # --------------------------------------------------------------

    logger.removeHandler(fh)

    print('------------- end -----------------', cur_business_type, cur_client_type, cur_scorecard)


    ################################################################
    #                         FILTER DATA                          #
    ################################################################

    # json_gini = json_gini[:-1]
    json_mean_indicators = [x for x in json_mean_indicators if x['value'] > 0 or x['date'] == min_date]

    # --------------------------------------------------------------

    return render(request, 'scorecard_monitoring.html', {'json_data_f': json.dumps(json_data_f), 'json_gini': json.dumps(json_gini), 'json_data_all': json.dumps(json_data_all), 'json_scores': json.dumps(json_scores), 'json_mean_scores': json.dumps(json_mean_scores), 'json_mean_indicators': json.dumps(json_mean_indicators),  
        'cur_scorecard': cur_scorecard, 'cur_business_type': cur_business_type, 'cur_client_type': cur_client_type, 'scorecards_db': scorecards_db, 'scorecards_db_list': scorecards_db_list, 'date_base': date_base
        })



def scorecards_per_month(request):

    month = datetime.strftime(datetime.now(), '%Y-%m')
    req = sr.scorecards_per_month.replace("MONTH1", str(month))
    data = execute_db(c.db_p, req, pr=True)
    th = data[0]
    scorecards = data[1:]

    # return render_to_response('month.html', {'title': 'Monthly', 'th': th, 'scorecards': scorecards}, RequestContext(request))
    return render(request, 'month.html', {'title': 'Monthly', 'th': th, 'scorecards': scorecards})


def one_scorecard_counts(request):

    if request.POST:
        cur_scorecard = str(request.POST['scorecard'])
        date_from = str(request.POST['date_from'])
        date_upto = str(request.POST['date_upto'])
    else:
        cur_scorecard = 'dn_cl_eq_1217'
        date_from = '2016-07'
        date_upto = '2018-07'

    now = datetime(datetime.now().year, datetime.now().month, 1)
    ctr = datetime(2015, 1, 1)
    dates = [ctr.strftime('%Y-%m')]
    while ctr <= now:
        ctr += timedelta(days=32)
        dates.append( datetime(ctr.year, ctr.month, 1).strftime('%Y-%m'))

    # descending ordering
    dates = dates[::-1]

    # print(request.method, cur_scorecard, date_from, date_upto, '<<<<<<<<<<<<<<')

    # Checking cache
    cur_file = cur_scorecard + '_months_' + date_from + '_' + date_upto + '.json'
    # If the file exists load it
    if cur_file in os.listdir(temp_files + '/scorecard_counts/'):
        
        # Load json 
        with open(temp_files + '/scorecard_counts/' + cur_file) as infile:
            json_data  = json.load(infile)

        print(cur_file, 'was loaded')

    # If the file doesn't exist make request to db       
    else:
        req = sr.one_scorecard_counts
        req = req.replace("SCORECARD1", cur_scorecard).replace("DATE_FROM", date_from).replace("DATE_UPTO", date_upto)
        data = execute_db(c.db_p, req, pr=True)[1:]

        if data:

            json_data = [{'date': el[0], 'count': el[1]} for el in data if el[0] is not None]

            # Save json
            with open(temp_files + '/scorecard_counts/' + cur_file, 'w') as outfile:
                json.dump(json_data, outfile)

        else:
            json_data = []

    
    scorecard_options = [
     'MainScore_201606_NewClients_EquifaxHit',
     'MainScore_201709_Digital_New_EquifaxHit', 
     'MainScore_201609_RepeatClients', 
     'dn_cl_eq_1217']

    return render(request, 'one_scorecard_counts.html', {'json': json.dumps(json_data), 'cur_scorecard': cur_scorecard, 
        'scorecard_options': scorecard_options, 'cur_date_from': date_from, 'cur_date_upto': date_upto, 'dates': dates})




def gini_example(request):

    json_example = [

        {"station_id":27,"variable":"yF2","date":"2018-02-11T19:00:00.000","value":53.05,"name":"Idaho Natl Lab, ID, USA","region_id":4},
        {"station_id":27,"variable":"MUFD","date":"2018-02-11T20:00:00.000","value":20.9,"name":"Idaho Natl Lab, ID, USA","region_id":4},
      ]

    data = pd.read_csv(os.path.join(temp_files, 'gini.csv'), sep='\t', encoding='cp1251', decimal=',')
    data['business_type'] = 0
    # for scorecard in list(data['ScoreCard_Name'].unique()):
    #     if len(data[data['ScoreCard_Name'] == scorecard]) == 1:
    #         data.drop(data[data['ScoreCard_Name'] == scorecard].index, inplace=True)

    data['client_type'] = data['Segment'].map(lambda x: 'new' if 'new' in x.lower() else 'repeat')
    # print(data)

    df = data.reset_index().melt(id_vars=['Month', 'Segment', 'ScoreCard_Name', 'business_type', 'client_type'], value_vars=['GINI', 'KS'])
    json_data = []

    for i in range(len(df)):
        json_data.append({"station_id": df.loc[i, 'Segment'], "variable": df.loc[i, 'variable'], "date": df.loc[i, 'Month'],
            "value": df.loc[i, 'value'], "name": df.loc[i, 'Segment'] + ': ' + df.loc[i, 'ScoreCard_Name'], 
            "business_type": df.loc[i, 'Segment'].split("_")[0].lower(),
            "client_type": df.loc[i, 'client_type'].lower()
            })

    # # Save json
    # cur_file = 'gini.json'
    # with open(temp_files + '/scorecard_gini/' + cur_file, 'w') as outfile:
    #     json.dump(json_data, outfile)

    return render_to_response('gini.html', {'gini_json': json.dumps(json_data)}, RequestContext(request))


@login_required
def gini_badrate(request):

    save_event(request)

    now = datetime.now()
    # init logging
    logger = logging.getLogger("log")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=temp_files + '/scorecard_gini/log/' + datetime.strftime(now, '%Y_%m_%d') + ".log")  # create the logging file handler
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add handler to logger object
    # logger.info()
    # logger.removeHandler(fh)

    cur_log = datetime.strftime(now, '%H_%M_%S')
 
    min_appl_count = 100
    months_ago = 4


    month1 = now.month - months_ago
    year1 = now.year
    if month1 <= 0:
        months_last_year = months_ago - now.month
        month1 = 12 - months_last_year
        year1 -= 1  # previous year

    date_for_indicator = str(year1) + '-' +  '0'*(2 - len(str(month1))) + str(month1)

    month2 = now.month - months_ago - 1
    year2 = now.year
    if month2 <= 0:
        months_last_year = months_ago + 1 - now.month
        month2 = 12 - months_last_year
        year2 -= 1  # previous year

    date_prev = str(year2) + '-' +  '0'*(2 - len(str(month2))) + str(month2)

    # date_for_indicator = str(now.year) + '-' +  '0'*(2 - len(str(now.month - months_ago))) + str(now.month - months_ago)
    # date_prev = str(now.year) + '-' +  '0'*(2 - len(str(now.month - months_ago - 1))) + str(now.month - months_ago - 1)
    # print(date_for_indicator)

    # Checking cache
    cur_file = 'gini' + '_2016-07' + '_' + date_for_indicator + '.json'
    # If the file exists load it
    if cur_file in os.listdir(temp_files + '/scorecard_gini/'):
        
        # Load json 
        with open(temp_files + '/scorecard_gini/' + cur_file) as infile:
            json_data  = json.load(infile)

        print(cur_file, 'was loaded')
        logger.info(cur_file + ' was loaded')

    # If the file doesn't exist make request to db       
    else:

        last_file = max([el for el in os.listdir(temp_files + '/scorecard_gini/') if el != 'log'])
        print(last_file)
        p = re.compile(r'\d_(.*).json')
        last_date = p.findall(cur_file)[-1]
        # last_date = cur_file.split('_')[-1].split('.')[0]
        print(last_date)

        req = sr.scorecards_list_per_month
        req = req.replace("MONTH1", date_for_indicator).replace("COUNT1", str(min_appl_count))
        data = execute_db(c.db_p, req, pr=True)

        logger.info(cur_log + '\n' + req + '\n')
        
        df = pd.DataFrame(data[1:], columns=data[0])
        df['gini'] = -1
        df['ks'] = -1

        print(df)

        # Get scorecards names by python
        data = execute_db(c.db_p, sr.scorecards_by_python, pr=True)[1:]
        if data:
            scorecards_by_python = [el[0] for el in data]
        else:
            scorecards_by_python = []

        if not df.empty:

            df.insert(0, 'MONTH', date_for_indicator)
            df.insert(3, 'BUREAU', np.nan)
            df.insert(4, 'SEGMENT', np.nan)

            for i in range(len(df)):
                CUR_SCORECARD = df.loc[i, 'SCORECARD']
                CUR_BUSINESS_TYPE = df.loc[i, 'BUSINESS_TYPE']
                CUR_CLIENT_TYPE = df.loc[i, 'CLIENT_TYPE']

                # Init business type
                if CUR_BUSINESS_TYPE == 'Core':
                    CUR_BUSINESS_TYPE = "Core', 'from_SPR7_to_SPR4', 'from_SPR15_to_SPR4"

                # Init bureau
                cond1 = "1=1"
                if 'equifaxhit' in CUR_SCORECARD.lower() or 'eq' in CUR_SCORECARD.lower():
                    cond1 = "Bureau = 'Equifax hit'"
                    df.loc[i, 'BUREAU'] = 'EquifaxHit'

                if 'nobureauhit' in CUR_SCORECARD.lower():
                    df.loc[i, 'BUREAU'] = 'NoBureauHit'

                if 'nbch' in CUR_SCORECARD.lower():
                    df.loc[i, 'BUREAU'] = 'NbchHit'

                # Init score field
                if CUR_SCORECARD in scorecards_by_python:
                    cur_score = 'MainPROBABballs as MainSCOREballs'  # MainSCOREballs, MainPROBABballs,
                else:
                    cur_score = 'MainSCOREballs' 

                req = sr.data_for_gini_badrate
                req = req.replace("MONTH1", date_for_indicator).replace("SCORECARD1", CUR_SCORECARD).replace("BUSINESS_TYPE1", CUR_BUSINESS_TYPE).replace("CLIENT_TYPE1", CUR_CLIENT_TYPE).replace("COND1", cond1).replace("SCORE1", cur_score)
                res = execute_db(c.db_p, req, pr=True)

                logger.info(cur_log + '\n' + req + '\n')

                temp = pd.DataFrame(res[1:], columns=res[0])
                temp['MainSCOREballs'] = temp['MainSCOREballs'].astype(float)
                print(temp.head())

                try:
                    gini = 100 * abs(2 * roc_auc_score(temp[temp['BadRate'].notnull()]['BadRate'], temp[temp['BadRate'].notnull()]['MainSCOREballs']) - 1)
                    ks = 100 * stats.ks_2samp(temp[temp['BadRate'] == 0]['MainSCOREballs'], temp[temp['BadRate'] == 1]['MainSCOREballs']).statistic   
                    df.loc[i, 'gini'] = gini
                    df.loc[i, 'ks'] = ks     
                except:
                    pass

            df = df.replace({'CLIENT_TYPE': {'Новый': 'New', 'Повторный': 'Repeat'}})

            print(df)
            logger.info(cur_log + '\n' + df.to_string() + '\n')

            df = df[df['gini'] != -1]  # filter by not null values
            # save all data in excel
            # data = pd.read_excel(temp_files + '/scorecard_gini/gini_basis.xlsx', encoding='cp1251')  # checkpoint from 2018-03
            data = pd.read_csv(temp_files + '/scorecard_gini/gini_basis_' + date_prev + '.csv', encoding='cp1251', sep='\t')
            data = pd.concat([data, df], axis=0)
            data.index = range(len(data))
            print(data)
            data.to_csv(temp_files + '/scorecard_gini/gini_basis_' + date_for_indicator + '.csv', encoding='cp1251', sep='\t', index=False)
     
            df = data.reset_index().melt(id_vars=['MONTH', 'segment', 'SCORECARD', 'BUSINESS_TYPE', 'CLIENT_TYPE', 'BUREAU'], value_vars=['gini', 'ks'])
            logger.info(cur_log + '\n' + df.to_string() + '\n')

            # # Load old json 
            # with open(temp_files + '/scorecard_gini/' + last_file) as infile:
            #     json_data  = json.load(infile)

            # print(last_file, 'was loaded')

            json_data = []

            for i in range(len(df)):
                
                CUR_CLIENT_TYPE = df.loc[i, 'CLIENT_TYPE']

                json_data.append({

                    # "station_id": df.loc[i, 'segment'] + '_' + df.loc[i, 'SCORECARD'].replace(' ', '_'),
                    "station_id": df.loc[i, 'BUSINESS_TYPE']  + '_' + CUR_CLIENT_TYPE + '_' + df.loc[i, 'SCORECARD'].replace(' ', '_'),    
                    "date": df.loc[i, 'MONTH'],
                    "variable": df.loc[i, 'variable'],
                    "value": df.loc[i, 'value'], 
                    #"name": df.loc[i, 'segment'] + ': ' + df.loc[i, 'SCORECARD'], 
                    "name": str(df.loc[i, 'BUSINESS_TYPE']) + '_' + CUR_CLIENT_TYPE + ': ' + str(df.loc[i, 'SCORECARD']), 
                    "business_type": str(df.loc[i, 'BUSINESS_TYPE']).lower(),
                    "client_type": CUR_CLIENT_TYPE.lower()
                })

            # Save json
            with open(temp_files + '/scorecard_gini/' + cur_file, 'w') as outfile:
                json.dump(json_data, outfile)

        else:
            json_data = []

        # # data = pd.read_csv(os.path.join(temp_files, 'all_ginis.csv'), sep='\t', encoding='cp1251', decimal=',')
        # data = pd.read_excel(os.path.join(temp_files, cur_file.split('.')[0] + '.xlsx'), encoding='cp1251', decimal=',')
        # data.fillna('', inplace=True)
        # # data = data[data['type'] == 'Big']
        
        # # for scorecard in list(data['ScoreCard_Name'].unique()):
        # #     if len(data[data['ScoreCard_Name'] == scorecard]) == 1:
        # #         data.drop(data[data['ScoreCard_Name'] == scorecard].index, inplace=True)

        # df = data.reset_index().melt(id_vars=['month', 'segment', 'scorecard', 'business_type', 'client_type', 'bureau'], value_vars=['gini', 'ks'])

        # json_data = []

        # for i in range(len(df)):
        #     json_data.append({
        #         # "station_id": df.loc[i, 'segment'] + '_' + df.loc[i, 'scorecard'].replace(' ', '_'),
        #         "station_id": df.loc[i, 'business_type']  + '_' + df.loc[i, 'client_type'] + '_' + df.loc[i, 'scorecard'].replace(' ', '_'),    
        #         "date": df.loc[i, 'month'],
        #         "variable": df.loc[i, 'variable'],
        #         "value": df.loc[i, 'value'], 
        #         #"name": df.loc[i, 'segment'] + ': ' + df.loc[i, 'scorecard'], 
        #         "name": str(df.loc[i, 'business_type']) + '_' + str(df.loc[i, 'client_type']) + ': ' + str(df.loc[i, 'scorecard']), 
        #         "business_type": str(df.loc[i, 'business_type']).lower(),
        #         "client_type": str(df.loc[i, 'client_type']).lower()
        #     })

        # # Save json
        # with open(temp_files + '/scorecard_gini/' + cur_file, 'w') as outfile:
        #     json.dump(json_data, outfile)

    # return render_to_response('gini.html', {'gini_json': json.dumps(json_data)}, RequestContext(request))

    ################################################################
    #                     DELETE OLD FILES                         #
    ################################################################

    # delete old gini (keep only 3 last files)
    date_old = str(now.year) + '-' +  '0'*(2 - len(str(now.month - (months_ago + 2) - 1))) + str(now.month - (months_ago + 2) - 1)
    old_file = temp_files + '/scorecard_gini/' + 'gini_basis_' + date_old + '.csv'
    try:
        os.remove(old_file)  # delete
    except: pass

    old_file = temp_files + '/scorecard_gini/' + 'gini_2016-07_' + date_old + '.json'
    try:
        os.remove(old_file)  # delete
    except: pass


    # delete logs
    all_log_files = os.listdir(temp_files + '/scorecard_gini/' + 'log/')
    sec_in_day = 86400
    life_time = 14  # days

    for path in all_log_files:
        f_time = os.stat(os.path.join(temp_files + '/scorecard_gini/' + 'log/', path)).st_mtime
        now = time.time()
        if f_time < now - life_time * sec_in_day:
            try:
                os.remove(temp_files + '/scorecard_gini/' + 'log/' + path)
            except: pass

    logger.removeHandler(fh)

    return render(request, 'gini.html', {'gini_json': json.dumps(json_data)})


def gini_fpd(request):

    now = datetime.now()
    min_appl_count = 100
    months_ago = 4

    cur_week_start = now - timedelta(days=(now.weekday() + 7))
    cur_week_end = cur_week_start + timedelta(days=6)

    date_for_indicator = str(now.year) + '-' +  '0'*(2 - len(str(now.month - months_ago))) + str(now.month - months_ago)
    # print(date_for_indicator)

    # Checking cache
    cur_file = 'gini' + '_2016-07' + '_' + date_for_indicator + '.json'
    # If the file exists load it
    if cur_file in os.listdir(temp_files + '/scorecard_gini/'):
        
        # Load json 
        with open(temp_files + '/scorecard_gini/' + cur_file) as infile:
            json_data  = json.load(infile)

        print(cur_file, 'was loaded')

    # If the file doesn't exist make request to db       
    else:

        last_file = max(os.listdir(temp_files + '/scorecard_gini/'))
        print(last_file)
        p = re.compile(r'\d_(.*).json')
        last_date = p.findall(cur_file)[-1]
        # last_date = cur_file.split('_')[-1].split('.')[0]
        print(last_date)

        req = sr.scorecards_list_per_week
        req = req.replace("WEEK1", date_for_indicator).replace("COUNT1", str(min_appl_count))
        data = execute_db(c.db_p, req, pr=True)

        df = pd.DataFrame(data[1:], columns=data[0])
        df['gini'] = -1
        df['ks'] = -1

        print(df)

        # Get scorecards names by python
        data = execute_db(c.db_p, sr.scorecards_by_python, pr=True)[1:]
        if data:
            scorecards_by_python = [el[0] for el in data]
        else:
            scorecards_by_python = []

        if not df.empty:

            for i in range(len(df)):
                CUR_SCORECARD = df.loc[i, 'SCORECARD']
                CUR_BUSINESS_TYPE = df.loc[i, 'BUSINESS_TYPE']
                CUR_CLIENT_TYPE = df.loc[i, 'CLIENT_TYPE']
                CUR_SCORECARD = df.loc[i, 'SCORECARD']

                # Init business type
                if CUR_BUSINESS_TYPE == 'Core':
                    CUR_BUSINESS_TYPE = "Core', 'from_SPR7_to_SPR4"

                # Init bureau
                cond1 = "1=1"
                if 'equifaxhit' in CUR_SCORECARD.lower() or 'eq' in CUR_SCORECARD.lower():
                    cond1 = "Bureau = 'Equifax hit'"
                    df.loc[i, 'BUREAU'] = 'EquifaxHit'

                if 'nobureauhit' in CUR_SCORECARD.lower():
                    df.loc[i, 'BUREAU'] = 'NoBureauHit'

                if 'nbch' in CUR_SCORECARD.lower():
                    df.loc[i, 'BUREAU'] = 'NbchHit'

                # Init score field
                if CUR_SCORECARD in scorecards_by_python:
                    cur_score = 'MainPROBABballs as MainSCOREballs'  # MainSCOREballs, MainPROBABballs,
                else:
                    cur_score = 'MainSCOREballs' 

                req = sr.data_for_gini_fpd
                req = req.replace("WEEK1", date_for_indicator).replace("SCORECARD1", CUR_SCORECARD).replace("BUSINESS_TYPE1", CUR_BUSINESS_TYPE).replace("CLIENT_TYPE1", CUR_CLIENT_TYPE).replace("COND1", cond1).replace("SCORE1", cur_score)
                res = execute_db(c.db_p, req, pr=True)

                temp = pd.DataFrame(res[1:], columns=res[0])
                temp['MainSCOREballs'] = temp['MainSCOREballs'].astype(float)
                print(temp.head())

                try:
                    gini = 100 * abs(2 * roc_auc_score(temp[temp['BadRate'].notnull()]['BadRate'], temp[temp['BadRate'].notnull()]['MainSCOREballs']) - 1)
                    ks = 100 * stats.ks_2samp(temp[temp['BadRate'] == 0]['MainSCOREballs'], temp[temp['BadRate'] == 1]['MainSCOREballs']).statistic   
                    df.loc[i, 'gini'] = gini
                    df.loc[i, 'ks'] = ks     
                except:
                    pass

                df['WEEK'] = date_for_indicator

            print(df)

            df = df[df['gini'] != -1]  # filter by not null values

            df = df.reset_index().melt(id_vars=['MONTH', 'segment', 'SCORECARD', 'BUSINESS_TYPE', 'CLIENT_TYPE', 'BUREAU'], value_vars=['gini', 'ks'])

            # Load json 
            with open(temp_files + '/scorecard_gini/' + last_file) as infile:
                json_data  = json.load(infile)

            print(last_file, 'was loaded')

            for i in range(len(df)):
                json_data.append({
                    # "station_id": df.loc[i, 'segment'] + '_' + df.loc[i, 'SCORECARD'].replace(' ', '_'),
                    "station_id": df.loc[i, 'BUSINESS_TYPE']  + '_' + df.loc[i, 'CLIENT_TYPE'] + '_' + df.loc[i, 'SCORECARD'].replace(' ', '_'),    
                    "date": df.loc[i, 'WEEK'],
                    "variable": df.loc[i, 'variable'],
                    "value": df.loc[i, 'value'], 
                    #"name": df.loc[i, 'segment'] + ': ' + df.loc[i, 'SCORECARD'], 
                    "name": str(df.loc[i, 'BUSINESS_TYPE']) + '_' + str(df.loc[i, 'CLIENT_TYPE']) + ': ' + str(df.loc[i, 'SCORECARD']), 
                    "business_type": str(df.loc[i, 'BUSINESS_TYPE']).lower(),
                    "client_type": str(df.loc[i, 'CLIENT_TYPE']).lower()
                })

            # Save json
            with open(temp_files + '/scorecard_gini/' + cur_file, 'w') as outfile:
                json.dump(json_data, outfile)

        else:
            json_data = []

    return render(request, 'gini.html', {'gini_json': json.dumps(json_data)})


@login_required
# calculate gini locally
def scorecard_monitoring2(request):

    get_keys = list(request.GET.keys())

    save_event(request)

    if 'business_type' in get_keys and 'client_type' in get_keys and 'scorecard' in get_keys: 
        
        cur_scorecard = request.GET['scorecard']
        cur_business_type = request.GET['business_type']
        cur_client_type = request.GET['client_type'].rsplit('_', 1)[1]

        # print(request.POST['client_type'], cur_client_type)
    else:

        cur_scorecard_obj = Scorecard.objects.filter(name='MainScore_201709_Core_New_EquifaxHit', business_type='Core')[0] # MainScore_201709_Core_New_EquifaxHit dn_pd_eq_1217
        cur_scorecard = str(cur_scorecard_obj.name)
        cur_business_type = str(cur_scorecard_obj.business_type)
        cur_client_type = str(cur_scorecard_obj.client_type)

        # print(type(cur_client_type))
        # print('Новый', cur_client_type, 'Новый' == cur_client_type) #, len('Новый'), len(cur_client_type))

    now = datetime.now()

    table_name = '[TestForRisk].[dbo].[shorokh_scorecard_variables_%s]'
    table_date = str(now.year) + '_' + '0' * (2 - len(str(now.month))) + str(now.month) + '_01'

    # init logging
    logger = logging.getLogger("log")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=temp_files + '/scorecard_monitoring/log/' + datetime.strftime(now, '%Y_%m_%d') + ".log")  # create the logging file handler
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add handler to logger object
    # logger.info()
    # logger.removeHandler(fh)

    cur_log = datetime.strftime(now, '%H_%M_%S')

    logger.info('current filters: [' + cur_business_type + '] [' + cur_client_type + '] [' + cur_scorecard + ']')
    print('------------ start -------------', cur_business_type, cur_client_type, cur_scorecard)

    # cur_week_start = now - timedelta(days=(now.weekday() + 1))
    # cur_week_end = cur_week_start + timedelta(days=6)

    prev_week_end = now - timedelta(days=(now.weekday() + 1))
    # print(prev_week_end)

    ################################################################
    #                          INIT DIRS                           #
    ################################################################

    sm_base = "/scorecard_monitoring/"

    ################################################################
    #                       SCORECARDS LIST                        #
    ################################################################
    
    min_appl_count = 100
    cur_file = 'scorecards_' + datetime.strftime(prev_week_end, "%Y_%m_%d") + '.csv'

    if cur_file in os.listdir(temp_files + sm_base + 'scorecards_list/'):
        scorecards = pd.read_csv(temp_files + sm_base + 'scorecards_list/' + cur_file, encoding='cp1251', sep='\t')
        logger.info(cur_log + ' scorecards list loaded from ' + cur_file)
        print(cur_log + ' scorecards list loaded from ' + cur_file)
   
    else:

        # take 4 weeks ago
        weeks_to_past = 4  # weeks count for option selection
        dt = datetime.now()
        cur_week_start = dt - timedelta(days=(dt.weekday() + 7))
        cur_week_end = cur_week_start + timedelta(days=6)
        weeks_ago = []
        for i in range(weeks_to_past):
            week_start = cur_week_start - timedelta(days=7)
            week_end = week_start + timedelta(days=6)
            cur_week_start = week_start
            weeks_ago.append("'"+ datetime.strftime(week_start, "%Y-%m-%d") + ' --- ' + datetime.strftime(week_end, "%Y-%m-%d") + "'")
        
        req = sr.scorecards_list_per_weeks
        req = req.replace("WEEKS1", ','.join(weeks_ago)).replace("COUNT1", str(min_appl_count))

        res = execute_db(c.db_p, req, pr=True)
        logger.info(cur_log + '\n' + req + '\n')
        scorecards = pd.DataFrame(res[1:], columns=res[0])

        # scorecards_db = Scorecard.objects.filter(working=True)
        if not scorecards.empty:
            scorecards.to_csv(temp_files + sm_base + 'scorecards_list/' + cur_file, encoding='cp1251', sep='\t', index=False)
    
    
        ################################################################
        #                      UPDATE SCORECARDS                       #
        ################################################################

        for i in range(len(scorecards)):
            check_scorecard = Scorecard.objects.filter(name=scorecards.loc[i, 'SCORECARD'], business_type=scorecards.loc[i, 'BUSINESS_TYPE'], 
               client_type=scorecards.loc[i, 'CLIENT_TYPE'])
            if check_scorecard:
                pass
            else:
                new_scorecard = Scorecard(name=scorecards.loc[i, 'SCORECARD'], business_type=scorecards.loc[i, 'BUSINESS_TYPE'], 
                client_type=scorecards.loc[i, 'CLIENT_TYPE'])
                new_scorecard.working = True
                new_scorecard.save()

            # set-off scorecards
            for scorecard_db in Scorecard.objects.all().order_by('business_type', 'client_type'):
                if scorecards[(scorecards['SCORECARD'] == str(scorecard_db.name)) & (scorecards['CLIENT_TYPE'] == str(scorecard_db.client_type)) 
                & (scorecards['BUSINESS_TYPE'] == str(scorecard_db.business_type))].empty:
                    scorecard_db.working = False
                    scorecard_db.save()

    # print(scorecards)

    # scorecards by python
    cur_file = 'scorecards_by_python_' + datetime.strftime(prev_week_end, "%Y_%m_%d") + '.pkl'

    if cur_file in os.listdir(temp_files + sm_base + 'scorecards_list/'):
        
        with open(temp_files + sm_base + 'scorecards_list/' + cur_file, 'rb') as file:
            scorecards_by_python = pickle.load(file)   
        logger.info(cur_log + ' scorecards by python list loaded from ' + cur_file)
        print(cur_log + ' scorecards by python list loaded from ' + cur_file)

    else:
        req = sr.scorecards_by_python
        res = execute_db(c.db_p, req, pr=True)[1:]
        logger.info(cur_log + '\n' + req + '\n')
        if res:
            scorecards_by_python = [el[0] for el in res]
        else:
            scorecards_by_python = []

        if scorecards_by_python:
            with open(temp_files + sm_base + 'scorecards_list/' + cur_file, 'wb') as file:
                pickle.dump(scorecards_by_python, file)

    # print(scorecards_by_python)

    ################################################################
    #                        REQUESTS DICT                         #
    ################################################################

    cur_file = 'requests.pkl'
    if cur_file in os.listdir(temp_files + sm_base + '/'):
        with open(temp_files + sm_base + '/' + cur_file, 'rb') as file:
            requests_out = pickle.load(file)
    else:
        requests_out = {} # dict for all requests

    ################################################################
    #                           GINI                               #
    ################################################################

    min_appl_count = 100
    months_ago_gini = 4

    month1 = now.month - months_ago_gini
    year1 = now.year
    if month1 <= 0:
        months_last_year = months_ago_gini - now.month
        month1 = 12 - months_last_year
        year1 -= 1  # previous year

    date_for_indicator = str(year1) + '-' +  '0'*(2 - len(str(month1))) + str(month1)

    month2 = now.month - months_ago_gini - 1
    year2 = now.year
    if month2 <= 0:
        months_last_year = months_ago_gini + 1 - now.month
        month2 = 12 - months_last_year
        year2 -= 1  # previous year

    date_prev = str(year2) + '-' +  '0'*(2 - len(str(month2))) + str(month2)
    
    print('date_for_indicator', date_for_indicator)
    print('date_prev', date_prev)
    
    cur_file = 'gini_basis_' + date_for_indicator + '.csv'
    if cur_file not in os.listdir(temp_files + sm_base + '/scorecards_gini/'):
        
        # calculate all ginis

        last_file = max([el for el in os.listdir(temp_files + sm_base + '/scorecards_gini/') if el != 'log'])
        print(last_file)
        p = re.compile(r'\d_(.*)\.csv')
        print(cur_file, p.findall(cur_file))
        # last_date = p.findall(cur_file)[-1]
        last_date = cur_file.split('_')[-1].split('.')[0]
        print(last_date)

        req = sr.scorecards_list_per_month
        req = req.replace("MONTH1", date_for_indicator).replace("COUNT1", str(min_appl_count))
        data = execute_db(c.db_p, req, pr=True)

        logger.info(cur_log + '\n' + req + '\n')
        
        df = pd.DataFrame(data[1:], columns=data[0])
        df['gini'] = -1
        df['ks'] = -1

        print(df)

        if not df.empty:

            df.insert(0, 'MONTH', date_for_indicator)
            df.insert(3, 'BUREAU', np.nan)
            df.insert(4, 'SEGMENT', np.nan)

            #df = df[df['SCORECARD'].isin(['dn_pd_eq_1217', 'cr_no_upsale_0318'])]
            # df.index = range(len(df))
            # print(df)

            for i in range(len(df)):
                CUR_SCORECARD = df.loc[i, 'SCORECARD']
                CUR_BUSINESS_TYPE = df.loc[i, 'BUSINESS_TYPE']
                CUR_CLIENT_TYPE = df.loc[i, 'CLIENT_TYPE']
                bureau = ''

                # Init business type
                if CUR_BUSINESS_TYPE == 'Core':
                    CUR_BUSINESS_TYPE1 = "Core', 'from_SPR7_to_SPR4', 'from_SPR14_to_SPR4"
                else:
                    CUR_BUSINESS_TYPE1 = CUR_BUSINESS_TYPE

                # Init bureau
                cond1 = "1=1"
                if 'equifaxhit' in CUR_SCORECARD.lower() or 'eq' in CUR_SCORECARD.lower():
                    cond1 = "Bureau = 'Equifax hit'"
                    bureau = 'EquifaxHit'

                if 'nobureauhit' in CUR_SCORECARD.lower():
                    bureau = 'NoBureauHit'

                if 'nbch' in CUR_SCORECARD.lower():
                    bureau = 'NbchHit'

                df.loc[i, 'BUREAU'] = bureau

                # Init score field
                if CUR_SCORECARD in scorecards_by_python:
                    cur_score = 'MainPROBABballs as MainSCOREballs'  # MainSCOREballs, MainPROBABballs,
                else:
                    cur_score = 'MainSCOREballs' 

                req = sr.data_for_gini_badrate
                req = req.replace("MONTH1", date_for_indicator).replace("SCORECARD1", CUR_SCORECARD).replace("BUSINESS_TYPE1", CUR_BUSINESS_TYPE1).replace("CLIENT_TYPE1", CUR_CLIENT_TYPE).replace("COND1", cond1).replace("SCORE1", cur_score)
               
                label = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + 'gini'
                if label not in requests_out.keys():
                    requests_out[label] = req

                res = execute_db(c.db_p, req, pr=True)

                logger.info(cur_log + '\n' + req + '\n')

                temp = pd.DataFrame(res[1:], columns=res[0])
                temp['MainSCOREballs'] = temp['MainSCOREballs'].astype(float)
                print(temp.head())

                try:
                    gini = 100 * abs(2 * roc_auc_score(temp[temp['BadRate'].notnull()]['BadRate'], temp[temp['BadRate'].notnull()]['MainSCOREballs']) - 1)
                    ks = 100 * stats.ks_2samp(temp[temp['BadRate'] == 0]['MainSCOREballs'], temp[temp['BadRate'] == 1]['MainSCOREballs']).statistic   
                    df.loc[i, 'gini'] = gini
                    df.loc[i, 'ks'] = ks     
                except:
                    pass

                ################################################################
                #           Scorecard variables effectiveness part             #
                ################################################################

                try:
                    
                    report_date = str(year1) + '-' + '0' * (2 - len(str(month1))) + str(month1)

                    share_date_month = month1 + 4   
                    share_date_year = year1
                    if share_date_month > 12:
                        share_date_month = share_date_month % 12
                        share_date_year += 1  # previous year
                    share_date = str(share_date_year) + '_' + '0' * (2 - len(str(share_date_month))) + str(share_date_month) + '_01'

                    prev_share_date_month = month1 + 3   
                    prev_share_date_year = year1
                    if prev_share_date_month > 12:
                        prev_share_date_month = prev_share_date_month % 12
                        prev_share_date_year += 1  # previous year
                    prev_share_date = str(prev_share_date_year) + '_' + '0' * (2 - len(str(prev_share_date_month))) + str(prev_share_date_month) + '_01'

                    print('report_date', report_date)
                    print('share_date', share_date)
                    print('prev_share_date', prev_share_date)


                    print()

                    # Create new table

                    req = '''
                    IF OBJECT_ID('TABLE1', 'U') IS NULL begin

                    CREATE table TABLE1 (
                        MONTH NVARCHAR(255),
                        BUSINESS_TYPE NVARCHAR(255),
                        CLIENT_TYPE NVARCHAR(255),
                        BUREAU NVARCHAR(255),
                        SEGMENT NVARCHAR(255),
                        SCORECARD NVARCHAR(255),
                        Variable NVARCHAR(255),  
                        Point FLOAT,
                        AllAppl FLOAT,
                        FinAppl FLOAT,
                        AllShare FLOAT,
                        FinShare FLOAT,
                        FinRate FLOAT,
                        Indicator FLOAT,
                        IndicatorType NVARCHAR(255),
                    )

                    insert
                    into TABLE1
                    select * from TABLE1
                    
                    end

                    '''.replace('TABLE1', table_name) % (share_date, share_date, share_date, prev_share_date)

                    print('Create new table')
                    print(req)

                    conn = pymssql.connect(c.db_d)
                    cur = conn.cursor()
                    cur.execute(req)
                    conn.commit()
                    conn.close()

                    # Get new report data

                    table = '[DSS].[spr].[SPR_ALL_DATA]'

                    scorecard = CUR_SCORECARD
                    client_type = CUR_CLIENT_TYPE
                    business_type = CUR_BUSINESS_TYPE
                    
                    date1 = report_date + '-01'
                    
                    date2_month = month1 + 1
                    date2_year = year1
                    if date2_month > 12:
                        date2_month = date2_month % 12
                        date2_year += 1

                    date2 = str(date2_year) + '-' + '0' * (2 - len(str(date2_month))) + str(date2_month) + '-01'
                    
                    cond2 = '1=1'

                    segment = '%s_%s' % (business_type, 'New' if 'Новый' in client_type else 'Repeat')
                    if bureau:
                        segment += '_%s' % bureau

                    print(date1, date2)

                    req_ve = sr.scorecard_variables_effectiveness.replace("TABLE1", table).replace("SCORECARD1", scorecard).replace("CLIENT_TYPE1", client_type).replace("BUSINESS_TYPE1", CUR_BUSINESS_TYPE1)\
                                 .replace("DATE1", date1).replace("DATE2", date2).replace("COND2", cond2)

                    print('Get new report data')
                    print(req_ve)

                    conn = pymssql.connect(c.db_p)
                    df3 = pd.read_sql(req_ve, conn)
                    conn.close()
                    df3.rename(columns={'CHARACT': 'Variable'}, inplace=True)

                    # df3['POINT'] = df3['POINT'].astype(float)

                    # check 0 group
                    for var in list(df3['Variable'].unique()):
                        if 0.0 not in list(df3[df3['Variable'] == var]['Point']):
                            print('no null here')
                            df3 = df3.append(pd.DataFrame([[var, 0, 0, 0, 0, 0, 0, 0]], columns=list(df3.columns)))

                    df3 = df3.sort_values(by=['Variable', 'Point'])
                    df3.index = range(len(df3))

                    df3.insert(0, 'MONTH', report_date)
                    df3.insert(1, 'BUSINESS_TYPE', business_type)
                    df3.insert(2, 'CLIENT_TYPE', 'New' if 'Новый' in client_type else 'Repeat')
                    df3.insert(3, 'BUREAU', bureau)
                    df3.insert(4, 'SEGMENT', segment)
                    df3.insert(5, 'SCORECARD', scorecard)

                    df3.fillna(0, inplace=True)

                    indicator_type = list(df3.columns)[-1]
                    df3.rename(columns={indicator_type: 'Indicator'}, inplace=True)
                    df3['IndicatorType'] = indicator_type

                    df3 = df3.sort_values(by=['Variable', 'Point'], ascending=[True, True])

                    df3.to_excel(temp_files + sm_base + '/scorecards_gini/eff/%s_%s_%s_%s.xlsx' % (report_date, business_type, client_type, scorecard), index=False)

                    print(df3.head(10))

                    # Delete old report data

                    conn = pymssql.connect(c.db_d)
                    cur = conn.cursor()

                    req = '''
                    delete from TABLE1 where month = '%s' 
                    and scorecard = '%s' 
                    and business_type = '%s'
                    and client_type = '%s'
                    '''.replace('TABLE1', table_name) % (table_date, report_date, scorecard, business_type,  'New' if 'Новый' in client_type else 'Repeat')

                    print('Delete old report data')
                    print(req)

                    cur.execute(req)
                    conn.commit()
                    conn.close()

                    # Insert new report data

                    conn = pymssql.connect(c.db_d)
                    cur = conn.cursor()
                    query = "INSERT INTO TABLE1 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, " \
                            "%s, %s, %s, %s, %s, %s, %s)"
                    query = query.replace('TABLE1', table_name % table_date)

                    print('Insert new report data')
                    print(query)
                    
                    sql_data = tuple(map(tuple, df3.values))
                    print(sql_data)
                    cur.executemany(query, sql_data)
                    conn.commit()
                    cur.close()
                    conn.close()


                    # Get new share data
                    date1 = prev_share_date.replace('_', '-')

                    date2_month = month1 + 4
                    date2_year = year1
                    if date2_month > 12:
                        date2_month = date2_month % 12
                        date2_year += 1  # previous year
                    date2 = str(date2_year) + '-' + '0' * (2 - len(str(date2_month))) + str(date2_month) + '-01'
                    month_new = str(prev_share_date_year) + '-' + '0' * (2 - len(str(prev_share_date_month))) + str(prev_share_date_month)

                    print(date1, date2)

                    req_ve = sr.scorecard_variables_effectiveness.replace("TABLE1", table).replace("SCORECARD1", scorecard).replace("CLIENT_TYPE1", client_type).replace("BUSINESS_TYPE1", CUR_BUSINESS_TYPE1)\
                                                 .replace("DATE1", date1).replace("DATE2", date2).replace("COND2", cond2)

                    print('Get new share data')
                    print(req_ve)

                    conn = pymssql.connect('mck-p-dwh')
                    df_new = pd.read_sql(req_ve, conn)
                    conn.close()
                    df_new.rename(columns={'CHARACT': 'Variable'}, inplace=True)


                    # check 0 group
                    for var in list(df_new['Variable'].unique()):
                        if 0.0 not in list(df3[df3['Variable'] == var]['Point']):
                            print('no null here')
                            df_new = df_new.append(pd.DataFrame([[var, 0, 0, 0, 0, 0, 0, 0]], columns=list(df3.columns)))

                    df_new = df_new.sort_values(by=['Variable', 'Point'])
                    df_new.index = range(len(df_new))

                    df_new.insert(0, 'MONTH', month_new)
                    df_new.insert(1, 'BUSINESS_TYPE', business_type)
                    df_new.insert(2, 'CLIENT_TYPE', 'New' if 'Новый' in client_type else 'Repeat')
                    df_new.insert(3, 'BUREAU', bureau)
                    df_new.insert(4, 'SEGMENT', segment)
                    df_new.insert(5, 'SCORECARD', scorecard)

                    df_new.fillna(0, inplace=True)

                    df_new.rename(columns={indicator_type: 'Indicator'}, inplace=True)
                    df_new['IndicatorType'] = indicator_type

                    df_new = df_new.sort_values(by=['Variable', 'Point'], ascending=[True, True])

                    print(df_new.head(10))

                    df_new.to_excel(temp_files + sm_base + '/scorecards_gini/eff/%s_%s_%s_%s.xlsx' % (month_new, business_type, client_type, scorecard), index=False)

                    # Insert new share data

                    conn = pymssql.connect(c.db_d)
                    cur = conn.cursor()
                    query = "INSERT INTO TABLE1 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, " \
                            "%s, %s, %s, %s, %s, %s, %s)"
                    query = query.replace('TABLE1', table_name % table_date)
                    print(query)
                    sql_data = tuple(map(tuple, df_new.values))
                    print(sql_data)
                    cur.executemany(query, sql_data)
                    conn.commit()
                    cur.close()
                    conn.close()

                    # End
                except Exception as e:
                    print(e)

                ################################################################
                #         Scorecard variables effectiveness part end           #
                ################################################################


            df = df.replace({'CLIENT_TYPE': {'Новый': 'New', 'Повторный': 'Repeat'}})

            print(df)
            logger.info(cur_log + '\n' + df.to_string() + '\n')

            df = df[df['gini'] != -1]  # filter by not null values
            # save all data in excel
            # data = pd.read_excel(temp_files + '/scorecard_gini/gini_basis.xlsx', encoding='cp1251')  # checkpoint from 2018-03
            gini_df = pd.read_csv(temp_files + sm_base + '/scorecards_gini/gini_basis_' + date_prev + '.csv', encoding='cp1251', sep='\t')
            gini_df = pd.concat([gini_df, df], axis=0)
            gini_df.index = range(len(gini_df))
            print(gini_df)
            gini_df.to_csv(temp_files + sm_base + '/scorecards_gini/gini_basis_' + date_for_indicator + '.csv', encoding='cp1251', sep='\t', index=False)
     
    
    # If file exists
    else:
        
        gini_df = pd.read_csv(temp_files + sm_base + '/scorecards_gini/gini_basis_' + date_for_indicator + '.csv', encoding='cp1251', sep='\t')
        print(cur_log + ' gini loaded from ' + 'gini_basis_' + date_for_indicator + '.csv')

    d = {'Новый': 'New', 'Повторный': 'Repeat'}
    gini_df = gini_df[(gini_df['BUSINESS_TYPE'] == cur_business_type) & (gini_df['SCORECARD'] == cur_scorecard) & (gini_df['CLIENT_TYPE'] == d[cur_client_type])]

    gini_df.index = range(len(gini_df))

    # print(gini)
    json_gini = []

    for i in range(len(gini_df)):
        json_gini.append({ 'date': gini_df.loc[i, 'MONTH'], 'value': round(gini_df.loc[i, 'gini'], 3)})
    
    # print(json_gini)

    ################################################################
    #                        RISK GROUPS                           #
    ################################################################

    cur_file = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + datetime.strftime(prev_week_end, "%Y_%m_%d") + '.csv'

    if cur_file in os.listdir(temp_files + sm_base + 'scorecards_data/'):

        file_size = os.stat(temp_files + sm_base + 'scorecards_data/' + cur_file).st_size / (1024 ** 2)  # Get file size
        if file_size > 50:  # If file size more than threshold
            data_main = pd.DataFrame()  # Create result data frame
            chunks = pd.read_table(temp_files + sm_base + 'scorecards_data/' + cur_file, chunksize=1000, iterator=True, encoding='cp1251', sep='\t')
            for chunk in chunks:  # For each part in parts
                data_main = pd.concat([data_main, chunk], axis=0)  # Join file parts

        else:
            data_main = pd.read_csv(temp_files + sm_base + 'scorecards_data/' + cur_file, encoding='cp1251', sep='\t')
        logger.info(cur_log + ' data for ' + cur_scorecard +' loaded from ' + cur_file)
        print(cur_log + ' data for ' + cur_scorecard +' loaded from ' + cur_file)
   
    else:

        # cur_business_type = scorecards[scorecards['SCORECARD'] == cur_scorecard]['BUSINESS_TYPE'].values[0]
        # cur_client_type = scorecards[scorecards['SCORECARD'] == cur_scorecard]['CLIENT_TYPE'].values[0]

        # Init business type
        if str(cur_business_type) == 'Core':
            cur_business_type_f = "Core', 'from_SPR7_to_SPR4', 'from_SPR15_to_SPR4"
        else:
            cur_business_type_f = str(cur_business_type)

        # Init score field
        if cur_scorecard in scorecards_by_python:
            cur_score = 'MainPROBABballs as MainSCOREballs'  # MainSCOREballs, MainPROBABballs,
        else:
            cur_score = 'MainSCOREballs' 

        cond1 = '1=1'  # special condition

        req = sr.scorecards_rg_indicators
        req = req.replace("SCORECARD1", str(cur_scorecard)).replace("BUSINESS_TYPE1", cur_business_type_f).replace("CLIENT_TYPE1", str(cur_client_type)).replace("COND1", cond1).replace("SCORE1", cur_score)
        res = execute_db(c.db_p, req, pr=True)
        
        label = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + 'financed'
        if label not in requests_out.keys():
            requests_out[label] = req.replace('1=1', 'Financed = 1')

        logger.info(cur_log + '\n' + req + '\n')
        data_main = pd.DataFrame(res[1:], columns=res[0])
        data_main.replace({None: np.nan}, inplace=True)
        if not data_main.empty:
            data_main.to_csv(temp_files + sm_base + 'scorecards_data/' + cur_file, encoding='cp1251', sep='\t', index=False)

    
    # print(data.head())
    # try:
    #     data_main['MainSCOREballs'] = data_main['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
    #     data_main = data_main.replace({'MainSCOREballs': {'NULL': -1}})
    #     data_main['MainSCOREballs'] = data_main['MainSCOREballs'].astype(float)
    # except Exception as e:
    #     print(e)

    # data_main['MainSCOREballs'].fillna(-1, inplace=True)
    # data_main = data_main.replace({'MainSCOREballs': {'NULL': -1}})

    # data_main['MainSCOREballs'] = data_main['MainSCOREballs'].astype(float)

    ################################################################
    #                       FINANCED PART                          #
    ################################################################

    data_f = data_main[data_main['Financed'] == 1].copy()
    data_f_g = data_f.groupby(['CRM_CREATION_Month', 'RISK_GROUP']).agg({'UCDB_ID': ['count']})
    data_f = data_f_g.transpose().reset_index(level=0, drop=True).transpose().reset_index()
    data_f.rename(columns={'CRM_CREATION_Month': 'date'}, inplace=True)

    # print(data_f)
    
    # print(data_f)

    # json_data = [
    #       { 'date': "2018-01", 'A': 60, 'B': 15, 'C': 9, 'D': 6},
    # ]

    # json_gini = [
    #       { 'date': "2018-01", 'value': 0.65 },
    # ]

    json_data_f = []

    for date in sorted(data_f['date'].unique()):
        temp = data_f[data_f['date'] == date]
        lst = {'date': date}
        for el in zip(temp['RISK_GROUP'], temp['count']):
            lst[el[0]] = el[1]
            for rg in list('ABCD'):
                if rg not in lst.keys():
                    lst[rg] = 0
        json_data_f.append(lst)

    # print(json_data_f)

    ################################################################
    #                      ALL APPLICATIONS                        #
    ################################################################

    data_all = data_main.copy()
    data_all_g = data_all.groupby(['CRM_CREATION_Month', 'RISK_GROUP']).agg({'UCDB_ID': ['count']})
    data_all = data_all_g.transpose().reset_index(level=0, drop=True).transpose().reset_index()
    data_all.rename(columns={'CRM_CREATION_Month': 'date'}, inplace=True)
    
    # print(data_all.head())
    json_data_all = []

    for date in sorted(data_all['date'].unique()):
        temp = data_all[data_all['date'] == date]
        lst = {'date': date}
        for el in zip(temp['RISK_GROUP'], temp['count']):
            lst[el[0]] = el[1]
            for rg in list('ABCD'):
                if rg not in lst.keys():
                    lst[rg] = 0
        json_data_all.append(lst)

    # print(json_data_f)

    ################################################################
    #                     SCORE STABILITY GROUP                    #
    ################################################################

    #  -------------------- v.1 (training qcut) --------------------

    # cur_file = cur_scorecard + '_bins.pkl'

    # # load bins
    # if cur_file in os.listdir(temp_files + sm_base + 'scorecards_dev_bins/'):
    #     with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'rb') as file:
    #         bins = pickle.load(file)
    #     logger.info(cur_log + ' bins for ' + cur_scorecard +' loaded from ' + cur_file)

    # else:
    #     scores_file = cur_scorecard + '_scores.xlsx'
    #     if scores_file in os.listdir(temp_files + sm_base + 'scorecards_dev_scores/'):
            
    #         df = pd.read_excel(temp_files + sm_base + 'scorecards_dev_scores/' + scores_file)
    #         print(df.head())
    #         _, bins = pd.qcut(df['Prop'], 10, retbins=True)
    #         bins = list(bins)
    #         bins[0] = -np.inf
    #         bins[-1] = np.inf
    #         df['dev'] = pd.cut(df['Prop'], bins=bins)
    #         # get dev scores
    #         data = df['dev'].value_counts(normalize=True).sort_index().to_frame()  
    #         data.index.rename('group', inplace=True)
    #         data.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_dev_scores.csv', sep='\t', encoding='cp1251')

    #         with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'wb') as file:
    #             pickle.dump(bins, file) 
    #     else:
    #         bins = []

    # if bins:
    #     bins[0] = -np.inf
    #     bins[-1] = np.inf

    # data_bins = pd.read_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_dev_scores.csv', sep='\t', encoding='cp1251', index_col='group')
    
    # # print(data_bins)

    # dates_ss = []

    # # data_f = data_main[data_main['Financed'] == 1].copy()
    # data_f = data_main.copy()

    # for date in sorted(list(data_f['CRM_CREATION_Month'].unique())):

    #     temp = data_f[data_f['CRM_CREATION_Month'] == date]
    #     temp['N'] = pd.cut(temp['MainSCOREballs'], bins=bins)
    #     # get dev scores
    #     data_bins[date] = temp['N'].value_counts(normalize=True).sort_index().values
    #     dates_ss.append(date)

    # print(data_bins)
    # if not data_bins.empty:
    #     data_bins.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + max(dates_ss) + '_scores.csv', sep='\t', encoding='cp1251')

    # json_scores = []
    # for date in dates_ss:
    #     data_bins[date+'_d'] = data_bins[date]-data_bins['dev']
    #     data_bins[date+'_w'] = np.log(data_bins[date]/data_bins['dev'])
    #     data_bins = data_bins.replace({date+'_w': {np.inf: 0, -np.inf: 0}})
    #     data_bins[date+'_s'] = data_bins[date+'_d']*data_bins[date+'_w'] 
    #     ssi = data_bins[date+'_s'].sum()
    #     json_scores.append({'date': date, 'value': round(ssi, 5)})

    # print(json_scores)
    # print(data_bins)
    # data_bins.to_excel(temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + max(dates_ss) + '_scores_calc.xlsx')


    #  ----------------------- v.2 (nth week qcut)  ---------------

    from_month_number = 2

    # data_f = data_main[data_main['Financed'] == 1].copy()
    data_ss = data_main.copy()
    # data_ss = data_ss[(~data_ss['MainSCOREballs'].isin(['NULL']))]
    data_ss = data_ss[(data_ss['MainSCOREballs'].notnull()) & (~data_ss['MainSCOREballs'].isin(['NULL']))]

    try:
        data_ss['MainSCOREballs'] = data_ss['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
        data_ss['MainSCOREballs'] = data_ss['MainSCOREballs'].astype(float)
    except Exception as e:
        print(e)

    all_dates = sorted(list(data_ss['CRM_CREATION_Month'].unique()))

    # check scorecard's work 
    if len(all_dates) > 0:

        try:
            date_base = all_dates[from_month_number - 1]
        except:
            date_base = all_dates[0]

        
        # cur_file = cur_scorecard + '_from_'+ str(date_base) +'_month_bins.pkl'
        cur_file = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_from_'+ str(date_base) +'_month_bins.pkl'

        # load bins
        if cur_file in os.listdir(temp_files + sm_base + 'scorecards_dev_bins/'):
            with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'rb') as file:
                bins = pickle.load(file)
            logger.info(cur_log + ' bins for ' + cur_scorecard +' loaded from ' + cur_file)

        else:

            df = data_ss[(data_ss['CRM_CREATION_Month'] == date_base)]
        
            _, bins = pd.qcut(df['MainSCOREballs'], 10, retbins=True, duplicates='drop')
            bins = list(bins)
            bins[0] = -np.inf
            bins[-1] = np.inf
            df[date_base + '_dev'] = pd.cut(df['MainSCOREballs'], bins=bins)
            # get dev scores
            data = df[date_base + '_dev'].value_counts(normalize=True).sort_index().to_frame()  
            data.index.rename('group', inplace=True)
            data.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_from_'+ str(date_base) +'_month_dev_scores.csv', sep='\t', encoding='cp1251')

            with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'wb') as file:
                pickle.dump(bins, file) 

        if bins:
            bins[0] = -np.inf
            bins[-1] = np.inf

        data_bins = pd.read_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_from_'+ str(date_base) +'_month_dev_scores.csv', sep='\t', encoding='cp1251', index_col='group')
        
        # print(data_bins)

        dates_ss = all_dates[(all_dates.index(date_base) + 1):]
        # print('dates_ss', dates_ss)

        if dates_ss:

            for date in dates_ss:

                temp = data_ss[data_ss['CRM_CREATION_Month'] == date]
                temp['N'] = pd.cut(temp['MainSCOREballs'], bins=bins)
                # get dev scores
                data_bins[date] = temp['N'].value_counts(normalize=True).sort_index().values

            # print(data_bins)
            if not data_bins.empty:
                data_bins.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + max(dates_ss) + '_scores.csv', sep='\t', encoding='cp1251')

            json_scores = []
            for date in dates_ss:
                data_bins[date+'_d'] = data_bins[date]-data_bins[date_base + '_dev']
                data_bins[date+'_w'] = np.log(data_bins[date]/data_bins[date_base + '_dev'])
                data_bins = data_bins.replace({date+'_w': {np.inf: 0, -np.inf: 0}})
                data_bins[date+'_s'] = data_bins[date+'_d']*data_bins[date+'_w'] 
                ssi = data_bins[date+'_s'].sum()
                json_scores.append({'date': date, 'value': round(ssi, 5)})

            # print(json_scores)
            # print(data_bins)
            data_bins.to_excel(temp_files + sm_base + 'scorecards_dev_bins/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + max(dates_ss) + '_scores_calc.xlsx')
        else:
            json_scores = []
    # scorecard too new
    else:
        json_scores = []
        date_base = ''


    ################################################################
    #                          MEAN VALUES                         #
    ################################################################

    json_mean_scores = []
    json_mean_indicators = []

    indicators = ['3+ 4 *2WoB', '30+ 3MoB']

    min_date = data_main['CRM_CREATION_Month'].min()  # for filtering in the end

    for date in sorted(list(data_main['CRM_CREATION_Month'].unique())):

        temp = data_main[data_main['CRM_CREATION_Month'] == date].copy()
        temp = temp[(temp['MainSCOREballs'].notnull()) & (~temp['MainSCOREballs'].isin(['NULL']))]

        try:
            temp['MainSCOREballs'] = temp['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
            temp['MainSCOREballs'] = temp['MainSCOREballs'].astype(float)
        except Exception as e:
            print(e)


        score_mean = temp['MainSCOREballs'].mean() if len(temp) > 0 else 0.0
        json_mean_scores.append({"date": date, "score_mean": score_mean})
        
        # FPD
        fpd_mean = temp[temp['FPD'].notnull()]['FPD'].mean() if len(temp[temp['FPD'].notnull()]) > 0 else 0
        json_mean_indicators.append({"date": date, "label": 'fpd', "value": round(fpd_mean, 5)})


        for indicator in indicators:
            temp = temp[temp[indicator].notnull()]
            temp['ind_' + indicator] = temp[indicator] / (temp['LOAN_AMOUNT'] + temp['LOAN_COMISSION'])
            indicator_value = temp[temp["ind_" + indicator].notnull()]["ind_" + indicator].mean() if len(temp[temp["ind_" + indicator].notnull()]) > 0 else 0
            json_mean_indicators.append({"date": date, "label": indicator, "value": round(indicator_value, 5)})

        del temp

    ################################################################
    #                        SHARE OF SCORES                       #
    ################################################################

    temp_ind = data_main[(data_main['Financed'] == 1) & (data_main['CRM_CREATION_Month'] == date_for_indicator)].copy()
    temp_ind = temp_ind[(temp_ind['MainSCOREballs'].notnull()) & (~temp_ind['MainSCOREballs'].isin(['NULL']))]
    
    temp = data_main[data_main['CRM_CREATION_Month'] == date_for_indicator].copy()
    temp = temp[(temp['MainSCOREballs'].notnull()) & (~temp['MainSCOREballs'].isin(['NULL']))]

    try:
        temp_ind['MainSCOREballs'] = temp_ind['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
        temp_ind['MainSCOREballs'] = temp_ind['MainSCOREballs'].astype(float)
    except Exception as e:
        print(e)

    try:
        temp['MainSCOREballs'] = temp['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
        temp['MainSCOREballs'] = temp['MainSCOREballs'].astype(float)
    except Exception as e:
        print(e)

    if cur_scorecard in scorecards_by_python:
        temp_ind['MainSCOREballs'] = temp_ind['MainSCOREballs'].map(lambda x: round(x, 2))
    else:
        temp_ind['MainSCOREballs'] = temp_ind['MainSCOREballs'].map(lambda x: round(x, -1))

    if cur_scorecard in scorecards_by_python:
        temp['MainSCOREballs'] = temp['MainSCOREballs'].map(lambda x: round(x, 2))
    else:
        temp['MainSCOREballs'] = temp['MainSCOREballs'].map(lambda x: round(x, -1))
    
    indicator = '30+ 3MoB'
    temp_ind['indicator'] = temp_ind[indicator].map(lambda x: 1 if x > 0 else 0)

    temp_1 = temp_ind[temp_ind['indicator'] == 1]
    temp_0 = temp_ind[temp_ind['indicator'] == 0]

    json_scores_share = []

    for dfi in zip([temp, temp_1, temp_0], ['score_all_applications', 'score_indicator_1', 'score_indicator_0']):
    # for dfi in zip([temp_1, temp_0], ['score_indicator_1', 'score_indicator_0']):

        label = dfi[1]
        temp = dfi[0].groupby(['MainSCOREballs']).agg({'MainSCOREballs': 'count'})
        temp['MainSCOREballs'] = temp['MainSCOREballs'] / temp['MainSCOREballs'].sum()

        # print(temp)

        for index in temp.index:
            json_scores_share.append({'score': index, 'key': label, 'share': temp.loc[index, 'MainSCOREballs']})

        del temp
    
    # print(json_scores_share) 
    # print(temp.head())
    # print(temp_1.head())
    # print(temp_0.head())
    # print(temp['indicator'].value_counts())


    ################################################################
    #                    VARIABLES  EFFECTIVENESS                  #
    ################################################################

    n_months = 100
    cur_file = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_variables_effectiveness_' + date_for_indicator + '.pkl'
    
    if cur_file in os.listdir(temp_files + sm_base + '/scorecard_variables/'):

        with open(temp_files + sm_base + '/scorecard_variables/' + cur_file, 'rb') as file:
            temp_dict = pickle.load(file)
        print(cur_file, 'was loaded')

        scorecard_variables = OrderedDict(temp_dict)

    else:
        scorecard_variables = OrderedDict()

        req = sr.scorecard_variables.replace('TABLE1', table_name % table_date)

        CUR_CLIENT_TYPE = 'New' if cur_client_type == 'Новый' else 'Repeat'
        req = req.replace("SCORECARD1", cur_scorecard).replace("BUSINESS_TYPE1", cur_business_type).replace("CLIENT_TYPE1", CUR_CLIENT_TYPE)
        try:
            res = execute_db(c.db_d, req, pr=True)
            logger.info(cur_log + '\n' + req + '\n')
            variables = pd.DataFrame(res[1:], columns=res[0])
            variables.columns = [x.lower() for x in variables.columns]

            variables = variables.loc[:, ['month', 'variable', 'point', 'allappl', 'finappl', 
                'allshare', 'finshare', 'finrate', 'indicator', 'indicatortype']]

            if not variables.empty:

                # check coef=0 group
                for month in list(variables['month'].unique()):
                    temp = variables[variables['month'] == month]
                    for var in list(temp['variable'].unique()):
                        if 0.0 not in list(temp[temp['variable'] == var]['point']):
                            print(var, 'no null here')
                            variables = variables.append(pd.DataFrame([[month, var] + [0]*7 + [temp['indicatortype'].values[0]]], columns=list(variables.columns)))

                variables = variables.sort_values(by=['variable', 'point'])
                variables.index = range(len(variables))


                variables.fillna(0, inplace=True)
                for col in ['allappl', 'finappl']:
                    variables[col] = variables[col].astype(int)
                variables = variables.round({k:3 for k in ['allshare', 'finshare', 'finrate', 'indicator']})
                
                dates = sorted(list(variables['month'].unique()), reverse=True)
                dates = dates[:min(n_months, variables['month'].nunique())]

                for date in dates:
                    # scorecard_variables[date] = variables[variables['month'] == date].loc[:, 'variable':].to_html(index=False)
                    indicator_name = variables[variables['month'] == date]['indicatortype'].values[0]
                    temp = variables[variables['month'] == date].loc[:, 'variable':]
                    temp.rename(columns={'indicator': indicator_name}, inplace=True)
                    temp.drop(['indicatortype'], axis=1, inplace=True)
                    scorecard_variables[date] = temp.to_html(index=False)
                    del temp

                print(variables.head(10))
            else:
                # scorecard_variables = ''
                pass
        except Exception as e:
            print(e)
            # scorecard_variables = ''
            pass

        # save dict
        with open(temp_files + sm_base + '/scorecard_variables/' + cur_file, 'wb') as file:
            pickle.dump(scorecard_variables, file)
    
    ################################################################
    #                     CURRENT SCORECARDS                       #
    ################################################################

    scorecards_db = Scorecard.objects.filter(working=True).order_by('business_type', 'client_type')
    # scorecards_db = Scorecard.objects.all()
    scorecards_db_list = []
    for el in scorecards_db:
        scorecards_db_list.append([str(el.business_type), str(el.client_type), str(el.name)])
        # print(str(el.business_type), str(el.client_type), str(el.name))

    ################################################################
    #                     DELETE OLD FILES                         #
    ################################################################

    old_file = temp_files + sm_base + 'scorecards_data/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + datetime.strftime(prev_week_end - timedelta(days=7), "%Y_%m_%d") + '.csv'
    try:
        os.remove(old_file)  # delete main data for risk groups
    except: pass

    old_file = temp_files + sm_base + 'scorecards_list/' + 'scorecards_' + datetime.strftime(prev_week_end - timedelta(days=7), "%Y_%m_%d") + '.csv'
    try:
        os.remove(old_file)  # delete csv scorecards list
    except: pass

    old_file = temp_files + sm_base + 'scorecards_list/' + 'scorecards_by_python_' + datetime.strftime(prev_week_end - timedelta(days=7), "%Y_%m_%d") + '.pkl'
    try:
        os.remove(old_file)  # delete pkl scorecards by python list
    except: pass

    months_ago = 1
    old_file_1 = temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + str(now.year) + '-' + '0'*(2 - len(str(now.month - months_ago))) + str(now.month - months_ago) + '_scores.csv'
    old_file_2 = temp_files + sm_base + 'scorecards_dev_bins/' + cur_scorecard + '_' + str(now.year) + '-' + '0'*(2 - len(str(now.month - months_ago))) + str(now.month - months_ago) + '_scores_calc.xlsx'
    for file in [old_file_1, old_file_2]:
        try:
            os.remove(file)  # delete old scorecard's scores bins
        except: pass

    # delete old gini (keep only two last files)
    date_old = str(now.year) + '-' +  '0'*(2 - len(str(now.month - (months_ago_gini + 1) - 1))) + str(now.month - (months_ago_gini + 1) - 1)
    old_file = temp_files + sm_base + 'scorecards_gini/' + 'gini_basis_' + date_old + '.csv'
    # print(old_file)
    try:
        os.remove(old_file)  # delete
    except: pass

    # delete old variables effectiveness (keep only two last files)
    date_old = str(now.year) + '-' +  '0'*(2 - len(str(now.month - (months_ago_gini + 1) - 1))) + str(now.month - (months_ago_gini + 1) - 1)
    old_file = temp_files + sm_base + 'scorecard_variables/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_variables_effectiveness_' + date_old + '.pkl'
    # print(old_file)
    try:
        os.remove(old_file)  # delete
    except: pass

    # delete logs
    all_log_files = os.listdir(temp_files + sm_base + 'log/')
    sec_in_day = 86400
    life_time = 14  # days

    for path in all_log_files:
        f_time = os.stat(os.path.join(temp_files + sm_base + 'log/', path)).st_mtime
        now = time.time()
        if f_time < now - life_time * sec_in_day:
            try:
                os.remove(temp_files + sm_base + 'log/' + path)
            except: pass

    # dirty delete
    for root_path in ['scorecards_data', 'scorecards_list', 'scorecards_dev_bins']:
        all_log_files = os.listdir(temp_files + sm_base + root_path + '/')
        sec_in_day = 86400
        life_time = 45  # days

        for path in all_log_files:
            f_time = os.stat(os.path.join(temp_files + sm_base + root_path + '/', path)).st_mtime
            now = time.time()
            if f_time < now - life_time * sec_in_day:
                try:
                    os.remove(temp_files + sm_base + root_path + '/' + path)
                except: pass


    # --------------------------------------------------------------

    logger.removeHandler(fh)

    print('------------- end --------------', cur_business_type, cur_client_type, cur_scorecard)

    if requests_out:
        with open(temp_files + sm_base + '/requests.pkl', 'wb') as file:
            pickle.dump(requests_out, file)

    ################################################################
    #                         FILTER DATA                          #
    ################################################################

    json_mean_indicators = [x for x in json_mean_indicators if x['value'] > 0 or x['date'] == min_date]


    # n_last = 1
    # json_gini = json_gini[:-n_last]
    # json_data_f = json_data_f[:-n_last]
    # json_data_all = json_data_all[:-n_last]
    # json_mean_scores = json_mean_scores[:-n_last]
    # json_mean_indicators = json_mean_indicators[:-n_last]
    # json_scores_share = json_scores_share[:-n_last]
    # json_scores = json_scores[:-n_last]
    print('privet Andrey')

    # --------------------------------------------------------------

    return render(request, 'scorecard_monitoring2.html', {
        'requests_out': requests_out,

        'json_data_f': json.dumps(json_data_f), 
        'json_gini': json.dumps(json_gini), 
        'json_data_all': json.dumps(json_data_all),
        'json_scores': json.dumps(json_scores), 
        'json_mean_scores': json.dumps(json_mean_scores), 
        'json_mean_indicators': json.dumps(json_mean_indicators),

        'json_scores_share': json.dumps(json_scores_share),

        'cur_scorecard': cur_scorecard, 
        'cur_business_type': cur_business_type, 
        'cur_client_type': cur_client_type, 
        'scorecards_db': scorecards_db, 
        'scorecards_db_list': scorecards_db_list, 
        'date_base': date_base,

        'scorecard_variables': scorecard_variables
        })


@login_required
# calculate gini locally
def stacked_to_norm_example(request):


    cur_scorecard_obj = Scorecard.objects.filter(name='MainScore_201709_Core_New_EquifaxHit', business_type='Core')[0] # MainScore_201709_Core_New_EquifaxHit dn_pd_eq_1217
    cur_scorecard = str(cur_scorecard_obj.name)
    cur_business_type = str(cur_scorecard_obj.business_type)
    cur_client_type = str(cur_scorecard_obj.client_type)


    now = datetime.now()

    prev_week_end = now - timedelta(days=(now.weekday() + 1))
    # print(prev_week_end)

    ################################################################
    #                          INIT DIRS                           #
    ################################################################

    sm_base = "/scorecard_monitoring/"

    cur_log = datetime.strftime(now, '%H_%M_%S')


    ################################################################
    #                        RISK GROUPS                           #
    ################################################################

    cur_file = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + datetime.strftime(prev_week_end, "%Y_%m_%d") + '.csv'

    if cur_file in os.listdir(temp_files + sm_base + 'scorecards_data/'):

        file_size = os.stat(temp_files + sm_base + 'scorecards_data/' + cur_file).st_size / (1024 ** 2)  # Get file size
        if file_size > 50:  # If file size more than threshold
            data_main = pd.DataFrame()  # Create result data frame
            chunks = pd.read_table(temp_files + sm_base + 'scorecards_data/' + cur_file, chunksize=1000, iterator=True, encoding='cp1251', sep='\t')
            for chunk in chunks:  # For each part in parts
                data_main = pd.concat([data_main, chunk], axis=0)  # Join file parts

        else:
            data_main = pd.read_csv(temp_files + sm_base + 'scorecards_data/' + cur_file, encoding='cp1251', sep='\t')
        print(cur_log + ' data for ' + cur_scorecard +' loaded from ' + cur_file)
   
    else:

        # cur_business_type = scorecards[scorecards['SCORECARD'] == cur_scorecard]['BUSINESS_TYPE'].values[0]
        # cur_client_type = scorecards[scorecards['SCORECARD'] == cur_scorecard]['CLIENT_TYPE'].values[0]

        # Init business type
        if str(cur_business_type) == 'Core':
            cur_business_type_f = "Core', 'from_SPR7_to_SPR4', 'from_SPR15_to_SPR4"
        else:
            cur_business_type_f = str(cur_business_type)

        # Init score field
        if cur_scorecard in scorecards_by_python:
            cur_score = 'MainPROBABballs as MainSCOREballs'  # MainSCOREballs, MainPROBABballs,
        else:
            cur_score = 'MainSCOREballs' 

        cond1 = '1=1'  # special condition

        req = sr.scorecards_rg_indicators
        req = req.replace("SCORECARD1", str(cur_scorecard)).replace("BUSINESS_TYPE1", cur_business_type_f).replace("CLIENT_TYPE1", str(cur_client_type)).replace("COND1", cond1).replace("SCORE1", cur_score)
        res = execute_db(c.db_p, req, pr=True)
        
        label = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + 'financed'
        if label not in requests_out.keys():
            requests_out[label] = req.replace('1=1', 'Financed = 1')
        data_main = pd.DataFrame(res[1:], columns=res[0])
        data_main.replace({None: np.nan}, inplace=True)
        if not data_main.empty:
            data_main.to_csv(temp_files + sm_base + 'scorecards_data/' + cur_file, encoding='cp1251', sep='\t', index=False)

    
    # print(data.head())
    # try:
    #     data_main['MainSCOREballs'] = data_main['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
    #     data_main = data_main.replace({'MainSCOREballs': {'NULL': -1}})
    #     data_main['MainSCOREballs'] = data_main['MainSCOREballs'].astype(float)
    # except Exception as e:
    #     print(e)

    # data_main['MainSCOREballs'].fillna(-1, inplace=True)
    # data_main = data_main.replace({'MainSCOREballs': {'NULL': -1}})

    # data_main['MainSCOREballs'] = data_main['MainSCOREballs'].astype(float)

    ################################################################
    #                       FINANCED PART                          #
    ################################################################

    data_f = data_main[data_main['Financed'] == 1].copy()
    data_f_g = data_f.groupby(['CRM_CREATION_Month', 'RISK_GROUP']).agg({'UCDB_ID': ['count']})
    data_f = data_f_g.transpose().reset_index(level=0, drop=True).transpose().reset_index()
    data_f.rename(columns={'CRM_CREATION_Month': 'date'}, inplace=True)

    # print(data_f)
    
    # print(data_f)

    # json_data = [
    #       { 'date': "2018-01", 'A': 60, 'B': 15, 'C': 9, 'D': 6},
    # ]

    # json_gini = [
    #       { 'date': "2018-01", 'value': 0.65 },
    # ]

    json_data_f = []

    for date in sorted(data_f['date'].unique()):
        temp = data_f[data_f['date'] == date]
        lst = {'date': date}
        for el in zip(temp['RISK_GROUP'], temp['count']):
            lst[el[0]] = el[1]
            for rg in list('ABCD'):
                if rg not in lst.keys():
                    lst[rg] = 0
        json_data_f.append(lst)

    # print(json_data_f)

    ################################################################
    #                      ALL APPLICATIONS                        #
    ################################################################

    data_all = data_main.copy()
    data_all_g = data_all.groupby(['CRM_CREATION_Month', 'RISK_GROUP']).agg({'UCDB_ID': ['count']})
    data_all = data_all_g.transpose().reset_index(level=0, drop=True).transpose().reset_index()
    data_all.rename(columns={'CRM_CREATION_Month': 'date'}, inplace=True)
    
    # print(data_all.head())
    json_data_all = []

    for date in sorted(data_all['date'].unique()):
        temp = data_all[data_all['date'] == date]
        lst = {'date': date}
        for el in zip(temp['RISK_GROUP'], temp['count']):
            lst[el[0]] = el[1]
            for rg in list('ABCD'):
                if rg not in lst.keys():
                    lst[rg] = 0
        json_data_all.append(lst)

    # print(json_data_f)

    ################################################################
    #                     SCORE STABILITY GROUP                    #
    ################################################################


    #  ----------------------- v.2 (nth week qcut)  ---------------

    from_month_number = 2

    # data_f = data_main[data_main['Financed'] == 1].copy()
    data_ss = data_main.copy()
    # data_ss = data_ss[(~data_ss['MainSCOREballs'].isin(['NULL']))]
    data_ss = data_ss[(data_ss['MainSCOREballs'].notnull()) & (~data_ss['MainSCOREballs'].isin(['NULL']))]

    try:
        data_ss['MainSCOREballs'] = data_ss['MainSCOREballs'].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
        data_ss['MainSCOREballs'] = data_ss['MainSCOREballs'].astype(float)
    except Exception as e:
        print(e)

    all_dates = sorted(list(data_ss['CRM_CREATION_Month'].unique()))

    # check scorecard's work 
    if len(all_dates) > 0:

        try:
            date_base = all_dates[from_month_number - 1]
        except:
            date_base = all_dates[0]

        
        # cur_file = cur_scorecard + '_from_'+ str(date_base) +'_month_bins.pkl'
        cur_file = cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_from_'+ str(date_base) +'_month_bins.pkl'

        # load bins
        if cur_file in os.listdir(temp_files + sm_base + 'scorecards_dev_bins/'):
            with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'rb') as file:
                bins = pickle.load(file)
            logger.info(cur_log + ' bins for ' + cur_scorecard +' loaded from ' + cur_file)

        else:

            df = data_ss[(data_ss['CRM_CREATION_Month'] == date_base)]
        
            _, bins = pd.qcut(df['MainSCOREballs'], 10, retbins=True, duplicates='drop')
            bins = list(bins)
            bins[0] = -np.inf
            bins[-1] = np.inf
            df[date_base + '_dev'] = pd.cut(df['MainSCOREballs'], bins=bins)
            # get dev scores
            data = df[date_base + '_dev'].value_counts(normalize=True).sort_index().to_frame()  
            data.index.rename('group', inplace=True)
            data.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_from_'+ str(date_base) +'_month_dev_scores.csv', sep='\t', encoding='cp1251')

            with open(temp_files + sm_base + 'scorecards_dev_bins/' + cur_file, 'wb') as file:
                pickle.dump(bins, file) 

        if bins:
            bins[0] = -np.inf
            bins[-1] = np.inf

        data_bins = pd.read_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_from_'+ str(date_base) +'_month_dev_scores.csv', sep='\t', encoding='cp1251', index_col='group')
        
        # print(data_bins)

        dates_ss = all_dates[(all_dates.index(date_base) + 1):]
        # print('dates_ss', dates_ss)

        if dates_ss:

            for date in dates_ss:

                temp = data_ss[data_ss['CRM_CREATION_Month'] == date]
                temp['N'] = pd.cut(temp['MainSCOREballs'], bins=bins)
                # get dev scores
                data_bins[date] = temp['N'].value_counts(normalize=True).sort_index().values

            # print(data_bins)
            if not data_bins.empty:
                data_bins.to_csv(temp_files + sm_base + 'scorecards_dev_bins/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + max(dates_ss) + '_scores.csv', sep='\t', encoding='cp1251')

            json_scores = []
            for date in dates_ss:
                data_bins[date+'_d'] = data_bins[date]-data_bins[date_base + '_dev']
                data_bins[date+'_w'] = np.log(data_bins[date]/data_bins[date_base + '_dev'])
                data_bins = data_bins.replace({date+'_w': {np.inf: 0, -np.inf: 0}})
                data_bins[date+'_s'] = data_bins[date+'_d']*data_bins[date+'_w'] 
                ssi = data_bins[date+'_s'].sum()
                json_scores.append({'date': date, 'value': round(ssi, 5)})

            # print(json_scores)
            # print(data_bins)
            data_bins.to_excel(temp_files + sm_base + 'scorecards_dev_bins/' + cur_business_type + '_' + cur_client_type + '_' + cur_scorecard + '_' + max(dates_ss) + '_scores_calc.xlsx')
        else:
            json_scores = []
    # scorecard too new
    else:
        json_scores = []
        date_base = ''


    return render(request, 'stacked_to_norm_example.html', {

        'json_data_f': json.dumps(json_data_f), 
        'json_data_all': json.dumps(json_data_all),
        'json_scores': json.dumps(json_scores),

        'cur_scorecard': cur_scorecard, 
        'cur_business_type': cur_business_type, 
        'cur_client_type': cur_client_type, 
        'date_base': date_base,


        })