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


def index(request):
    return render(request, 'index.html')

def index_with_map(request):

    now = datetime.now()
    prev_week_end = now - timedelta(days=(now.weekday() + 1))

    json_region_fpd = {}

    if request.user.is_authenticated:
        
        # Checking cache
        cur_file = 'map_fpd_' + datetime.strftime(prev_week_end, "%Y_%m_%d") + '.json'

        # If the file exists load it
        if cur_file in os.listdir(temp_files + '/map/' + 'fpd/'):
            
            # Load jsons
            with open(temp_files + '/map/' + 'fpd/' + cur_file) as infile:
                json_region_fpd  = json.load(infile)

            print(cur_file, 'was loaded')

        # If the file doesn't exist make request to db       
        else:

            req = '''
                select 
                    substring(p.RegionResident, 1, 2) as region_id
                    , sum(isnull(a.FPD, 0)) as fpd_sum
                    , count(a.UCDB_ID) as count
                    --, cast(sum(isnull(a.FPD, 0)) as float) / count(a.UCDB_ID) as fpd
                from dss.spr.spr_all_data a
                left join [DSS].[dbo].[SPR_R_ParticipantInfo] p on p.UCDB_ID = a.UCDB_ID and p.Role = 'Borrower'
                where 1=1
                    and a.Financed = 1
                    and p.RegionResident is not null
                    and substring(p.RegionResident, 1, 2) != '00'
                group by substring(p.RegionResident, 1, 2)
                order by substring(p.RegionResident, 1, 2) desc
            '''
            res = execute_db(c.db_p, req, pr=True)
            df = pd.DataFrame(res[1:], columns=res[0])
            df.columns = [x.lower() for x in df.columns]

            df['fpd'] = df['fpd_sum']/df['count']

            print(df.head())

            json_region_fpd = []

            for i in range(len(df)):
                # json_region_fpd.append({'region_id': df.loc[i, 'region_id'], 'count': df.loc[i, 'count'], 'fpd': df.loc[i, 'fpd']})
                json_region_fpd.append({'region_id': df.loc[i, 'region_id'], 'fpd': df.loc[i, 'fpd'], 'count': int(df.loc[i, 'count'])})

            print(json_region_fpd)

            # Save json
            with open(temp_files + '/map/' + 'fpd/' + cur_file, 'w') as outfile:
                json.dump(json_region_fpd, outfile)

            ################################################################
            #                     DELETE OLD FILES                         #
            ################################################################

            old_file = temp_files + '/map/fpd/map_fpd_' + datetime.strftime(prev_week_end - timedelta(days=21), "%Y_%m_%d") + '.json'
            try:
                os.remove(old_file)  # delete main data for risk groups
            except: pass

    return render(request, 'index_with_map.html', {'json_region_fpd': json.dumps(json_region_fpd)})


@login_required
def family(request):
    return render(request, 'family.html')


@login_required
def index_risks(request):
    return render(request, 'index_risks.html')


@login_required
def index_stage_sankey(request):
    return render(request, 'index_stage_sankey.html')


@login_required
def index_marketing(request):
    return render(request, 'index_marketing.html')


def mosaic(request):
    return render(request, 'mosaic.html')


# def map(request):
#     return render(request, 'map.html')


@login_required
def map_fpd_juicy(request):

     # Checking cache
    cur_fpd = 'map_fpd_juicy' + '.json'

    # If the file exists load it
    if cur_fpd in os.listdir(temp_files + '/map/' + 'fpd/'):
        
        # Load jsons
        with open(temp_files + '/map/' + 'fpd/' + cur_fpd) as infile:
            json_map_fpd  = json.load(infile)

        print(cur_fpd, 'was loaded')

    # If the file doesn't exist make request to db       
    else:

        req = '''
            select --top 50
                a.UCDB_ID
                , jlat.value as lat
                , jlon.value as lon
            from dss.spr.spr_all_data a
            left join [DSS].[dbo].[SPR_JUICY_RESPONSE] jlat on jlat.UCDB_ID = a.UCDB_ID and jlat.[KEY] = 'IpLatitude'
            left join [DSS].[dbo].[SPR_JUICY_RESPONSE] jlon on jlon.UCDB_ID = a.UCDB_ID and jlon.[KEY] = 'IpLongitude'
            where 1=1
            and a.financed = 1

            and jlat.value is not null
            and jlat.value != '0'
            and jlon.value != '0'

            and a.fpd = 1
        '''
        res = execute_db(c.db_p, req, pr=True)
        df = pd.DataFrame(res[1:], columns=res[0])
        df.columns = [x.lower() for x in df.columns]

        print(df.head())

        json_map_fpd = []

        for i in range(len(df)):
            json_map_fpd.append({'ucdb_id': df.loc[i, 'ucdb_id'], 'lat': df.loc[i, 'lat'], 'lon': df.loc[i, 'lon']})

        print(json_map_fpd)
        # Save json
        with open(temp_files + '/map/' + 'fpd/' + cur_fpd, 'w') as outfile:
            json.dump(json_map_fpd, outfile)

    return render(request, 'map/map_fpd_juicy.html', {'json_map_fpd': json.dumps(json_map_fpd)})


@login_required
def map_fpd_canvas(request):

     # Checking cache
    cur_fpd = 'map_fpd_juicy' + '.json'

    # If the file exists load it
    if cur_fpd in os.listdir(temp_files + '/map/' + 'fpd/'):
        
        # Load jsons
        with open(temp_files + '/map/' + 'fpd/' + cur_fpd) as infile:
            json_map_fpd  = json.load(infile)

        print(cur_fpd, 'was loaded')

    # If the file doesn't exist make request to db       
    else:

        req = '''
            select --top 50
                a.UCDB_ID
                , jlat.value as lat
                , jlon.value as lon
            from dss.spr.spr_all_data a
            left join [DSS].[dbo].[SPR_JUICY_RESPONSE] jlat on jlat.UCDB_ID = a.UCDB_ID and jlat.[KEY] = 'IpLatitude'
            left join [DSS].[dbo].[SPR_JUICY_RESPONSE] jlon on jlon.UCDB_ID = a.UCDB_ID and jlon.[KEY] = 'IpLongitude'
            where 1=1
            and a.financed = 1

            and jlat.value is not null
            and jlat.value != '0'
            and jlon.value != '0'

            and a.fpd = 1
        '''
        res = execute_db(c.db_p, req, pr=True)
        df = pd.DataFrame(res[1:], columns=res[0])
        df.columns = [x.lower() for x in df.columns]

        print(df.head())

        json_map_fpd = []

        for i in range(len(df)):
            json_map_fpd.append({'ucdb_id': df.loc[i, 'ucdb_id'], 'lat': df.loc[i, 'lat'], 'lon': df.loc[i, 'lon']})

        print(json_map_fpd)
        # Save json
        with open(temp_files + '/map/' + 'fpd/' + cur_fpd, 'w') as outfile:
            json.dump(json_map_fpd, outfile)

    return render(request, 'map/map_fpd_canvas.html', {'json_map_fpd': json.dumps(json_map_fpd)})


@login_required
def map_fpd(request):

    now = datetime.now()
    prev_week_end = now - timedelta(days=(now.weekday() + 1))

    # Checking cache
    cur_file = 'map_fpd_' + datetime.strftime(prev_week_end, "%Y_%m_%d") + '.json'

    # If the file exists load it
    if cur_file in os.listdir(temp_files + '/map/' + 'fpd/'):
        
        # Load jsons
        with open(temp_files + '/map/' + 'fpd/' + cur_file) as infile:
            json_region_fpd  = json.load(infile)

        print(cur_file, 'was loaded')

    # If the file doesn't exist make request to db       
    else:

        req = '''
            select 
                substring(p.RegionResident, 1, 2) as region_id
                , sum(isnull(a.FPD, 0)) as fpd_sum
                , count(a.UCDB_ID) as count
                --, cast(sum(isnull(a.FPD, 0)) as float) / count(a.UCDB_ID) as fpd
            from dss.spr.spr_all_data a
            left join [DSS].[dbo].[SPR_R_ParticipantInfo] p on p.UCDB_ID = a.UCDB_ID and p.Role = 'Borrower'
            where 1=1
                and a.Financed = 1
                and p.RegionResident is not null
                and substring(p.RegionResident, 1, 2) != '00'
            group by substring(p.RegionResident, 1, 2)
            order by substring(p.RegionResident, 1, 2) desc
        '''
        res = execute_db(c.db_p, req, pr=True)
        df = pd.DataFrame(res[1:], columns=res[0])
        df.columns = [x.lower() for x in df.columns]

        df['fpd'] = df['fpd_sum']/df['count']

        print(df.head())

        json_region_fpd = []

        for i in range(len(df)):
            # json_region_fpd.append({'region_id': df.loc[i, 'region_id'], 'count': df.loc[i, 'count'], 'fpd': df.loc[i, 'fpd']})
            json_region_fpd.append({'region_id': df.loc[i, 'region_id'], 'fpd': df.loc[i, 'fpd'], 'count': int(df.loc[i, 'count'])})

        print(json_region_fpd)

        # Save json
        with open(temp_files + '/map/' + 'fpd/' + cur_file, 'w') as outfile:
            json.dump(json_region_fpd, outfile)

    ################################################################
    #                     DELETE OLD FILES                         #
    ################################################################

    old_file = temp_files + '/map/fpd/map_fpd_' + datetime.strftime(prev_week_end - timedelta(days=21), "%Y_%m_%d") + '.json'
    try:
        os.remove(old_file)  # delete main data for risk groups
    except: pass

    return render(request, 'map/map_fpd.html', {'json_region_fpd': json.dumps(json_region_fpd)})


@login_required
def map_russia(request):
    return render(request, 'map/map_russia.html')


@login_required
def map_offices(request):
    return render(request, 'map/map_offices.html')


@login_required
def maps(request):
    return render(request, 'index_map.html')


def page404(request):
    return render(request, '404.html')


@login_required
def vintages(request):

    save_event(request)

    print(request.POST)

    all_values_label = 'ALL'

    # fields = ['segment']  # Total features list for segmentation
    fields = ['client_type', 'business_type', 'execution_key', 'request_channel', 'branch', 'segment']  # Total features list for segmentation
    filters = {}  # dict for all values for filtering


    if request.POST:
        for field in fields:
            try:
                filters[field] = request.POST.getlist(field + '[]')
            except: pass

        if filters:
            for k, v in filters.items():
                print(k, v)
    else:
        for field in fields:
            filters[field] = []
    
    filtering_values = {}  # dict for all possible values
    
    # for field in fields:
    #     filtering_values[field] = []

    # df = pd.read_csv(os.path.join(temp_files, 'vintages/vintages_example.csv'), sep='\t')   
    # df = pd.read_csv(os.path.join(temp_files, 'vintages/Vintage2018-09-14.csv'), sep='\t', encoding='cp1251')   
    df = pd.read_csv(os.path.join(temp_files, 'vintages/Vintage2016-01-01-2018-01-01.csv'), sep=';', encoding='cp1251')
    df.columns = [x.lower() for x in df.columns]

    # !!!!!!!!!!!!!!!!!!!!!!!!
    df.rename(columns={'seg': 'segment', 'reuest_cnannel': 'request_channel', 'executionkey': 'execution_key'}, inplace=True)

    df['vintage_date'] = pd.to_datetime(df['vintage_date'])
    df['mob'] = df['mob'].astype(int)
    df['la_90_final'] = df['la_90_final'].astype(float)

    print(df.head())
    print(df.dtypes)

    # get all possible values for filtering
    for field in fields:
        filtering_values[field] = sorted(list(df[field].dropna().unique()))
        filtering_values[field].insert(0, all_values_label)

    # filtering data
    if filters:
        for k, v in filters.items():
            if len(v) > 0:
                if all_values_label not in v or (all_values_label in v and len(v) > 1):
                    df = df[df[k].isin(v)]

    if len(df) > 0:

        error = False

        # get fields for segmentation
        grouped_field = []
        for k, v in filters.items():
            if v:
                grouped_field.append(k)
                # df = df[df[k] == v]
        grouped_field.append('vintage_date') # add default value for segmentation

        print('total grouped fields are:', grouped_field)

        for col in list(df.columns[~df.columns.isin(grouped_field + ['mob'])]):
            try:
                df[col] = df[col].map(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)
                df[col] = df[col].astype(float)
            except:
                pass

        # # simple
        # df1 = df.groupby(grouped_field + ['mob'])['LA_LC', 'la_90_final', 'LA'].agg(np.sum).reset_index()
        # data = pd.crosstab([df1[el] for el in grouped_field], df1['mob'], df1['la_90_final'] / df1['LA'], aggfunc='sum')
        # # data = data.transpose().reset_index(level=0, drop=True).transpose().reset_index()
        # data.columns =[el for el in map(int, data.columns)]
        # mobs = list(data.columns)
        # data = data.reset_index()

        # with LA_LC
        add_numeric_fields = []
        calc_field = 'la'
        groupby_columns = ['la_90_final', calc_field] + add_numeric_fields
        df1 = df.groupby(grouped_field + ['mob'])[groupby_columns].agg(np.sum).reset_index()
        for col in add_numeric_fields + [calc_field]:
            df1[col] = round(df1[col], 3)

        data = pd.crosstab([df1[el] for el in grouped_field + [calc_field] + add_numeric_fields], df1['mob'], df1[calc_field + '_90_final'] / df1[calc_field], aggfunc='sum')
        # print(data.head())
        data = data.transpose().reset_index(level=0, drop=True).transpose().reset_index()

        # mobs = list(data.columns[~data.columns.isin(grouped_field + ['LA_LC'])])
        mobs = list(range(13))
        # data = data.reset_index()

        print(data.head())
        # data.to_excel(os.path.join(temp_files, 'vintages/data2.xlsx'), index=False)

        # print(data.columns)

        # проверка созревших данных
        months_ago = 3 + 2
        data = data[data.vintage_date < (datetime.now() - timedelta(days=30*months_ago))]

        for mob in mobs:
            last_date = datetime.now() - timedelta(days=30*(mob + 1))
            data.loc[data.vintage_date > last_date, mob] = 0


        json_vintages = []

        for mob in mobs:
            for i in range(len(data)):
                label = datetime.strftime(data.loc[i, 'vintage_date'], '%Y_%m_%d')
                for k, v in filters.items():
                    if v:
                        if k == 'execution_key':
                            label += '_EK_' + str(data.loc[i, k])
                        elif k == 'branch':
                            label += '_' + str(data.loc[i, k]).replace(' ', '_')
                        else:
                            label += '_' + str(data.loc[i, k])
                json_vintages.append({'mob': str(mob), 'label': label, 'value': data.loc[i, mob]})
            
            # lst = {'mob': str(mob)}
            # for date in list(data.index):
            #     lst['value_' + datetime.strftime(date, '%Y_%m_%d')] = data.loc[date, mob]
            # json_vintages.append(lst)
        # print(json_vintages[0])
        # json_vintages = []

        # print(data.dtypes)

        # int_frmt = lambda x: '{:,}'.format(x)
        # float_frmt = lambda x: '{:.2f}'.format(x) if x > 1e3 else '{:.5f}'.format(x)
        # data = data.to_html(index=False, formatters={'LA_LC': float_frmt})

        pd.options.display.float_format = lambda x: '{:.2f}'.format(x) if x > 1e3 else '{:.5f}'.format(x)
        data = data.to_html(index=False)

    else:
        data = ''
        error = True
        json_vintages = []

    json_vintages = [x for x in json_vintages if (x['value'] > 0 and float(x['mob']) >= 4) or (float(x['mob']) < 4)]
    
    return render(request, 'vintages.html', {'data': data, 'json_vintages': json.dumps(json_vintages), 'filters': filters, 
        'filtering_values': filtering_values, 'fields_list': fields, 'fields_json': json.dumps(fields), 'error': error})


def draw_graph_example(request):

    # with open(temp_files + '/graph_example.json', 'w') as outfile:
    #         json.dump(graph, outfile)

    with open(temp_files + '/graph_example.json') as infile:
            json_data  = json.load(infile)

    # return render_to_response('graph_example.html', {'my_graph': json.dumps(graph)}, RequestContext(request))
    return render(request, 'graph_example.html', {'my_graph': json.dumps(json_data)})


@login_required
def draw_graph_contacts(request):

    save_event(request)

    '''
    UserPhoneNumber - номер клиента
    PhoneNumber - номер контакта
    AgentPhoneNumber - номер агента
    '''

    cur_file_json = 'test_json.json'
    cur_file_data = 'test_data.csv'

    min_number_length = 5
    only_linked_ids = False

    req = sr.ma_contacts

    # If the file exists load it
    if cur_file_json in os.listdir(temp_files + '/contacts_graph/'):
        
        # Load json 
        with open(temp_files + '/contacts_graph/' + cur_file_json) as infile:
            json_data  = json.load(infile)

        print(cur_file_json, 'was loaded')
        logger.info(cur_file_json + ' was loaded')

        print(len(json_data['nodes']))
        print(len(json_data['links']))

        if only_linked_ids:
            for node in json_data['nodes']:
                if node['id'] not in [x['source'] for x in json_data['links']] and node['id'] not in [x['target'] for x in json_data['links']]:
                    json_data['nodes'].remove(node)

                for link in json_data['links']:
                    if node['id'] == link['source'] and node['id'] == link['target']:
                        json_data['nodes'].remove(node)

        print(len(json_data['nodes']))
        print(len(json_data['links']))

    else:

        if cur_file_data not in os.listdir(temp_files + '/contacts_graph/'):
            
            # res = execute_db('mck-p-dwh', req, pr=True)
            # df = pd.DataFrame(res[1:], columns=res[0])

            conn = pymssql.connect('mck-p-dwh')
            df = pd.read_sql(req, conn)
            conn.close()

            # df = pd.read_excel(temp_files + '/contacts_graph/contacts_fpd.xlsx')
                        
            print(df.head(20))

            if not df.empty:
                df.to_csv(temp_files + '/contacts_graph/' + cur_file_data, sep='\t')
        else:
            df = pd.read_csv(temp_files + '/contacts_graph/' + cur_file_data, sep='\t', encoding='cp1251')

    
        # file = 'contacts_2_weeks_d.csv'
        # file = 'shorokh_mac_180818_070918.csv'
        # df = pd.read_csv(os.path.join(temp_files, file), sep=';', encoding='cp1251')
        # df = pd.read_csv(os.path.join(temp_files, file), sep='\t', encoding='cp1251')
        
        # cur_file = min(df['CreatedDT']).split(' ')[0] + '_' + max(df['CreatedDT']).split(' ')[0] + '_len_' + str(len(df)) + '.json'

        ############################################
        #                  FILTER                  #
        ############################################



        df = df[~df[['UCDB_ID', 'ContactPhoneNumber']].duplicated()] 

        # df = df.iloc[:5000, :]
        # df = df[df['UCDB_ID'].isin([1037208699, 1037207740, 1036970547, 1036970812])] # example
        # df = df[df['UCDB_ID'].isin([1037007359, 1037195284, 1037225404, 1037096716])] # fpd4 
        # df = df[df['UCDB_ID'].isin([1037954548, 1037958848, 1038185568, 1038461765, 1038616567])] # fpd2 162k

        # df = df[df['UCDB_ID'].isin([1039606425, 1039368481, 1039444131, 1039560775])] # real

        df['ContactPhoneNumber'] = df['ContactPhoneNumber'].astype(str)
        df['UserPhoneNumber'] = df['UserPhoneNumber'].astype(str)
        df['UCDB_ID'] = df['UCDB_ID'].astype(str)

        # filtering
        for col in ['ContactPhoneNumber', 'UserPhoneNumber']:
            df[col] = df[col].map(lambda x: str(x).strip().replace(' ', '').replace('-', '')
                .replace('+7', '8').replace('+', '').replace('(', '').replace(')', '').replace('*', ''))
            df[col] = df[col].map(lambda x: re.sub('^%s' % "89", "9", x))
            df[col] = df[col].map(lambda x: re.sub('^%s' % "849", "49", x))

            df[col] = df[col].map(lambda x: re.sub('^%s' % "79", "9", x) if len(x) == 11 else x)
            # string[-len(pat):] + sub if string.endswith(pat) else string

            df[col] = df[col].map(lambda x: str(x).replace('.0', '') if '.0' in str(x) else x)

        df = df[df['ContactPhoneNumber'].str.len() > min_number_length]
        df = df[~df['ContactPhoneNumber'].str.contains('#')]

        df = df[~df[['UCDB_ID', 'ContactPhoneNumber']].duplicated()] 

        # df.drop_duplicates(inplace=True)
        
        links = []
        linked_ids = []

        dy = {}  # dict for labels offset

        #  connection via 3rd contact
        from itertools import combinations
        for number in sorted(list(df['ContactPhoneNumber'].unique())):
            contact_set = list(df[df['ContactPhoneNumber'] == number]['UCDB_ID'])
            if len(contact_set) > 1:
                contact_sets = [el for el in combinations(contact_set, 2)]
                # print(contact_sets, number)
                for el in contact_sets:
                    if el[0] != el[1]:  # check source != target
                        el_1 = str(min(el[0], el[1]))
                        el_2 = str(max(el[0], el[1]))
                        name_1 = str(df[(df['UCDB_ID'] == el_1) & (df['ContactPhoneNumber'] == number)]['Name'].values[0])
                        name_2 = str(df[(df['UCDB_ID'] == el_2) & (df['ContactPhoneNumber'] == number)]['Name'].values[0])
                        names = name_1 + ';' + name_2 if name_1 != name_2 else name_1
                        # print(name_1, name_2)
                        if el_1 + ':' + el_2 not in list(dy.keys()):
                            dy[el_1 + ':' + el_2] = 1
                        else:
                            dy[el_1 + ':' + el_2] += 1
                        if el_1 != el_2:
                            links.append({'source': el_1, 'target': el_2, "value": number, "names": names, "type": "3rd", "width": 1, "dy": dy[el_1 + ':' + el_2]})
                            linked_ids.append(el[0])
                            linked_ids.append(el[1])

        # connection from first view
        for id in list(df['UCDB_ID'].unique()):
            user_number = df[df['UCDB_ID'] == id]['UserPhoneNumber'].values[0]
            contacts_has_user_number = list(set(list(df[(df['UCDB_ID'] != id) & (df['ContactPhoneNumber'] == user_number)]['UCDB_ID'])))
            if contacts_has_user_number:
                print(id, user_number, contacts_has_user_number)
                linked_ids.append(id)
                linked_ids += contacts_has_user_number
                for el in contacts_has_user_number:
                    el_1 = min(id, el)
                    el_2 = max(id, el)
                    names = df[(df['UCDB_ID'].isin([el])) & (df['ContactPhoneNumber'] == user_number)]['Name'].values[0]
                    if el_1 + ':' + el_2 not in list(dy.keys()):
                        dy[el_1 + ':' + el_2] = 1
                    else:
                        dy[el_1 + ':' + el_2] += 1
                    if el_1 != el_2:
                        links.append({'source': el_1, 'target': el_2, "value": user_number, "names": names, "type": "1to1", "width": 3, "dy": dy[el_1 + ':' + el_2]})
        # print(links)

        linked_ids = sorted(list(set(linked_ids)))

        if only_linked_ids:
            nodes_ids = linked_ids
        else:
            nodes_ids = list(df['UCDB_ID'].unique())

        json_data = {

            "nodes": [{"id": id, "group": int(int(id) % 10), "number": df[df['UCDB_ID'] == id]['UserPhoneNumber'].values[0], 
            'status': df[df['UCDB_ID'] == id]['Status'].values[0], 'name': df[df['UCDB_ID'] == id]['ClientFirstName'].values[0] + ' '
            + df[df['UCDB_ID'] == id]['ClientLastName'].values[0]} for id in nodes_ids],
            "links": links
        }

        # Save json
        if len(json_data) > 0:
            with open(temp_files + '/contacts_graph/' + cur_file_json, 'w') as outfile:
                json.dump(json_data, outfile)

    # print(json_data)

    return render(request, 'graph_contacts.html', {'my_graph': json.dumps(json_data)})


@login_required
def financed_sankey(request):

    save_event(request)

    weeks_to_past = 20  # weeks count for option selection
    dt = datetime.now()
    cur_week_start = dt - timedelta(days=(dt.weekday() + 7))
    cur_week_end = cur_week_start + timedelta(days=6)
    cur_folder = '/scorecard_sankey/'

    if request.POST:
        week1 = str(request.POST['week_from'])
        week2 = str(request.POST['week_upto'])
    else:
        
        week1 = datetime.strftime(cur_week_start, "%Y-%m-%d") + ' --- ' + datetime.strftime(cur_week_end, "%Y-%m-%d")
        week2 = datetime.strftime(cur_week_start, "%Y-%m-%d") + ' --- ' + datetime.strftime(cur_week_end, "%Y-%m-%d")

    print(request.method, week1, week2)

    week_options = [datetime.strftime(cur_week_start, "%Y-%m-%d") + ' --- ' + datetime.strftime(cur_week_end, "%Y-%m-%d")]
    for i in range(weeks_to_past):
        week_start = cur_week_start - timedelta(days=7)
        week_end = week_start + timedelta(days=6)
        cur_week_start = week_start
        week_options.append(datetime.strftime(week_start, "%Y-%m-%d") + ' --- ' + datetime.strftime(week_end, "%Y-%m-%d"))

    linked_columns = ['CLIENT_TYPE', 'BUSINESS_TYPE', 'SCORECARD', 'RISK_GROUP']

    # Checking cache
    cur_file = 'rg_weeks_' + week1.split(' --- ')[0] + ' --- ' + week2.split(' --- ')[1] + '.json'
    # If the file exists load it
    if cur_file in os.listdir(temp_files + cur_folder):
        
        # Load json 
        with open(temp_files + cur_folder + cur_file) as infile:
            json_data  = json.load(infile)

        print(cur_file, 'was loaded')

    # If the file doesn't exist make request to db       
    else:
        req = sr.sankey_scorecards_rg_weeks
        req = req.replace("WEEK1", week1).replace("WEEK2", week2)
        res = execute_db(c.db_p, req, pr=True)
        df = pd.DataFrame(res[1:], columns=res[0])
        print(df)

        if not df.empty:

            # print(df)

            node_id = 0
            nodes_dict = {}

            json_data = {}
            json_data["nodes"] = []
            json_data["links"] = []

            for column in linked_columns:
                for field in sorted(list(df[column].unique())):
                    json_data["nodes"].append({"node": node_id, "name": field}) 
                    nodes_dict[field] = node_id
                    node_id += 1

            for i in range(len(linked_columns) - 1):

                grouped_columns = linked_columns[i:i + 2]
                group = df.groupby(grouped_columns).sum()

                for index in list(group.index):
                    # print(index, group.loc[index, "item_counts"])
                    json_data["links"].append({"source": nodes_dict[index[0]], "target": nodes_dict[index[1]], "value": int(group.loc[index, 'item_counts'])})

            # Save json
            with open(temp_files + cur_folder + cur_file, 'w') as outfile:
                json.dump(json_data, outfile)

        else:
            json_data = []


    # delete old files
    all_log_files = os.listdir(temp_files + '/scorecard_sankey/')
    sec_in_day = 86400
    life_time = 14  # days

    for path in all_log_files:
        f_time = os.stat(os.path.join(temp_files + '/scorecard_sankey/', path)).st_mtime
        now = time.time()
        if f_time < now - life_time * sec_in_day:
            try:
                os.remove(temp_files + '/scorecard_sankey/' + path)
            except: pass

    return render(request, 'financed_sankey.html', {'json': json.dumps(json_data), 'week_options': week_options, 'week1': week1, 'week2': week2})


@login_required
def entrance_sankey(request):

    save_event(request)

    weeks_to_past = 30  # weeks count for option selection
    dt = datetime.now()
    cur_week_start = dt - timedelta(days=(dt.weekday() + 7))
    cur_week_end = cur_week_start + timedelta(days=6)
    cur_folder = '/entrance_sankey/'

    if request.POST:
        week1 = str(request.POST['week_from'])
        week2 = str(request.POST['week_upto'])
    else:
        
        week1 = datetime.strftime(cur_week_start, "%Y-%m-%d") + ' --- ' + datetime.strftime(cur_week_end, "%Y-%m-%d")
        week2 = datetime.strftime(cur_week_start, "%Y-%m-%d") + ' --- ' + datetime.strftime(cur_week_end, "%Y-%m-%d")

    print(request.method, week1, week2)

    week_options = [datetime.strftime(cur_week_start, "%Y-%m-%d") + ' --- ' + datetime.strftime(cur_week_end, "%Y-%m-%d")]
    for i in range(weeks_to_past):
        week_start = cur_week_start - timedelta(days=7)
        week_end = week_start + timedelta(days=6)
        cur_week_start = week_start
        week_options.append(datetime.strftime(week_start, "%Y-%m-%d") + ' --- ' + datetime.strftime(week_end, "%Y-%m-%d"))

    linked_columns = ['CLIENT_TYPE', 'EntranceChannel', 'RequestChannel', 'MaxStage']
    linked_columns = ['CLIENT_TYPE', 'EntranceChannel', 'MaxStage']

    # Checking cache
    cur_file = 'entrances_weeks_' + week1.split(' --- ')[0] + ' --- ' + week2.split(' --- ')[1] + '.json'
    # If the file exists load it
    if cur_file in os.listdir(temp_files + cur_folder):
        
        # Load json 
        with open(temp_files + cur_folder + cur_file) as infile:
            json_data  = json.load(infile)

        print(cur_file, 'was loaded')

    # If the file doesn't exist make request to db       
    else:
        req = sr.sankey_entrance_weeks
        req = req.replace("WEEK1", week1).replace("WEEK2", week2)
        res = execute_db(c.db_p, req, pr=True)
        df = pd.DataFrame(res[1:], columns=res[0])

        # df.dropna(inplace=True)
        df.fillna('NULL', inplace=True)

        df['EntranceChannel'] = df['EntranceChannel'].map(lambda x: 'EntranceChannel=' + str(x))
        df['MaxStage'] = df['MaxStage'].map(lambda x: 'MaxStage=' + str(x))

        print(df)

        if not df.empty:

            # print(df)

            node_id = 0
            nodes_dict = {}

            json_data = {}
            json_data["nodes"] = []
            json_data["links"] = []

            for column in linked_columns:
                for field in sorted(list(df[column].unique())):
                    json_data["nodes"].append({"node": node_id, "name": field}) 
                    nodes_dict[field] = node_id
                    node_id += 1

            for i in range(len(linked_columns) - 1):

                grouped_columns = linked_columns[i:i + 2]
                group = df.groupby(grouped_columns).sum()

                for index in list(group.index):
                    # print(index, group.loc[index, "item_counts"])
                    json_data["links"].append({"source": nodes_dict[index[0]], "target": nodes_dict[index[1]], "value": int(group.loc[index, 'item_counts'])})

            # Save json
            with open(temp_files + cur_folder + cur_file, 'w') as outfile:
                json.dump(json_data, outfile)

        else:
            json_data = []

        print(json_data)

    return render(request, 'entrance_sankey.html', {'json': json.dumps(json_data), 'week_options': week_options, 'week1': week1, 'week2': week2})


@login_required
def stage_sankey(request, type):

    save_event(request)

    d = {
        'digital': {
            'table': '##for_sankey',
            'max_stage': 8
        },

        'core': {
            'table': '##for_sankey_core',
            'max_stage': 3
        }, 

        'migone': {
            'table': '##for_sankey_migone',
            'max_stage': 8
        }, 

        'all': {
            'table': '##for_sankey_all',
            'max_stage': 9
        },   
    }

    linked_columns = ['s' + str(i) for i in range(1, d[type]['max_stage'] + 1)]

    req = '''
        select * from TABLE1
    '''

    req = req.replace('TABLE1', d[type]['table'])
    # req = sr.sankey_scorecards_rg_weeks
    # req = req.replace("WEEK1", week1).replace("WEEK2", week2)
    res = execute_db(c.db_p, req, pr=True)
    df = pd.DataFrame(res[1:], columns=res[0])
    df.replace({None: np.nan}, inplace=True)

    df['financed'] = 0
    for i in range(len(df)):
        for el in list(df.loc[i, :]):
            if 'профинансирован' in str(el).lower():
                df.loc[i, 'financed'] = 1

    # print(df)

    if not df.empty:

        # print(df)

        node_id = 0
        nodes_dict = {}

        json_data = {}
        json_data["nodes"] = []
        json_data["links"] = []

        for column in linked_columns:
            for i, field in enumerate(sorted(list(df[df[column].notnull()][column].unique()))):
                # print(column, sorted(list(df[df[column].notnull()][column].unique())))
                if field not in list(nodes_dict.keys()):
                    json_data["nodes"].append({"node": node_id, "name": field, 'stage': column, 'x1': 250*int(column.split('s')[1]),
                        'y1': 300*(i)}) 
                    nodes_dict[field] = node_id
                    node_id += 1

        for i in range(len(linked_columns) - 1):

            grouped_columns = linked_columns[i:i + 2]
            if grouped_columns[0] == grouped_columns[1]:
                continue
            group = df.groupby(grouped_columns).sum()

            # print(group)

            for index in list(group.index):
                # print(index, group.loc[index, "item_counts"])
                json_data["links"].append({"source": nodes_dict[index[0]], "target": nodes_dict[index[1]], "value": int(group.loc[index, 'num']),
                    'financed': int(group.loc[index, 'financed'])})

            # Save json
            cur_file = 'test.json'
            with open(temp_files + '/stage_sankey/' + cur_file, 'w', encoding='utf-8') as outfile:
                json.dump(json_data, outfile)

    else:
        json_data = {}

    # print(json_data)

    return render(request, 'stage_sankey.html', {'json': json.dumps(json_data)})



@login_required
def monthly_report(request):

    save_event(request)

    now = datetime.now()
    date_for_report = datetime.strftime(datetime.now(), '%Y%m') + '01'

    mr_base = "/monthly_report/"

    ###################################
    #          Инициализация          #
    ###################################

    json_cor = []
    json_volumes_num = []
    json_volumes_num_total = []
    json_buckets = []
 
    ###################################
    #             Портфель            #
    ###################################

    portfolio_error_db = False

    # group segment
    joined_products = {
        'Новый_Other_CL': 'Other',
        'Повторный_Other_CL': 'Other',
        'Новый_Core_PD': 'Новый_PD',
        'Новый_Digital_PD': 'Новый_PD',
        'Повторный_Core_PD': 'Повторный_PD',
        'Повторный_Digital_PD': 'Повторный_PD',
        }

    # Checking cache
    cur_file_debt_share = 'portfolio_debt_share' + '_' + date_for_report + '.json'
    cur_file_debt_sum = 'portfolio_debt_sum' + '_' + date_for_report + '.json'
    cur_file_labels = 'portfolio_labels' + '_' + date_for_report + '.pkl'
    debt_fields = ['principal_debt_amount', 'comission_debt_amount', 'debt_amount']
    all_labels = ['Повторный_Core_CL', 'Новый_Core_CL', 'Новый_Digital_CL', 'Повторный_Digital_CL', 'Новый_PD', 'Повторный_PD', 'Other']
    product_charact = ['la', 'lc', 'duration']

    # If the file exists load it
    if cur_file_debt_share in os.listdir(temp_files + mr_base + 'portfolio/') and cur_file_debt_sum in os.listdir(temp_files + mr_base + 'portfolio/') and cur_file_labels in os.listdir(temp_files + mr_base + 'portfolio/'):
        
        # Load jsons
        with open(temp_files + mr_base + 'portfolio/' + cur_file_debt_share) as infile:
            json_debt_share  = json.load(infile)

        with open(temp_files + mr_base + 'portfolio/' + cur_file_debt_sum) as infile:
            json_debt_sum  = json.load(infile)

        with open(temp_files + mr_base + 'portfolio/' + cur_file_labels, 'rb') as file:
            labels = pickle.load(file)   

        print(cur_file_debt_share, cur_file_debt_sum, cur_file_labels, 'were loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.portfolio
        req = req.replace("DATE1", date_for_report)
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            portfolio_error_db = True

        if not df.empty:
            df.columns = [x.lower() for x in df.columns]

            df['label'] = df['client_type'] + '_' + df['business_type'] + '_' + df['segment']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))

            # print(df)
            # print(df.dtypes)

            for col in debt_fields:
                try:
                    df[col] = df[col].astype(float)
                except: pass
            data_debt = df.groupby(['date', 'label'])[debt_fields].agg([np.sum])
            data_debt.columns = data_debt.columns.map('{0[0]}_{0[1]}'.format)
            data_debt = data_debt.reset_index()

            for col in debt_fields:
                data_debt[col] = data_debt.groupby('date')[col + '_sum'].apply(lambda x: x / float(x.sum()))

            debt_shares = data_debt.melt(id_vars=['date', 'label'], value_vars=debt_fields)
            data_debt_share = pd.crosstab([debt_shares['date'], debt_shares['label']], debt_shares['variable'], debt_shares['value'], aggfunc='sum').reset_index()


            data_debt_sum = df.groupby(['date'])[debt_fields].agg([np.sum])
            data_debt_sum.columns = data_debt_sum.columns.map('{0[0]}_{0[1]}'.format)
            data_debt_sum = data_debt_sum.reset_index().rename(columns={'date': 'date'})

            data_debt_share['label'].replace(joined_products, inplace=True)

            # print(data)

            json_debt_share = []
            json_debt_sum = []

            # labels = sorted(list(data_debt_share['label'].unique()))
            labels = all_labels.copy()

            with open(temp_files + mr_base + 'portfolio/' + cur_file_labels, 'wb') as file:
                pickle.dump(labels, file)
            
            for date in sorted(data_debt_share['date'].unique()):
                temp = data_debt_share[data_debt_share['date'] == date]
                for debt in debt_fields:
                    lst = {'date': date, 'variable': debt}
                    for el in zip(temp['label'], temp[debt]):
                        lst[el[0]] = el[1]
                        for label in labels:
                            if label not in lst.keys():
                                lst[label] = 0
                    json_debt_share.append(lst)

            # Save json
            with open(temp_files + mr_base + 'portfolio/' + cur_file_debt_share, 'w') as outfile:
                json.dump(json_debt_share, outfile)

            for i in range(len(data_debt_sum)):
                for col in debt_fields:
                    json_debt_sum.append({'date': data_debt_sum.loc[i, 'date'], 'variable': col, 'value': data_debt_sum.loc[i, col + '_sum']})

            # Save json
            with open(temp_files + mr_base + 'portfolio/' + cur_file_debt_sum, 'w') as outfile:
                json.dump(json_debt_sum, outfile)

        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'
            cur_file_debt_share = 'portfolio_debt_share' + '_' + date_base + '.json'
            cur_file_debt_sum = 'portfolio_debt_sum' + '_' + date_base + '.json'
            cur_file_labels = 'portfolio_labels' + '_' + date_base + '.pkl'
            try:
                with open(temp_files + mr_base + 'portfolio/' + cur_file_debt_share) as infile:
                    json_debt_share  = json.load(infile)

                with open(temp_files + mr_base + 'portfolio/' + cur_file_debt_sum) as infile:
                    json_debt_sum  = json.load(infile)

                with open(temp_files + mr_base + 'portfolio/' + cur_file_labels, 'rb') as file:
                    labels = pickle.load(file)

                print(cur_file_debt_share, cur_file_debt_sum, cur_file_labels, 'were loaded as default')
            except:
                json_debt_share = []
                json_debt_sum = []
                labels = []

    ###################################
    #         Бакеты просрочки        #
    ###################################

    reserves_buckets_error_db = False

    # Checking cache
    cur_file_buckets = 'buckets' + '_' + date_for_report + '.json'
    cur_file_cor = 'cor' + '_' + date_for_report + '.json'
    bucket_types = ['la', 'lc', 'la_lc']

    buckets_keys = [
                'bucket_0', 
                'bucket_1_3', 
                'bucket_4_14', 
                'bucket_15_31', 
                'bucket_32_62', 
                'bucket_63_93', 
                'bucket_94_124', 
                'bucket_125_200', 
                'bucket_201_360', 
                'bucket_360', 
            ]

    # 'buckets_keys': json.dumps(buckets_keys), 
    # cur_file_labels = 'portfolio_labels' + '_' + date_for_report + '.pkl'
    
    # If the file exists load it
    if cur_file_buckets in os.listdir(temp_files + mr_base + 'reserves_buckets/') and cur_file_cor in os.listdir(temp_files + mr_base + 'reserves_buckets/'):
        
        # Load jsons
        with open(temp_files + mr_base + 'reserves_buckets/' + cur_file_buckets) as infile:
            json_buckets  = json.load(infile)

        with open(temp_files + mr_base + 'reserves_buckets/' + cur_file_cor) as infile:
            json_cor  = json.load(infile)

        print(cur_file_buckets, cur_file_cor, 'were loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.reserves_buckets
        req = req.replace("DATE1", date_for_report)
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            reserves_buckets_error_db = True

        if not df.empty:
            df.columns = [x.lower() for x in df.columns]

            df['label'] = df['client_type'] + '_' + df['business_type'] + '_' + df['segment']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))

            p = re.compile(r'reserves_bucket_\d')

            buckets_columns = [el for el in df.columns if 'bucket' in el]
            for col in buckets_columns + ['delta_provisions', 'revenue']:
                df[col] = df[col].astype(float)
            buckets = sorted(list(set([p.findall(el)[0] for el in buckets_columns])))
            map_buckets = dict(zip(buckets, buckets_keys))

            # bucket_types = [re.findall(r'_\w{2,5}$', el) for el in buckets]
            d = dict(zip([el + '_sum' for el in buckets_columns], buckets_columns))

            data_buckets = df.groupby(['date'])[buckets_columns].agg([np.sum])
            data_buckets.columns = data_buckets.columns.map('{0[0]}_{0[1]}'.format)
            data_buckets = data_buckets.reset_index()
            data_buckets.rename(columns=d, inplace=True)
            data_buckets.fillna(0, inplace=True)

            json_buckets = []

            for i in range(len(data_buckets)):
                for bucket_type in bucket_types:
                    lst = {'date': data_buckets.loc[i, 'date'], 'variable': bucket_type}
                    for bucket in buckets:
                        lst[map_buckets[bucket]] = data_buckets.loc[i, bucket + '_' + bucket_type]
                    json_buckets.append(lst)

            data_cor = df.groupby('date')['delta_provisions', 'revenue'].agg(np.sum).reset_index()
            # print(data_cor)

            json_cor = []

            for i in range(len(data_cor)):
                try:
                    cor = data_cor.loc[i, 'delta_provisions']/data_cor.loc[i, 'revenue']
                    if cor in [-np.inf, np.inf]:
                        cor = 0
                except: cor = 0
                json_cor.append({'date': data_cor.loc[i, 'date'], 'value': cor})

            json_cor = json_cor[1:]
            # Save json
            with open(temp_files + mr_base + 'reserves_buckets/' + cur_file_buckets, 'w') as outfile:
                json.dump(json_buckets, outfile)

            # Save json
            with open(temp_files + mr_base + 'reserves_buckets/' + cur_file_cor, 'w') as outfile:
                json.dump(json_cor, outfile)

        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'

            cur_file_buckets = 'portfolio_debt_share' + '_' + date_base + '.json'
            cur_file_cor = 'portfolio_debt_sum' + '_' + date_base + '.json'

            try:
                with open(temp_files + mr_base + 'reserves_buckets/' + cur_file_debt_share) as infile:
                    json_debt_share  = json.load(infile)

                with open(temp_files + mr_base + 'reserves_buckets/' + cur_file_debt_sum) as infile:
                    json_debt_sum  = json.load(infile)

                print(cur_file_buckets, cur_file_cor, 'were loaded as default')
            
            except:
                json_debt_share = []
                json_debt_sum = []


    ###################################
    #       Рисковые индикаторы       #
    ###################################

    indicators_error_db = False

    # Checking cache
    cur_file_indicators = 'indicators' + '_' + date_for_report + '.json'
    indicators = ['fpd', '30_3mob', '30_6mob', '30_12mob']
    products_for_indicators = all_labels.copy()
    
    # If the file exists load it
    if cur_file_indicators in os.listdir(temp_files + mr_base + 'indicators/'):
        
        # Load jsons
        with open(temp_files + mr_base + 'indicators/' + cur_file_indicators) as infile:
            json_risk_indicators  = json.load(infile)

        print(cur_file_indicators, 'was loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.indicators
        req = req.replace("DATE1", date_for_report)
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            indicators_error_db = True

        if not df.empty:
            
            df.columns = [x.lower() for x in df.columns]
            
            df['label'] = df['client_type'] + '_' + df['business_type'] + '_' + df['segment']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))
            # join products
            df['label'].replace(joined_products, inplace=True)

            for col in indicators + ['num', 'la_lc']:
                df[col] = df[col].astype(float)
                # df = df[df[col].notnull()]

            df.fillna(0, inplace=True)
            df = df.groupby(['date', 'label']).agg(np.sum).reset_index()

            # print(list(df['label'].unique()))

            json_risk_indicators = []

            for i in range(len(df)):
                for indicator in indicators:
                    if indicator != 'fpd':
                        lst = {'date': df.loc[i, 'date'], 'product':  df.loc[i, 'label'], 'indicator': indicator, 'value': df.loc[i, indicator]/df.loc[i, 'la_lc']}
                    else:
                        lst = {'date': df.loc[i, 'date'], 'product':  df.loc[i, 'label'], 'indicator': indicator, 'value': df.loc[i, indicator]/df.loc[i, 'num']}
                    json_risk_indicators.append(lst)

            # Save json
            with open(temp_files + mr_base + 'indicators/' + cur_file_indicators, 'w') as outfile:
                json.dump(json_risk_indicators, outfile)

        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'

            cur_file_indicators = 'indicators' + '_' + date_base + '.json'

            try:
                with open(temp_files + mr_base + 'indicators/' + cur_file_indicators) as infile:
                    json_risk_indicators  = json.load(infile)

                print(cur_file_indicators, 'was loaded as default')
            
            except:
                json_risk_indicators = []

    ###############################################
    #       Средние показатели по продуктам       #
    ###############################################

    means_error_db = False

    # Checking cache
    cur_file_means = 'means' + '_' + date_for_report + '.json'
    means = product_charact.copy()
    products_for_means = all_labels.copy()
    
    # If the file exists load it
    if cur_file_means in os.listdir(temp_files + mr_base + 'means/'):

        # Load jsons
        with open(temp_files + mr_base + 'means/' + cur_file_means) as infile:
            json_means  = json.load(infile)

        print(cur_file_means, 'was loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.mean_values
        req = req.replace("DATE1", date_for_report)
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            means_error_db = True

        if not df.empty:
            
            df.columns = [x.lower() for x in df.columns]
            
            df['label'] = df['client_type'] + '_' + df['business_type'] + '_' + df['segment']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))
            # join products
            df['label'].replace(joined_products, inplace=True)

            for col in means + ['num']:
                df[col] = df[col].astype(float)

            df.fillna(0, inplace=True)
            df = df.groupby(['date', 'label']).agg(np.sum).reset_index()

            # print(list(df['label'].unique()))

            json_means = []

            for i in range(len(df)):
                for mean in means:
                    lst = {'date': df.loc[i, 'date'], 'variable': mean, 'product':  df.loc[i, 'label'], 'value': df.loc[i, mean]/df.loc[i, 'num']}
                    json_means.append(lst)

            # Save json
            with open(temp_files + mr_base + 'means/' + cur_file_means, 'w') as outfile:
                json.dump(json_means, outfile)

        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'

            cur_file_means = 'means' + '_' + date_base + '.json'

            try:
                with open(temp_files + mr_base + 'means/' + cur_file_means) as infile:
                    json_means  = json.load(infile)

                print(cur_file_means, 'was loaded as default')
            
            except:
                json_means = []

    ###################################
    #    Среднее LA, LC, Duration     #
    ###################################

    mean_values_error_db = False

    # Checking cache
    cur_file_mv = 'mean_values' + '_' + date_for_report + '.json'
    mean_values_keys = ['la', 'lc', 'duration']
    products_for_mean_values = all_labels.copy()
    
    # If the file exists load it
    if cur_file_mv in os.listdir(temp_files + mr_base + 'mean_values/'):
        
        # Load jsons
        with open(temp_files + mr_base + 'mean_values/' + cur_file_mv) as infile:
            json_mean_values  = json.load(infile)

        print(cur_file_mv, 'was loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.mean_values
        req = req.replace("DATE1", date_for_report)
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            mean_values_error_db = True

        if not df.empty:
            
            df.columns = [x.lower() for x in df.columns]
            
            df['label'] = df['client_type'] + '_' + df['business_type'] + '_' + df['segment']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))
            # join products
            df['label'].replace(joined_products, inplace=True)

            for col in mean_values_keys + ['num', 'la_lc']:
                df[col] = df[col].astype(float)
                # df = df[df[col].notnull()]

            df.fillna(0, inplace=True)
            df = df.groupby(['date', 'label']).agg(np.sum).reset_index()

            # print(list(df['label'].unique()))

            json_mean_values = []

            for i in range(len(df)):
                for label in mean_values_keys:
                     json_mean_values.append({'date': df.loc[i, 'date'], 'label': label, 'value': df.loc[i, label]/df.loc[i, 'num'], 'product': df.loc[i, 'label']})

            # Save json
            with open(temp_files + mr_base + 'mean_values/' + cur_file_mv, 'w') as outfile:
                json.dump(json_mean_values, outfile)

        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'

            cur_file_mv = 'mean_values' + '_' + date_base + '.json'

            try:
                with open(temp_files + mr_base + 'mean_values/' + cur_file_mv) as infile:
                    json_mean_values  = json.load(infile)

                print(cur_file_mv, 'was loaded as default')
            
            except:
                json_mean_values = []

    ###################################
    #       LA LC_REAL разница        #
    ###################################

    lc_lc_error_db = False

    # Checking cache
    cur_file_lc_lc_diff = 'lc_lc_diff' + '_' + date_for_report + '.json'
    lc_lc_diff_values_keys = ['lc', 'lc_real']
    products_for_lc_lc_diff = all_labels.copy()
    
    # If the file exists load it
    if cur_file_lc_lc_diff in os.listdir(temp_files + mr_base + 'lc_lc_diff/'):
        
        # Load jsons
        with open(temp_files + mr_base + 'lc_lc_diff/' + cur_file_lc_lc_diff) as infile:
            json_lc_lc_diff  = json.load(infile)

        print(cur_file_lc_lc_diff, 'was loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.mean_values
        req = req.replace("DATE1", date_for_report)
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            lc_lc_error_db = True

        if not df.empty:
            
            df.columns = [x.lower() for x in df.columns]
            
            df['label'] = df['client_type'] + '_' + df['business_type'] + '_' + df['segment']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))
            # join products
            df['label'].replace(joined_products, inplace=True)

            for col in lc_lc_diff_values_keys + ['la']:
                df[col] = df[col].astype(float)

            df.fillna(0, inplace=True)
            df = df.groupby(['date', 'label']).agg(np.sum).reset_index()

            json_lc_lc_diff = []

            for i in range(len(df)):
                for label in lc_lc_diff_values_keys:
                     json_lc_lc_diff.append({'date': df.loc[i, 'date'], 'label': label, 'value': df.loc[i, label]/df.loc[i, 'la'], 'product': df.loc[i, 'label']})

            # Save json
            with open(temp_files + mr_base + 'lc_lc_diff/' + cur_file_lc_lc_diff, 'w') as outfile:
                json.dump(json_lc_lc_diff, outfile)

        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'

            cur_file_lc_lc_diff = 'lc_lc_diff' + '_' + date_base + '.json'

            try:
                with open(temp_files + mr_base + 'lc_lc_diff/' + cur_file_lc_lc_diff) as infile:
                    json_lc_lc_diff  = json.load(infile)

                print(cur_file_lc_lc_diff, 'was loaded as default')
            
            except:
                json_lc_lc_diff = []


    ###################################
    #           Net Revenue           #
    ###################################

    net_revenue_error_db = False

    # Checking cache
    cur_file_nr = 'net_revenue' + '_' + date_for_report + '.json'
    
    # If the file exists load it
    if cur_file_nr in os.listdir(temp_files + mr_base + 'net_revenue/'):
        
        # Load jsons
        with open(temp_files + mr_base + 'net_revenue/' + cur_file_nr) as infile:
            json_net_revenue  = json.load(infile)

        print(cur_file_nr, 'was loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.net_revenue
        req = req.replace("DATE1", date_for_report)
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            net_revenue_error_db = True

        if not df.empty:
            
            df.columns = [x.lower() for x in df.columns]
            
            df['label'] = df['client_type'] + '_' + df['business_type'] + '_' + df['segment']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))
            # join products
            joined_products_2 = joined_products.copy()
            joined_products_2['Other_Other_Other'] = 'Other'
            df['label'].replace(joined_products_2, inplace=True)

            for col in ['la', 'lc', 'lc_real', 'loss']:
                df[col] = df[col].astype(float)

            df.fillna(0, inplace=True)
            df = df.groupby(['date', 'label']).agg(np.sum).reset_index()

            # print(list(df['label'].unique()))

            json_net_revenue = []

            for i in range(len(df)):
                segment = 'PD' if 'pd' in df.loc[i, 'label'] or 'PD' in df.loc[i, 'label'] else 'CL'
                json_net_revenue.append({'date': df.loc[i, 'date'], 'label': df.loc[i, 'label'], 'value': df.loc[i, 'lc'] - df.loc[i, 'loss'],
                    'segment': segment})
            
            # Save json
            with open(temp_files + mr_base + 'net_revenue/' + cur_file_nr, 'w') as outfile:
                json.dump(json_net_revenue, outfile)

        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'

            cur_file_nr = 'net_revenue' + '_' + date_base + '.json'

            try:
                with open(temp_files + mr_base + 'net_revenue/' + cur_file_nr) as infile:
                    json_net_revenue  = json.load(infile)

                print(cur_file_nr, 'was loaded as default')
            
            except:
                json_net_revenue = []

    ###################################
    #       Объемы выдач (руб.)       #
    ###################################

    volumes_error_db = False

    # Checking cache
    cur_file_volumes_share = 'volumes_share' + '_' + date_for_report + '.json'
    cur_file_volumes_total = 'volumes_total' + '_' + date_for_report + '.json'
    volumes_fields = ['la', 'lc', 'la+lc']
    
    # If the file exists load it
    if cur_file_volumes_share in os.listdir(temp_files + mr_base + 'volumes/') and cur_file_volumes_total in os.listdir(temp_files + mr_base + 'volumes/'):
        
        # Load jsons
        with open(temp_files + mr_base + 'volumes/' + cur_file_volumes_share) as infile:
            json_volumes_rub  = json.load(infile)

        with open(temp_files + mr_base + 'volumes/' + cur_file_volumes_total) as infile:
            json_volumes_rub_total  = json.load(infile)

        print(cur_file_volumes_share, cur_file_volumes_total, 'were loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.volumes_rub
        req = req.replace("DATE1", date_for_report)
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            volumes_error_db = True

        if not df.empty:
            df.columns = [x.lower() for x in df.columns]

            df['label'] = df['client_type'] + '_' + df['business_type'] + '_' + df['segment']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))

            for col in volumes_fields:
                try:
                    df[col] = df[col].astype(float)
                except: pass
            data_volumes = df.groupby(['date', 'label'])[volumes_fields].agg([np.sum])
            data_volumes.columns = data_volumes.columns.map('{0[0]}_{0[1]}'.format)
            data_volumes = data_volumes.reset_index()

            for col in volumes_fields:
                data_volumes[col] = data_volumes.groupby('date')[col + '_sum'].apply(lambda x: x / float(x.sum()))

            volume_shares = data_volumes.melt(id_vars=['date', 'label'], value_vars=volumes_fields)
            data_volumes_share = pd.crosstab([volume_shares['date'], volume_shares['label']], volume_shares['variable'], volume_shares['value'], aggfunc='sum').reset_index()


            data_volumes_sum = df.groupby(['date'])[volumes_fields].agg([np.sum])
            data_volumes_sum.columns = data_volumes_sum.columns.map('{0[0]}_{0[1]}'.format)
            data_volumes_sum = data_volumes_sum.reset_index().rename(columns={'date': 'date'})

            data_volumes_share['label'].replace(joined_products, inplace=True)

            # print(data)

            json_volumes_rub = []
            json_volumes_rub_total = []

            # labels = sorted(list(data_debt_share['label'].unique()))
            labels = all_labels.copy()

            for date in sorted(data_volumes_share['date'].unique()):
                temp = data_volumes_share[data_volumes_share['date'] == date]
                for vol in volumes_fields:
                    lst = {'date': date, 'variable': vol}
                    for el in zip(temp['label'], temp[vol]):
                        lst[el[0]] = el[1]
                        for label in labels:
                            if label not in lst.keys():
                                lst[label] = 0
                    json_volumes_rub.append(lst)

            # Save json
            with open(temp_files + mr_base + 'volumes/' + cur_file_volumes_share, 'w') as outfile:
                json.dump(json_volumes_rub, outfile)

            for i in range(len(data_volumes_sum)):
                for col in volumes_fields:
                    json_volumes_rub_total.append({'date': data_volumes_sum.loc[i, 'date'], 'variable': col, 'value': data_volumes_sum.loc[i, col + '_sum']})

            # Save json
            with open(temp_files + mr_base + 'volumes/' + cur_file_volumes_total, 'w') as outfile:
                json.dump(json_volumes_rub_total, outfile)

        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'
            cur_file_volumes_share = 'volumes_share' + '_' + date_base + '.json'
            cur_file_volumes_total = 'volumes_total' + '_' + date_base + '.json'
            try:
                with open(temp_files + mr_base + 'volumes/' + cur_file_volumes_share) as infile:
                    json_volumes_rub  = json.load(infile)

                with open(temp_files + mr_base + 'volumes/' + cur_file_volumes_total) as infile:
                    json_volumes_rub_total  = json.load(infile)

                print(cur_file_volumes_share, cur_file_volumes_total, 'were loaded as default')
            except:
                json_volumes_rub = []
                json_volumes_rub_total = []
                labels = []

    ###################################
    #        Объемы выдач (шт.)       #
    ###################################

    volumes_num_error_db = False

    # Checking cache
    cur_file_volumes_num_share = 'volumes_num_share' + '_' + date_for_report + '.json'
    cur_file_volumes_num_total = 'volumes_num_total' + '_' + date_for_report + '.json'
    volumes_num_fields = ['num']
    
    # If the file exists load it
    if cur_file_volumes_num_share in os.listdir(temp_files + mr_base + 'volumes/') and cur_file_volumes_num_total in os.listdir(temp_files + mr_base + 'volumes/'):
        
        # Load jsons
        with open(temp_files + mr_base + 'volumes/' + cur_file_volumes_num_share) as infile:
            json_volumes_num  = json.load(infile)

        with open(temp_files + mr_base + 'volumes/' + cur_file_volumes_num_total) as infile:
            json_volumes_num_total  = json.load(infile)

        print(cur_file_volumes_num_share, cur_file_volumes_num_total, 'were loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.volumes_num
        req = req.replace("DATE1", date_for_report)
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            volumes_num_error_db = True

        if not df.empty:
            df.columns = [x.lower() for x in df.columns]

            df['label'] = df['client_type'] + '_' + df['business_type'] + '_' + df['segment']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))

            # print(df)
            # print(df.dtypes)

            for col in volumes_num_fields:
                try:
                    df[col] = df[col].astype(float)
                except: pass
            data_volumes_num = df.groupby(['date', 'label'])[volumes_num_fields].agg([np.sum])
            data_volumes_num.columns = data_volumes_num.columns.map('{0[0]}_{0[1]}'.format)
            data_volumes_num = data_volumes_num.reset_index()

            for col in volumes_num_fields:
                data_volumes_num[col] = data_volumes_num.groupby('date')[col + '_sum'].apply(lambda x: x / float(x.sum()))

            volume_num_shares = data_volumes_num.melt(id_vars=['date', 'label'], value_vars=volumes_num_fields)
            data_volumes_num_share = pd.crosstab([volume_num_shares['date'], volume_num_shares['label']], volume_num_shares['variable'], volume_num_shares['value'], aggfunc='sum').reset_index()


            data_volumes_num_sum = df.groupby(['date'])[volumes_num_fields].agg([np.sum])
            data_volumes_num_sum.columns = data_volumes_num_sum.columns.map('{0[0]}_{0[1]}'.format)
            data_volumes_num_sum = data_volumes_num_sum.reset_index().rename(columns={'date': 'date'})

            data_volumes_num_share['label'].replace(joined_products, inplace=True)

            json_volumes_num = []
            json_volumes_num_total = []

            # labels = sorted(list(data_debt_share['label'].unique()))
            labels = all_labels.copy()

            for date in sorted(data_volumes_num_share['date'].unique()):
                temp = data_volumes_num_share[data_volumes_num_share['date'] == date]
                for vol in volumes_num_fields:
                    lst = {'date': date, 'variable': vol}
                    for el in zip(temp['label'], temp[vol]):
                        lst[el[0]] = el[1]
                        for label in labels:
                            if label not in lst.keys():
                                lst[label] = 0
                    json_volumes_num.append(lst)

            # Save json
            with open(temp_files + mr_base + 'volumes/' + cur_file_volumes_num_share, 'w') as outfile:
                json.dump(json_volumes_num, outfile)

            for i in range(len(data_volumes_num_sum)):
                for col in volumes_num_fields:
                    json_volumes_num_total.append({'date': data_volumes_num_sum.loc[i, 'date'], 'variable': col, 'value': data_volumes_num_sum.loc[i, col + '_sum']})

            # Save json
            with open(temp_files + mr_base + 'volumes/' + cur_file_volumes_num_total, 'w') as outfile:
                json.dump(json_volumes_num_total, outfile)

        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'
            cur_file_volumes_num_share = 'volumes_num_share' + '_' + date_base + '.json'
            cur_file_volumes_num_total = 'volumes_num_total' + '_' + date_base + '.json'
            try:
                with open(temp_files + mr_base + 'volumes/' + cur_file_volumes_num_share) as infile:
                    json_volumes_rub  = json.load(infile)

                with open(temp_files + mr_base + 'volumes/' + cur_file_volumes_num_total) as infile:
                    json_volumes_rub_total  = json.load(infile)

                print(cur_file_volumes_num_share, cur_file_volumes_num_total, 'were loaded as default')
            except:
                json_volumes_num = []
                json_volumes_num_total = []
                labels = []

    #############################
    #       Approval rate       #
    #############################

    approval_rate_error_db = False

    # Checking cache
    cur_file_approval_rate = 'approval_rate' + '_' + date_for_report + '.json'
    sum_count_approved = ['sum_approved', 'count_approved']
    products_for_means = all_labels.copy()
    
    # If the file exists load it
    if cur_file_approval_rate in os.listdir(temp_files + mr_base + 'approval_rate/'):
        # Load jsons
        with open(temp_files + mr_base + 'approval_rate/' + cur_file_approval_rate) as infile:
            json_approval_rate  = json.load(infile)

        print(cur_file_approval_rate, 'was loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.approval_rate
        req = req.replace("DATE1", date_for_report)
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            approval_rate_error_db = True

        if not df.empty:
            
            df.columns = [x.lower() for x in df.columns]
            
            df['label'] = df['client_type'] + '_' + df['business_type'] + '_' + df['segment']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))
            # join products
            df['label'].replace(joined_products, inplace=True)

            for col in sum_count_approved:
                df[col] = df[col].astype(float)

            df.fillna(0, inplace=True)
            df = df.groupby(['date', 'label']).agg(np.sum).reset_index()

            json_approval_rate = []

            for i in range(len(df)):
                lst = {'date': df.loc[i, 'date'], 'product':  df.loc[i, 'label'], 'value': df.loc[i, 'sum_approved']/df.loc[i, 'count_approved']}
                json_approval_rate.append(lst)

            # Save json
            with open(temp_files + mr_base + 'approval_rate/' + cur_file_approval_rate, 'w') as outfile:
                json.dump(json_approval_rate, outfile)

        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'

            cur_file_approval_rate = 'approval_rate' + '_' + date_base + '.json'

            try:
                with open(temp_files + mr_base + 'approval_rate/' + cur_file_approval_rate) as infile:
                    json_approval_rate  = json.load(infile)

                print(cur_file_approval_rate, 'was loaded as default')
            
            except:
                json_approval_rate = []


    #############################
    #         Net income        #
    #############################

    net_income_error_db = False

    # Checking cache
    cur_file_net_income = 'net_income' + '_' + date_for_report + '.json'
    cur_file_net_income2 = 'net_income2' + '_' + date_for_report + '.json'

    finance_indicator = ['delta_provisions', 'revenues', 'costs', 'net_income']
    products_for_income = all_labels.copy()
    products_for_income_all = all_labels.copy() + ['All_company']
    
    # If the file exists load it
    if cur_file_net_income in os.listdir(temp_files + mr_base + 'net_income/') and cur_file_net_income2 in os.listdir(temp_files + mr_base + 'net_income/'):
        # Load jsons
        with open(temp_files + mr_base + 'net_income/' + cur_file_net_income) as infile:
            json_net_income  = json.load(infile)
        with open(temp_files + mr_base + 'net_income/' + cur_file_net_income2) as infile:
            json_net_income2  = json.load(infile)
        print(cur_file_net_income, cur_file_net_income2, 'were loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.net_income
        req = req.replace("DATE1", date_for_report)
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            net_income_error_db = True

        if not df.empty:
            
            df.columns = [x.lower() for x in df.columns]
            
            df['label'] = df['client_type'] + '_' + df['business_type'] + '_' + df['segment']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))
            # join products
            df['label'].replace(joined_products, inplace=True)

            for col in finance_indicator:
                df[col] = df[col].astype(float)
                # df = df[df[col].notnull()]

            df.fillna(0, inplace=True)
            df = df.groupby(['date', 'label']).agg(np.sum).reset_index()
            df_all_copmpany = df.copy()
            df_all_copmpany = df_all_copmpany.groupby(['date']).agg(np.sum).reset_index()
            df_all_copmpany['label'] = 'All_company'

            json_net_income = []
            json_net_income2 = []
            
            for i in range(len(df)):
                secnd = {'date': df.loc[i, 'date'], 'delta_provisions': df.loc[i, 'delta_provisions']*(-1), 'revenues': df.loc[i, 'revenues'], 'costs': df.loc[i, 'costs'], 'product': df.loc[i, 'label']}
                json_net_income2.append(secnd)
                for label in finance_indicator:
                    lst = {'date': df.loc[i, 'date'], 'label': label, 'value': df.loc[i, label], 'product': df.loc[i, 'label']}
                    json_net_income.append(lst)
            for i in range(len(df_all_copmpany)):
                secnd2 = {'date': df_all_copmpany.loc[i, 'date'], 'delta_provisions': df_all_copmpany.loc[i, 'delta_provisions']*(-1), 'revenues': df_all_copmpany.loc[i, 'revenues'], 'costs': df_all_copmpany.loc[i, 'costs'], 'product': df_all_copmpany.loc[i, 'label']}
                json_net_income2.append(secnd2)
                for label in finance_indicator:
                    lst = {'date': df_all_copmpany.loc[i, 'date'], 'label': label, 'value': df_all_copmpany.loc[i, label], 'product': df_all_copmpany.loc[i, 'label']}
                    json_net_income.append(lst)


            # Save json
            with open(temp_files + mr_base + 'net_income/' + cur_file_net_income, 'w') as outfile:
                json.dump(json_net_income, outfile)
            with open(temp_files + mr_base + 'net_income/' + cur_file_net_income2, 'w') as outfile:
                json.dump(json_net_income2, outfile)
        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'

            cur_file_net_income = 'net_income' + '_' + date_base + '.json'
            cur_file_net_income2 = 'net_income2' + '_' + date_base + '.json'

            try:
                with open(temp_files + mr_base + 'net_income/' + cur_file_net_income) as infile:
                    json_net_income  = json.load(infile)
                with open(temp_files + mr_base + 'net_income/' + cur_file_net_income2) as infile:
                    json_net_income2  = json.load(infile)
                print(cur_file_net_income, 'was loaded as default')
            
            except:
                json_net_income = []
                json_net_income2 = []

    ###############################################
    #               Roll rate core                #
    ###############################################

    roll_rate_core_error_db = False

    # Checking cache
    cur_file_roll_rate_core = 'roll_rate_core' + '_' + date_for_report + '.json'
    roll_rate_core = ['flow_rate', 'flow_rate_la']

    roll_rate_keys = [
        "1 - 30",
        "31 - 60",
        "61 - 90",
        "91 - 120",
        "121 - 150",
        "151 - 180",
        "181+",
    ]
    
    # If the file exists load it
    if cur_file_roll_rate_core in os.listdir(temp_files + mr_base + 'roll_rate/'):
        # Load jsons
        with open(temp_files + mr_base + 'roll_rate/' + cur_file_roll_rate_core) as infile:
            json_roll_rate_core  = json.load(infile)

        print(cur_file_roll_rate_core, 'was loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.roll_rate
        req = req.replace("DATE1", date_for_report).replace("BUSINESS_TYPE1", 'Core')
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            roll_rate_core_error_db = True

        if not df.empty:
            
            df.columns = [x.lower() for x in df.columns]
            
            df['label'] = df['bucket']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))

            for col in roll_rate_core:
                df[col] = df[col].astype(float)

            df.fillna(0, inplace=True)
            df = df.groupby(['date', 'label']).agg(np.sum).reset_index()

            json_roll_rate_core= []

            for i in range(len(df)):
                for mean in roll_rate_core:
                    lst = {'date': df.loc[i, 'date'], 'variable': mean, 'product':  df.loc[i, 'label'], 'value': df.loc[i, mean]}
                    json_roll_rate_core.append(lst)

            # Save json
            with open(temp_files + mr_base + 'roll_rate/' + cur_file_roll_rate_core, 'w') as outfile:
                json.dump(json_roll_rate_core, outfile)

        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'

            cur_file_roll_rate_core = 'roll_rate_core' + '_' + date_base + '.json'

            try:
                with open(temp_files + mr_base + 'roll_rate/' + cur_file_roll_rate_core) as infile:
                    json_roll_rate_core  = json.load(infile)

                print(cur_file_roll_rate_core, 'was loaded as default')
            
            except:
                json_roll_rate_core = []

    ###############################################
    #               Roll rate digital             #
    ###############################################

    roll_rate_digital_error_db = False

    # Checking cache
    cur_file_roll_rate_digital = 'roll_rate_digital' + '_' + date_for_report + '.json'
    roll_rate_digital = ['flow_rate', 'flow_rate_la']
    
    # If the file exists load it
    if cur_file_roll_rate_digital in os.listdir(temp_files + mr_base + 'roll_rate/'):
        # Load jsons
        with open(temp_files + mr_base + 'roll_rate/' + cur_file_roll_rate_digital) as infile:
            json_roll_rate_digital  = json.load(infile)

        print(cur_file_roll_rate_digital, 'was loaded')

    # If the file doesn't exist make request to db       
    else:

        req = srmr.roll_rate
        req = req.replace("DATE1", date_for_report).replace("BUSINESS_TYPE1", 'Digital')
        try:
            data = execute_db(c.db_d, req, pr=True)
            df = pd.DataFrame(data[1:], columns=data[0])
        except:
            df = pd.DataFrame()
            roll_rate_digital_error_db = True

        if not df.empty:
            
            df.columns = [x.lower() for x in df.columns]
            
            df['label'] = df['bucket']
            df.rename(columns={'history_month': 'date'}, inplace=True)
            df = df[df['date'].notnull()]
            df.index = range(len(df))

            for col in roll_rate_digital:
                df[col] = df[col].astype(float)
                # df = df[df[col].notnull()]

            df.fillna(0, inplace=True)
            df = df.groupby(['date', 'label']).agg(np.sum).reset_index()

            json_roll_rate_digital= []

            for i in range(len(df)):
                for mean in roll_rate_digital:
                    lst = {'date': df.loc[i, 'date'], 'variable': mean, 'product':  df.loc[i, 'label'], 'value': df.loc[i, mean]}
                    json_roll_rate_digital.append(lst)

            # Save json
            with open(temp_files + mr_base + 'roll_rate/' + cur_file_roll_rate_digital, 'w') as outfile:
                json.dump(json_roll_rate_digital, outfile)

        else:
            # Load jsons
            # date_base = '20181101'
            date_base = str(now.year) +  '0'*(2 - len(str(now.month - 1))) + str(now.month - 1) + '01'

            cur_file_roll_rate_digital = 'roll_rate_digital' + '_' + date_base + '.json'

            try:
                with open(temp_files + mr_base + 'roll_rate/' + cur_file_roll_rate_digital) as infile:
                    json_roll_rate_digital  = json.load(infile)

                print(cur_file_roll_rate_digital, 'was loaded as default')
            
            except:
                json_roll_rate_digital = []


    ###################################
    #        Delete old files         #
    ###################################
    
    months_ago = 3
    date_old = str(now.year) +  '0'*(2 - len(str(now.month - months_ago))) + str(now.month - months_ago) + '01'

    # delete old jsons (keep only 3 last months)
    if not portfolio_error_db:
        
        old_file = temp_files + mr_base + 'portfolio/' + 'portfolio_debt_share_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

        old_file = temp_files + mr_base + 'portfolio/' + 'portfolio_debt_sum_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

        old_file = temp_files + mr_base + 'portfolio/' + 'portfolio_labels_%s.pkl' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

    if not reserves_buckets_error_db:

        old_file = temp_files + mr_base + 'reserves_buckets/' + 'buckets_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

        old_file = temp_files + mr_base + 'reserves_buckets/' + 'cor_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

    if not indicators_error_db:

        old_file = temp_files + mr_base + 'indicators/' + 'indicators_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

    if not means_error_db:

        old_file = temp_files + mr_base + 'means/' + 'means_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

    if not mean_values_error_db:

        old_file = temp_files + mr_base + 'mean_values/' + 'mean_values_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

    if not lc_lc_error_db:

        old_file = temp_files + mr_base + 'lc_lc_diff/' + 'lc_lc_diff_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass 

    if not net_revenue_error_db:

        old_file = temp_files + mr_base + 'net_revenue/' + 'net_revenue_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

    if not volumes_error_db:
        
        old_file = temp_files + mr_base + 'volumes/' + 'volumes_share_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

        old_file = temp_files + mr_base + 'volumes/' + 'volumes_total_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

    if not volumes_num_error_db:
        
        old_file = temp_files + mr_base + 'volumes/' + 'volumes_num_share_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

        old_file = temp_files + mr_base + 'volumes/' + 'volumes_num_total_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

    if not approval_rate_error_db:

        old_file = temp_files + mr_base + 'approval_rate/' + 'approval_rate_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

    if not net_income_error_db:

        old_file = temp_files + mr_base + 'net_income/' + 'net_income_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

        old_file = temp_files + mr_base + 'net_income/' + 'net_income2_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

    if not roll_rate_core_error_db:

        old_file = temp_files + mr_base + 'roll_rate/' + 'roll_rate_core_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

    if not roll_rate_digital_error_db:

        old_file = temp_files + mr_base + 'roll_rate/' + 'roll_rate_digital_%s.json' % date_old
        try:
            os.remove(old_file)  # delete
        except: pass

    ###################################
    #             FILTER              #
    ###################################

    # json_risk_indicators = [x for x in json_risk_indicators if x['value'] > 0]
    # json_net_revenue = [x for x in json_net_revenue if x['date'] < '2018-09']


    ###################################
    #              КАНЕЦ              #
    ###################################

    return render(request, 'monthly_report.html', {

        'debt_share': json.dumps(json_debt_share), 
        'debt_keys': json.dumps(labels), 
        'debt_sum': json.dumps(json_debt_sum),
        'debt_fields': debt_fields, 

        'buckets_keys': json.dumps(buckets_keys), 
        'bucket_types': bucket_types, 
        'json_buckets': json.dumps(json_buckets), 
        'json_cor': json.dumps(json_cor), 

        'risk_indicators': json.dumps(json_risk_indicators), 
        'products_for_indicators': products_for_indicators,

        'json_means': json.dumps(json_means),
        'json_mean_values': json.dumps(json_mean_values),
        'products_for_mean_values': products_for_mean_values,

        'json_lc_lc_diff': json.dumps(json_lc_lc_diff),

        'json_net_revenue': json.dumps(json_net_revenue),

        'json_volumes_rub': json.dumps(json_volumes_rub),
        'json_volumes_rub_total': json.dumps(json_volumes_rub_total),

        'json_volumes_num': json.dumps(json_volumes_num),
        'json_volumes_num_total': json.dumps(json_volumes_num_total),

        'product_charact': product_charact,
        'volume_fields': volumes_fields,

        'json_approval_rate':  json.dumps(json_approval_rate),

        'json_net_income': json.dumps(json_net_income), 
        'products_for_income': products_for_income,

        'json_net_income2': json.dumps(json_net_income2),
        'products_for_income_all': products_for_income_all, 

        'roll_rate_keys': json.dumps(roll_rate_keys),
        'roll_rate_variables': roll_rate_core,
        'json_roll_rate_core': json.dumps(json_roll_rate_core),
        'json_roll_rate_digital': json.dumps(json_roll_rate_digital),

    })


@login_required
def monthly_report_delete(request):

    now = datetime.now()

    mr_base = "/monthly_report/"
    
    ###################################
    #        Delete old files         #
    ###################################
    
    months_ago = 0
    date_old = str(now.year) +  '0'*(2 - len(str(now.month - months_ago))) + str(now.month - months_ago) + '01'

    # delete old jsons
        
    old_file = temp_files + mr_base + 'portfolio/' + 'portfolio_debt_share_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass

    old_file = temp_files + mr_base + 'portfolio/' + 'portfolio_debt_sum_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass

    old_file = temp_files + mr_base + 'portfolio/' + 'portfolio_labels_%s.pkl' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass


    old_file = temp_files + mr_base + 'reserves_buckets/' + 'buckets_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass

    old_file = temp_files + mr_base + 'reserves_buckets/' + 'cor_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass


    old_file = temp_files + mr_base + 'indicators/' + 'indicators_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass


    old_file = temp_files + mr_base + 'means/' + 'means_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass


    old_file = temp_files + mr_base + 'mean_values/' + 'mean_values_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass


    old_file = temp_files + mr_base + 'lc_lc_diff/' + 'lc_lc_diff_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass 


    old_file = temp_files + mr_base + 'net_revenue/' + 'net_revenue_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass


    old_file = temp_files + mr_base + 'volumes/' + 'volumes_share_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass

    old_file = temp_files + mr_base + 'volumes/' + 'volumes_total_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass

        
    old_file = temp_files + mr_base + 'volumes/' + 'volumes_num_share_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass

    old_file = temp_files + mr_base + 'volumes/' + 'volumes_num_total_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass


    old_file = temp_files + mr_base + 'approval_rate/' + 'approval_rate_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass


    old_file = temp_files + mr_base + 'net_income/' + 'net_income_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass

    old_file = temp_files + mr_base + 'net_income/' + 'net_income2_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass


    old_file = temp_files + mr_base + 'roll_rate/' + 'roll_rate_core_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass


    old_file = temp_files + mr_base + 'roll_rate/' + 'roll_rate_digital_%s.json' % date_old
    try:
        os.remove(old_file)  # delete
        print(old_file)
    except: pass

    return redirect('/risks/monthly_report/')


@login_required
def sales_plan(request):

    tables_names = ['origination', 'amount', 'comission']
    tables = {}

    for name in tables_names:

        try:

            df = pd.read_excel(temp_files + '/sales_plan/sales_plan_%s.xlsx' % name)      
            df.fillna('', inplace=True)

            df_columns = list(df.columns)
            n = 2
            df_columns[n:n + 12] = [datetime.strftime(el, '%Y-%m') for el in df_columns[n:n + 12]]
            df.columns = df_columns

            df.index = df['business_type']

        except:

            df = pd.DataFrame([['Empty']], columns=['data'])

        # df.sort_index(inplace=True)

        tables[name] = df.to_html(index=False)
    tables = OrderedDict(tables)

    return render(request, 'sales_plan.html', {
        'tables': tables,

    })


# admin, users
@login_required
def monitoring(request):

    print('temp_files', temp_files)

    last_actions = Activity.objects.all().order_by('-id')[:10]
    last_unique_users_actions = []

    req = '''
    select 1 as id, url, count(url) as count from activity_activity
    group by url
    order by count(url) desc
    '''

    page_counts = Activity.objects.raw(req)

    
    # req = '''
    #     RANK() OVER(PARTITION BY user_id ORDER BY score DESC)
    # '''
    # print(Activity.objects.filter().annotate(rank=RawSQL(req, [])   ))

    sql = """
        SELECT user, url, event_date, RANK() OVER(ORDER BY id DESC) AS rank
        --SELECT * 
        FROM activity_activity
        --order by event_date desc
    """

   

    # res = Activity.objects.raw(sql)

    # print(res.columns)
    # print(res[0])

    # for row in res:
    #     print(row.url, row.count)


    # sql = '''
    #     SELECT * FROM activity_activity ORDER BY event_date DESC
    # '''

    # res = Activity.objects.raw(sql)
    # for row in res:
    #     # print(row.user, row.url, row.event_date)
    #     print(row)


    # print(Activity.objects.filter().extra(select={'rank': 'RANK() OVER(ORDER BY event_date DESC)' }))

    # from django.db.models import Func

    # class Rank(Func):
    #     function = 'RANK'
    #     template = '%(function)s() OVER (ORDER BY %(expressions)s DESC)'

    # print(Activity.objects.annotate(rank=Rank('event_date')))


    return render(request, 'monitoring.html', {'last_actions': last_actions, 'page_counts': page_counts})

def login_view(request):
    if request.user.is_authenticated:
        return redirect('/')
    else:
        try:
            username = request.POST['username']
            password = request.POST['password']
            # print(username, password)
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                save_event(request, user=user)
                # Redirect to a success page.
                return redirect('/')
            else:

                # Return an 'invalid login' error message.
                return render(request, 'index.html', {'error': True})
        except Exception as e:

            print(e)

            return redirect('/')


def logout_view(request):
    
    save_event(request)
    logout(request)
    try:
        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
    except:
        return redirect('/')
