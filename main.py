from mangum import Mangum
from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import tempfile
import joblib
import json
import pandas as pd
import sqlalchemy as sa
import itertools
from datetime import timedelta

class InputData(BaseModel):
    data: list

app = FastAPI()
handler = Mangum(app)


value_map = json.load(open('target_encoder_map.json', 'r'))
city_location_map = json.load(open('city_location_map.json', 'r'))


def transform_data(df):
    X_COLS = ['industry', 'metric', 'location_id', 'city_id', 'is_sheltered', 'location_type_id',
              'mission_day_of_year', 'mission_start_hour']
    CATERGORY_COLS = ['industry', 'metric', 'location_id', 'city_id', 'is_sheltered', 'location_type_id']
    df = df[X_COLS]
    df = df.dropna()
    df['mission_day_of_year'] = df['mission_day_of_year'].astype(int)
    df['mission_start_hour'] = df['mission_start_hour'].astype(int)
    str_cols = ['industry', 'metric', 'is_sheltered']
    for c in str_cols:
        df[c] = df[c].astype(str).str.lower().str.replace('industry: ', '').str.replace(' ', '').str.replace(
            'signups', 'signup')

    df.loc[~df['location_id'].astype(str).isin(value_map['location_id'].keys()), 'location_id'] = 'other'
    for col in CATERGORY_COLS:
        df[col] = df[col].astype(str)
        val_map = value_map[col]
        df[col] = df[col].map(val_map).values
    df = df[X_COLS]
    return df


def get_s3_client():
    s3 = boto3.client('s3')
    return s3


def load_model_from_s3(bucket, key):
    s3_client = get_s3_client()
    with tempfile.TemporaryFile() as fp:
        s3_client.download_fileobj(Fileobj=fp, Bucket=bucket, Key=key)
        fp.seek(0)
        return joblib.load(fp)


def query_to_pandas(query):
    HOST = 'prod-oppizi-com.cmqi0rqok9gw.us-east-1.rds.amazonaws.com'
    PORT = 5432
    USER = 'ganymede'
    DB = 'oppizi'
    PW = 'UP5Z4YoLDGSvoVnotmcd'
    query = sa.text(query)
    conn_str = f"postgresql+psycopg2://{USER}:{PW}@{HOST}:{PORT}/{DB}"
    engine = sa.create_engine(conn_str)
    with engine.begin() as conn:
        df = pd.read_sql(sql=query, con=conn)
    engine.dispose()
    return df


def get_locations(cities):
    # query = f"""
    # SELECT ID as location_id,  city_id, is_sheltered, type_id FROM oppizi.public."Location" WHERE  city_id IN ({','.join(cities)})
    # """
    # df = query_to_pandas(query)
    # out_list = list(df.values)
    out_list = []
    for city in cities:
        out_list += city_location_map[city]
    out_list = [(x['location_id'], x['city_id'], x['is_sheltered'], x['location_type_id']) for x in out_list]
    return out_list



def build_campaign(industry, start_date, end_date, missions_per_week, cities, metrics):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    start_day = start_date.dayofyear
    end_day = end_date.dayofyear
    days = list(range(start_day, end_day))
    hours = list(range(7,20))
    metrics = json.loads(metrics)
    cities = [str(int(c)) for c in json.loads(cities)]
    mission_combo_df = pd.DataFrame(itertools.product(*[[industry], metrics, get_locations(cities), days, hours]),
                                    columns=['industry', 'metric','loc_tupe','mission_day_of_year','mission_start_hour'])
    mission_combo_df[['location_id','city_id','is_sheltered','location_type_id']] = pd.DataFrame(mission_combo_df.loc_tupe.tolist(), index=mission_combo_df.index)
    mission_combo_df = mission_combo_df.drop(columns=['loc_tupe'])
    model_input_df = transform_data(mission_combo_df)
    model = joblib.load('xgb_target_encoder.tar.gz')
    mission_combo_df['est_conversion_rate'] = model.predict(model_input_df)
    mission_combo_df = mission_combo_df.groupby([c for c in mission_combo_df.columns if c not in ['est_conversion_rate', 'metric']])['est_conversion_rate'].mean().reset_index()
    mission_combo_df = mission_combo_df.sort_values('est_conversion_rate').reset_index(drop=True)

    # mission_combo_df = mission_combo_df.drop_duplicates(subset=['location_id','mission_day_of_year'], keep='last')
    out_campaign_df = pd.DataFrame()
    for week_num in range(int(len(days)/7)):
        week_start = start_date + timedelta(week_num*7)
        week_end = start_date + timedelta((week_num+1)*7)
        week_mission_df = mission_combo_df.copy()[(mission_combo_df['mission_day_of_year']>week_start.dayofyear) &
                                                  (mission_combo_df['mission_day_of_year']<week_end.dayofyear)]
        week_mission_df = week_mission_df.drop_duplicates(['location_id','mission_day_of_year'], keep='last')
        week_mission_df = week_mission_df.sort_values('est_conversion_rate').reset_index(drop=True)
        week_mission_df = week_mission_df.groupby('mission_day_of_year').apply(lambda x: x.tail(int(missions_per_week)))
        week_mission_df = week_mission_df.sort_values('est_conversion_rate').reset_index(drop=True)
        # if len(out_campaign_df)>0:
        #     week_mission_df = week_mission_df[~week_mission_df['location_id'].isin(out_campaign_df.tail(missions_per_week*2)['location_id'])]
        week_mission_df = week_mission_df.tail(missions_per_week)
        week_mission_df['mission_date'] = week_mission_df['mission_day_of_year'].apply(lambda x: (week_start + timedelta(x-week_start.dayofyear)).date())
        week_mission_df['week_number'] = week_num+1
        out_campaign_df = out_campaign_df.append(week_mission_df)
    out_campaign_df = out_campaign_df.sort_values(['mission_date', 'mission_start_hour']).reset_index(drop=True)
    out_campaign_df = out_campaign_df.reset_index().rename(columns={'index':'mission_number'})
    out_campaign_df['merge_col'] = out_campaign_df['mission_number']
    leading_cols = ['mission_number', 'est_conversion_rate', 'mission_date']
    out_campaign_df = out_campaign_df[leading_cols+[c for c in out_campaign_df.columns if c not in leading_cols]]
    return out_campaign_df


@app.post("/mission_convergence_model")
def run_mission_model(
        input_data: InputData
):
    # bucket = "ai-campaign-generation"
    # model_prefix = "models/xgb_target_encoder.tar.gz"
    # model = load_model_from_s3(bucket, model_prefix)
    model = joblib.load('xgb_target_encoder.tar.gz')
    input_df = pd.DataFrame(input_data.data)
    input_df = transform_data(input_df)
    predictions = [float(x) for x in model.predict(input_df)]
    print(predictions)
    return {"predictions":predictions, "input_data": input_data}


@app.post("/campaign_generation_model")
def run_campaign_model(
        input_data: InputData
):
    output = []
    for campaign_input in input_data.data:
        campaign_df = build_campaign(**campaign_input)
        output.append({"campaign":campaign_df.to_dict(orient='records'), "campaign_inputs": campaign_input})
    return output


@app.get("/")
def hello_world():
    return {'message': 'BingBong'}


@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f'Hello from FastAPI, {name}!'}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)