from mangum import Mangum
from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import tempfile
import joblib
import json
import pandas as pd


class InputData(BaseModel):
    data: list

app = FastAPI()
handler = Mangum(app)


value_map = json.load(open('target_encoder_map.json', 'r'))

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


@app.post("/mission_convergence_model")
def read_root(
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


@app.get("/")
def hello_world():
    return {'message': 'Hello from FastAPI'}


@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f'Hello from FastAPI, {name}!'}



# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app)