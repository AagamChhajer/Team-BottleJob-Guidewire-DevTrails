import pandas as pd
import numpy as np
import redis
import redis.client 

def setup_redis():
    redis_client = redis.StrictRedis(host = 'localhost' , port = 6379 , db = 0)
    return redis_client

def store_model_redis(redis_client , model ,model_key):
    import joblib
    joblib_bytes = joblib.dump(r"C:\Users\Shreyansh Singh\Desktop\ML3\app\Machine_Failure_classification.pkl")
    redis_client.set(model_key , joblib_bytes)
    
def load_model_redis(redis_client , model_key):
 import joblib
 model_bytes = redis.get(model_key)
 model = joblib.load(joblib)