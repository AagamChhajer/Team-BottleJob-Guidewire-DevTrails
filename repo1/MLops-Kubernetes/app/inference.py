import pandas as pd
import joblib
import redis
import hashlib
import json

model = joblib.load('Machine_Failure_classification.pkl')

redis_client = redis.StrictRedis(host='localhost' , port = 6379 , db = 0 , decode_responses=True)

def hash_input(input_data):
    """Generate a hash for the input and use it as a redis key"""
    input_str = json.dumps(input_data , sort_keys = True)
    return hashlib.sha256(input_str.encode()).hexdigest()

def get_prediction(input_data):
    """Get prediction from redis or calculate if not cached"""
    
    key = hash_input(input_data)
    cached_result = redis_client.get(key)
    
    if cached_result:
        print("cache hit!")
        return json.loads(cached_result)
    else:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        
        result = 'Failure' if prediction[0] == 1 else 'No Failure'
        
        redis_client.set(key , json.dumps(result))
        print("cache miss")
        return result
    
def get_user_input():
    print("Give the inputs for the prediction of the Machine Failure ")
    footfall = int(input("FootFall: "))
    tempMode = int(input("TempMode: "))  # Change TempNode to tempMode
    AQ = int(input("Air Quality Index: "))
    USS = int(input("USS: "))
    CS = int(input("CS: "))
    VOC = int(input("VOC: "))
    RP = int(input("RP: "))
    IP = int(input("IP: "))
    temperature = int(input("Temperature: "))  # Change Temp to Temperature

    input_data = {
        'footfall': footfall,
        'tempMode': tempMode,  # Use tempMode as seen in training
        'AQ': AQ,
        'USS': USS,
        'CS': CS,
        'VOC': VOC,
        'RP': RP,
        'IP': IP,
        'Temperature': temperature  # Use Temperature as seen in training
    }

    return input_data


if __name__ == '__main__':
    print("Running inference script...")

    # Get user input
    input_data = get_user_input()

    # Get the prediction
    result = get_prediction(input_data)
    print("Prediction:", result)