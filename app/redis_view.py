import redis
import pandas as pd

def view_redis_table():
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

    keys = redis_client.keys('*')

    data = []

    for key in keys:
        value = redis_client.get(key)  
        data.append([key, value])

    df = pd.DataFrame(data, columns=['Key', 'Value'])

    print(df)

if __name__ == "__main__":
    view_redis_table()
