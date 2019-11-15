from matplotlib import pyplot as plt
import boto3
import pandas as pd

def plot_track_and_waypoints(waypoints, draw_wp=True, figsize=(16/1.5, 10/1.5)):
    if figsize is not None:
        plt.figure(figsize=figsize)
    if draw_wp:
        plt.scatter(waypoints[:, 0], waypoints[:, 1], c='y', s=10, marker='D')
        for i in range(len(waypoints)):
            plt.text(waypoints[i, 0], waypoints[i, 1], str(i))
    plt.plot(waypoints[:, 2], waypoints[:, 3], c='k')
    plt.plot(waypoints[:, 4], waypoints[:, 5], c='k')
    
    
def get_log(logStreamName, logGroupName='/aws/robomaker/SimulationJobs'):
    client = boto3.client('logs')
    kwargs = {
            'logGroupName': logGroupName,
            'logStreamNames': [logStreamName], 
            'limit': 10000,
        }
    while True:
        print('#', end='')
        resp = client.filter_log_events(**kwargs)
        yield from resp['events']
        try:
            kwargs['nextToken'] = resp['nextToken']
        except KeyError:
            print()
            break
    return resp

def json_to_df(resp, EPISODE_PER_ITER=20):
    df = pd.DataFrame(resp)[['message', 'timestamp']]
    trace = df[df['message'].apply(lambda x: 'SIM_TRACE_LOG:' in x)]
    trace_df = pd.DataFrame([x for x in trace['message'].apply(lambda x: x.split('SIM_TRACE_LOG:')[1].split(','))], 
                        columns=['episode', 'steps', 'x', 'y', 'yaw', 'steer', 'throttle', 'action', 
                                 'reward', 'done', 'on_track', 'progress',  'closest_waypoint', 'track_len', 
                                 'timestamp'])
    trace_df = trace_df.astype({'episode': 'int32', 'steps': 'int32', 'x': 'float', 'y': 'float', 'progress': 'float', 'steer': 'float',
                            'timestamp': 'double', 'throttle': 'float', 'reward': 'float', 'closest_waypoint': 'int32', 'yaw': 'float', 
                               'track_len': 'float'})
    trace_df['x'] = trace_df['x']*100
    trace_df['y'] = trace_df['y']*100
    trace_df['iteration'] = trace_df['episode'] // EPISODE_PER_ITER + 1
    return trace_df

def get_amazon_log(logStreamName, logGroupName='/aws/robomaker/SimulationJobs', convert_to_df=True, EPISODE_PER_ITER=20):
    resp = [x for x in get_log(logStreamName, logGroupName)]
    print('file downloaded')
    if convert_to_df:
        df = json_to_df(resp, EPISODE_PER_ITER=EPISODE_PER_ITER)
        return df
    else:
        return resp