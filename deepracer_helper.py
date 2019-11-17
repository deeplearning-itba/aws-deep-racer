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
    
    
import cv2
import numpy as np
import tensorflow as tf

def visualize_gradcam_discrete_ppo(sess, rgb_img, category_index=0, num_of_actions=6):
    '''
    @inp: model session, RGB Image - np array, action_index, total number of actions 
    @return: overlayed heatmap
    '''
    
    img_arr = np.array(rgb_img)
    img_arr = rgb2gray(img_arr)
    img_arr = np.expand_dims(img_arr, axis=2)
    
    x = sess.graph.get_tensor_by_name('main_level/agent/main/online/network_0/observation/observation:0')
    y = sess.graph.get_tensor_by_name('main_level/agent/main/online/network_1/ppo_head_0/policy:0')
    feed_dict = {x:[img_arr]}

    #Get he policy head for clipped ppo in coach
    model_out_layer = sess.graph.get_tensor_by_name('main_level/agent/main/online/network_1/ppo_head_0/policy:0')
    # Se queda solo con una acción
    loss = tf.multiply(model_out_layer, tf.one_hot([category_index], num_of_actions))
    reduced_loss = tf.reduce_sum(loss[0])
    # Ultima capa convolucional
    conv_output = sess.graph.get_tensor_by_name('main_level/agent/main/online/network_1/observation/Conv2d_4/Conv2D:0')
    grads = tf.gradients(reduced_loss, conv_output)[0]
    output, grads_val = sess.run([conv_output, grads], feed_dict=feed_dict)
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.sum(weights * output, axis=3)

    ##im_h, im_w = 120, 160##
    im_h, im_w = rgb_img.shape[:2]

    cam = cams[0] #img 0
    image = np.uint8(rgb_img[:, :, ::-1] * 255.0) # RGB -> BGR
    cam = cv2.resize(cam, (im_w, im_h)) # zoom heatmap
    cam = np.maximum(cam, 0) # relu clip
    heatmap = cam / np.max(cam) # normalize
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) # grayscale to color
    cam = np.float32(cam) + np.float32(image) # overlay heatmap
    cam = 255 * cam / (np.max(cam) + 1E-5) ##  Add expsilon for stability
    cam = np.uint8(cam)[:, :, ::-1] # to RGB

    return cam


def load_session(pb_path):
    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                                    log_device_placement=True))
    print("load graph:", pb_path)
    with gfile.FastGFile(pb_path,'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]
    names = []
    for t in graph_nodes:
        names.append(t.name)
    x = sess.graph.get_tensor_by_name('main_level/agent/main/online/network_0/observation/observation:0')
    y = sess.graph.get_tensor_by_name('main_level/agent/main/online/network_1/ppo_head_0/policy:0')
    
    return sess, x, y

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])