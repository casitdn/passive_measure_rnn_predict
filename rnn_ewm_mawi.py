'''
Created on 2019-6-5-
passive network measurement experiments
@author: xie w.
'''

import tensorflow as tf
import pandas as pd
#import pandas.Series as Series

from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np

#filepath = "./dataset_edit/savelink2005_o.csv"
filepath = "./dataset/mawi_sampleB_2004_2007.csv"



def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def split_to_df(string): 
    return pd.Series(literal_eval(string)) 





def splitDataToColumn_01(df):
    df[ 'total_bytes' ] =  df['total_bytes'] / 1000000.0 ## unit => M
    df[ 'tcp_udp_icmp_rate' ] = df['tcp_udp_icmp_rate'].str.replace('%','') #work
    df[ ['tcp','udp','icmp'] ] = df.tcp_udp_icmp_rate.str.split('-', expand=True)
#     df_data:             date  total_bytes    tcp_udp_icmp_rate  ...       tcp     upd   icmp
#     0    2005-01-01   1562599468   90.75%-7.61%-0.57%  ...    90.75%   7.61%  0.57%
    
    df = df.drop(['tcp_udp_icmp_rate'], 1)
    df[ 'ipaddr_top' ] = df['ipaddr_top'].str.replace('%','') #work
    df[ ['top1','top2','top3','top4','top5'] ] = df.ipaddr_top.str.split('-', expand=True)
    df = df.drop(['ipaddr_top'], 1)
    
    return df
    
    
def moving_average(df, n, column):
    """Calculate the moving average for the given data.
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    ## Series is sub dataFrame data, add colume
    MA = pd.Series(df[column].rolling(n, min_periods=n).mean(), name='MA_'+ column + '_' + str(n))
    df = df.join(MA)
    return df

def exponential_moving_average(df, n,column):
    """
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    EMA = pd.Series(df[column].ewm(span=n, min_periods=n).mean(), name='EMA_' + column + '_' + str(n))
    df = df.join(EMA)
    return df


def network_basic_multiLyaer(
        data_n, features, batch_size, dataset_train_length,
        num_units, learning_rate, epochs, pred_col_name ):
    number_of_features = len(features)

    #multi-cell
    num_layers = 3
    data_length = len(data_n.index) - (len(data_n.index) % batch_size)

    ## data_n['MA_top1_12'] if step1 have None predict y value, will cause future y is Nones
    dataset_train_x = data_n[features].as_matrix()[:dataset_train_length]
    dataset_train_y = data_n[pred_col_name].as_matrix()[:dataset_train_length]

    dataset_test_x = data_n[features].as_matrix()[dataset_train_length:]
    dataset_test_y = data_n[pred_col_name].as_matrix()[dataset_train_length:]
    
    # 2) *** Build the network ***
    ## batch_size is splited minibatch_size 
    plh_batch_x = tf.placeholder(
        dtype=tf.float32, name='plc_batch_x',
        shape=[None, batch_size, number_of_features],
    )   

    plh_batch_y = tf.placeholder(
        dtype=tf.float32, shape=[None, batch_size, 1], name='plc_batch_x'
    )
    
    labels_series = tf.unstack(plh_batch_y, axis=1)

    cell = [tf.nn.rnn_cell.BasicRNNCell(num_units=num_units) for _ in range(num_layers)]
    #cell = tf.nn.rnn_cell.MultiRNNCell([cells] * num_layers) #
    cell = tf.nn.rnn_cell.MultiRNNCell(cell)
     
    states_series, current_state = tf.nn.dynamic_rnn(
        cell=cell, inputs=plh_batch_x, dtype=tf.float32)
    
    ## [1, 0, 2] is original dimension index 
    states_series = tf.transpose(states_series, [1, 0, 2])    
    # last_state is  (batch_size,num_units)
    last_state = tf.gather(params=states_series, indices=states_series.get_shape()[0] - 1)    
    last_label = tf.gather(
        params=labels_series, indices=len(labels_series) - 1)
    weight = tf.Variable(tf.truncated_normal([num_units, 1]))
    bias = tf.Variable(tf.constant(0.1, shape=[1]))
    prediction = tf.matmul(last_state, weight) + bias
    ## L2
    loss = tf.reduce_mean(tf.squared_difference(last_label, prediction))
    #loss = tf.reduce_mean(tf.square(last_label - prediction))    
    ### for cross_entropy
    #last_label_reshaped = tf.reshape(last_label, [-1])
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( labels=last_label_reshaped, logits= prediction ))
    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss)

    l_loss = []
    predicted_data = []

    # *** Start the session ***
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # Train the network
        for i_epochs in range(epochs):
            ### split dataset to batch_size minibatch, to match input plh_batch_x
            ###
            for i_batch in range(dataset_train_length // batch_size):
                i_batch_start = i_batch * batch_size
                i_batch_end = i_batch_start + batch_size

                x = dataset_train_x[
                    i_batch_start:i_batch_end, :].reshape(
                    1, batch_size, number_of_features)
                y = dataset_train_y[i_batch_start:i_batch_end].reshape(
                    1, batch_size, 1)

                feed = {plh_batch_x: x, plh_batch_y: y}

                _loss, _train_step, _pred, _last_label, _prediction = sess.run(
                    fetches=[loss, train_step, prediction, last_label, prediction],
                    feed_dict=feed)

                l_loss.append(_loss)
               
            print('Epoch: {}, Loss: {}'.format(i_epochs, truncate(l_loss[-1], 8)))

        # Test the Network

        for i_test in range(data_length - dataset_train_length - batch_size):

            ##dataset_test_x is still dataframe
            x = dataset_test_x[
                i_test:i_test + batch_size, :].reshape(
                (1, batch_size, number_of_features))
            y = dataset_test_y[
                i_test:i_test + batch_size].reshape(
                (1, batch_size, 1))

            feed2 = {plh_batch_x: x, plh_batch_y: y}

            #_last_state.shape =  BasicRNNCell(num_units
            _last_state, _last_label, test_pred = sess.run(
                fetches=[last_state, last_label, prediction], feed_dict=feed)
            
            _loss, _train_step, _pred, _last_label, _prediction = sess.run(
                fetches=[loss, train_step, prediction, last_label, prediction],
                feed_dict=feed2)            
            
            predicted_data.append(test_pred[-1][0])  # The last one
                       
    # predicted_data = [x for x in predicted_data if str(x) != 'nan']
    predicted_data.extend(
        [None] * (len(data_n) - dataset_train_length - len(predicted_data)))
    df = pd.DataFrame(predicted_data, columns=['predict'])

    return df.set_index(data_n.index[dataset_train_length:])



def plot_2_graphs(plt,dataframe,d_col_name,predict_dataframe):

    x1=[]; y1=[]
    x2=[]; y2=[]

    for index,row in dataframe.iterrows():
        x1.append(int(index))
        y1.append(float(row[d_col_name]))
        
    
    for index, row in predict_dataframe.iterrows():
        x2.append(index)
        y2.append(float(row['predict']))
            
#     plt.plot(x1,y1, marker='o')
#     plt.plot(x2,y2, marker='o')

    plt.plot(x1,y1, 'g', label='raw', alpha=0.25)
    plt.plot(x2,y2, 'r', label='predict')
    
    pred_x_index = x2[0]
    dash_colors = 'b' #'#000017'  #'#171717'
    plt.axvline(x=pred_x_index, linewidth=0.6, color=dash_colors,linestyle='--')
    
    #plt.title('Score in Episodes Timestep')
    
    #plt.xlabel('Episodes / Timesteps')
    plt.xlabel(d_col_name+' days from 2004-01-01', fontsize=16)
    plt.ylabel('percentage %', fontsize=18)
    
    #plt.legend()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    
    plt.show()    
    
    
def main():

    #data = pd.read_csv(filepath).tail(8000)
    df_data = pd.read_csv(filepath)
    
    print(df_data ,df_data.shape)
    
    ### predict value ### 
    #pred_col_name = 'top1'
    #pred_col_name = 'top2'
    pred_col_name = 'top3'
    MA_n = 12
    EMA_n = 10
    
    df_data = splitDataToColumn_01(df_data)
    
    df_data = moving_average(df_data, MA_n, pred_col_name)
    
    df_data = exponential_moving_average(df_data, EMA_n, pred_col_name)
    
    # Cut DataFrame from index 
    df_data = df_data.iloc[20::]
    #Reset index # DataFrame add 1,2,3 index col
    df_data = df_data.reset_index()
    df_data = df_data.drop('index', 1)
    
    
    batch_size = 5 #3
    test_dataset_size = 0.25 #0.05
    num_units = 50 #64 #2 #50  #12 #128
    #learning_rate = 0.001
    learning_rate = 0.0001 # for multi rnn cell
    
    
    epochs = 30 #8  #30
    

    dataset_train_length = len(df_data.index) -\
        int(len(df_data.index) * test_dataset_size)
        
    training_data = df_data.iloc[:dataset_train_length]
        
    features = ['total_bytes','tcp','udp']
            
    pred_col_name_proc = "EMA_" + pred_col_name + "_" + str(EMA_n)
    
    # Train and test the RNN
    predicted_data = network_basic_multiLyaer(
        df_data, features, batch_size,
        dataset_train_length, num_units,
        learning_rate, epochs, pred_col_name_proc
    )    

    # Append test close data and the predicted data
    test_close = pd.DataFrame(df_data['top1'][dataset_train_length::])
    df = pd.concat([training_data, predicted_data, test_close])
       
    # 3) Plot ---------------------------------------------------------
    fig, _ = plt.subplots(facecolor='#ffffff')
    ax0 = plt.subplot2grid(
        (10, 8), (0, 0),
        rowspan=6, colspan=8,
        facecolor='#ffffff'
    )    
    
    
    plot_2_graphs(plt, df_data, pred_col_name_proc, predicted_data)

    tracking_error = np.std(
        predicted_data['predict'] -
        df_data[pred_col_name_proc][dataset_train_length::]) 
    print('Tracking_error: ' + str(tracking_error))


    #plt.show()

    pass


if __name__ == "__main__":
    main()  
    
    
