import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_rsi_values(data, period = 14):
    """
    RSI indicator value calculation
    Input: 
    data frame with price of the stock
    period: moving window for gain and loss
    Output: 
    Price and RSI values
    """
    rsi_data = pd.DataFrame(index = data.index)
    rsi_data['price'] = data['price']
    # change
    rsi_data['change'] = rsi_data['price'].diff(periods = 1)
    # gain
    rsi_data['gain'] = rsi_data['change']
    rsi_data.loc[rsi_data['gain'] < 0, ['gain']] = 0.0
    # loss
    rsi_data['loss'] = rsi_data['change']
    rsi_data.loc[rsi_data['loss'] > 0, ['loss']] = 0.0
    rsi_data['loss'] = abs(rsi_data['loss'])
    # average gain
    rsi_data['avg_gain'] = rsi_data['gain'].rolling(window = period).mean()
    # average loss
    rsi_data['avg_loss'] = rsi_data['loss'].rolling(window= period).mean()
    # rs
    rsi_data['rs'] = rsi_data['avg_gain'] / rsi_data['avg_loss']
    # rsi
    rsi_data['rsi'] = 100 - 100 / (1 + rsi_data['rs'])
    return rsi_data[['price', 'rsi']]


def get_rsi_signal(rsi, buy_threshold = 20, sell_threshold = 80):
    """
    RSI buy sell signal calculation
    Input:
    data: data frame with rsi indicator values
    buy_threshold: threshold for getting buy signal
    sell_threshold: threshold for getting sell signal
    Output: data frame with buy and sell signal
    """
    signals = pd.DataFrame(index=rsi.index)
    signals['price'] = rsi['price'] 
    signals['rsi'] = rsi['rsi']
    signals['buy']= 0.0
    signals['sell']= 0.0
    signals['buy'] = np.where(signals['rsi'] < buy_threshold, -1.0, 0.0)
    signals['sell'] = np.where(signals['rsi'] > sell_threshold, 1.0, 0.0)
    signals['buy'] = signals['buy'].diff()
    signals['sell'] = signals['sell'].diff()
    signals.loc[signals['buy']==-1.0,['buy']]=0 
    signals.loc[signals['sell']== 1.0,['sell']]=0 
    signals['buy_sell'] = signals['buy'] + signals['sell']
    return signals[['price','buy_sell']]

def plot_rsi_buy_sell(rsi, signals, buy_threshold = 20, sell_threshold = 80):
    """
    Plot rsi with buy and sell signal
    """
    graph = plt.figure(figsize=(20,5))
    ax2 = graph.add_subplot(1,1,1)
    rsi[['rsi']].plot(ax=ax2,title = 'RSI signals')
    ax2.axhline(y= buy_threshold, color = "g", lw = 2.)
    ax2.axhline(y= sell_threshold, color = "r", lw = 2.)
    ax2.plot(signals.loc[signals.buy_sell == 1].index, rsi.rsi[signals.buy_sell == 1],"^", markersize = 12, color ='g')
    ax2.plot(signals.loc[signals.buy_sell == -1].index, rsi.rsi[signals.buy_sell == -1],"v", markersize = 12, color ='m')
    # plt.show()
    plt.show()
    
    
 