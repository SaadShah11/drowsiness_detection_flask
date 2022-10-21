import numpy as np
import pandas as pan
import matplotlib.pyplot as plt

#write a funtion that will gather all responses of 1,0,-1 and draw vertical line with low alpha value so they are transparent
#alpha will be set in the plot i.e plt.axvline(alpha=0.2.......................

def drawResponses(dat):
    dat_1 = dat[dat['Response'] == 1]
    dat_0 = dat[dat['Response'] == 0]
    dat_minus_1 = dat[dat['Response'] == -1]
    for f in dat_1['Frame Number']:
        plt.axvline(f, color='green', alpha=0.1)
    for f in dat_0['Frame Number']:
        plt.axvline(f, color='yellow', alpha=0.1)
    for f in dat_minus_1['Frame Number']:
        plt.axvline(f, color='red', alpha=0.1)


def drawPlots():
    dat = pan.read_csv('panda_EAR.csv')
    #print(dat)
    ear = dat['Average EAR']

    m = dat['Average EAR'].head(50).mean()

    t1 = dat['Average EAR'].head(50).mean()-dat['Average EAR'].head(50).std()
    t2 = dat['Average EAR'].head(50).mean()-2*dat['Average EAR'].head(50).std()
    t1_5 = dat['Average EAR'].head(50).mean()-1.5*dat['Average EAR'].head(50).std()
    print("m="+str(m)+"t1="+"t1= "+str(t1)+"t2="+str(t2),"T1.5= "+str(t1_5) )


    #ROLL_WINDOW_SIZE = 3
    LONG_ROLL_WINDOW_SIZE = 300 # should be roughly 15 secs if real frame rate is approx 15 secs

    dat['long rolling'] = dat['Average EAR'].rolling(LONG_ROLL_WINDOW_SIZE, min_periods=LONG_ROLL_WINDOW_SIZE//2).mean()
    #dat['rolling center'] = dat['Average EAR'].rolling(ROLL_WINDOW_SIZE, min_periods=ROLL_WINDOW_SIZE//2, center=True).mean()


    # dat.plot(x='Frame Number', y=['Average EAR', 'long rolling'], kind='line')

    # plt.axhline(y=m, color='r')

    # plt.axhline(y=t1, color='g')

    # plt.axhline(y=t2, color='purple')
    # plt.axhline(y=t1_5, color='red')
    # drawResponses(dat)
    print("before: ",dat['long rolling'].tail(1))

    long_rolling = dat['long rolling'].tail(1).mean()
    return long_rolling

    # plt.show()


if __name__ == "__main__":
    drawPlots()


#T2 = dat['rolling center'].head(50).mean()-2*dat['Average EAR'].head(50).std()


