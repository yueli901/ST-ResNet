import numpy as np
import pandas as pd
from .timestamp import string2timestamp

class STMatrix(object):
    """
    Data cleaning & create all available (X,y) pairs to match the input of the model.
    """
    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        """
        Create a dictionary that matches timestamps (keys) with their indexes (values).
        """
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        """
        Check and print out missing time intervals.
        """
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        """
        Returns the corresponding in-out flow data of the given timestamp.
        @Input: a timestamp
        @Output: corresponding in-out flow data for the timestamp. shape = (1,32,32,2)
        """
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        """
        Check if specified depends are available to be used.
        Returns False if at least one timestamp in the given range of depends is not available in the dataset.
        @Input: a list or a range of timestamps to be checked.
        @Output: True/False
        """
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=3, len_period=3, len_trend=3, PeriodInterval=1, TrendInterval=7):
        """
        Generate the dataset from original dataset to be used directly in the model.
        Instance-based dataset --> sequences with format as (x, y) where x is a sequence of images and y is an image.
        All possible (x, y) pairs are considered as long as the data are available.
        
        @Input:
            len_closeness: # of timeframes to be extracted as closeness
            len_period: # of timeframes to be extracted as period
            len_trend: # of timeframes to be extracted as trend
            PeriodInterval: Duration of time interval for period in # of days
            TrendInterval: Duration of time interval for trend in # of days
        @Output: 
            XC: Closeness data series in shape (None,6,32,32) by default
            XP: Period data series in shape (None,6,32,32) by default
            XT: Trend data series in shape (None,6,32,32) by default
            Y: Current data shape = (None,2,32,32) 
            timestamps_Y: Current timestamp in string format.
        """
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)  # duration between two time frames
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]
        # for an example of depends [range(1, 4), [48, 96, 144], [336, 672, 1008]]
        # in-out flow data with 1,4,48,96,144,336,672,1008 will be used as X to predict the y at i.
        
        # Starting from the maximum # in depends is sufficient.
        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        
        # For a timestamp i, if any timestamp depends in closeness, period and trend has no corresponding data, 
        # this current timestamp i cannot be used in the model.
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
                
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            # Extract `len_closeness` frames prior to the current timestamp i for closeness.
            # For example, for [Timestamp('2013-07-01 00:00:00')]
            # the following closeness timestamps were drawn
            # [Timestamp('2013-06-30 23:30:00'), Timestamp('2013-06-30 23:00:00'), Timestamp('2013-06-30 22:30:00')]

            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            # Extract frames that are 1*PeriodInterval, 2*PeriodInterval, ..., len_period*PeriodInterval prior to i.
            # For default, the in-out flows 1, 2, 3 days before at the same time as timeframe i are extracted.
            
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            # Extract frames that are 1*TrendInterval, 2*TrendInterval, ..., len_trend*TrendInterval prior to i
            # For default, the in-out flows 7, 14, 21 days before at the same time as timeframe i are extracted.
            
            y = self.get_matrix(self.pd_timestamps[i])
            # Extract current timestamp i as the y
            
            if len_closeness > 0:
                XC.append(np.vstack(x_c))
                # a.shape=[2,32,32] b.shape=[2,32,32] c=np.vstack((a,b)) -->c.shape = [4,32,32]
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
            
        XC = np.asarray(XC) 
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y
