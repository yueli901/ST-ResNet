import os
import time
import pickle
from copy import copy
import numpy as np
import h5py

from preprocessing.STMatrix import STMatrix
from preprocessing.timestamp import timestamp2vec
from preprocessing.MinMaxNormalization import MinMaxNormalization

# path setting
DATAPATH = os.path.abspath('/Users/yueli/Desktop/PhD/research/data/TaxiBJ/')
CACHEPATH = os.path.join(DATAPATH, 'CACHE')

def load_holiday(timeslots, fname=os.path.join(DATAPATH, 'BJ_Holiday.txt')):
    """
    Generate the holiday component of the external variables.
    @Input: a list of timestamps in string format
    @Output: [[1],[1],[0],[0],[0]...] shape = (len(timeslots),1)
    """
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    return H[:, None]  # transform to 2 dimension


def load_meteorol(timeslots, fname=os.path.join(DATAPATH, 'BJ_Meteorology.h5')):
    '''
    Generate the meteorol component of the external variables.
    According to the paper:
    "The weather at future time interval t is unknown.
     Instead, one can use the forecasting weather at time interval t 
     or the approximate weather at time interval t-1." 
    Considering time interval is only 30 minutes, we use weather at t-1 to approximate weather at t in this code.
    @Input: a list of timestamps in string format
    @Output: np.ndarray with 2 dimensions. shape = (len(timeslots),Weather.shape[1]+WindSpeed.shape[1]+Temperature.shape[1])
        by default, shape = (len(timeslots),17+1+1)
    '''
    f = h5py.File(fname, 'r')
    Timeslot = f['date'][:]
    WindSpeed = f['WindSpeed'][:]
    Weather = f['Weather'][:]
    Temperature = f['Temperature'][:]
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1 # change here if weather at t is wanted.
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    # print("meteorol shape: ", WS.shape, WR.shape, TE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    # print('merge_shape:', merge_data.shape)
    return merge_data


def load_stdata(fname):
    """
    Split the data and date(timestamps)
    @Input: filename
    @Output: data, timestamps
    """
    f = h5py.File(fname, 'r')
    data = f['data'][:]
    timestamps = f['date'][:]
    f.close()
    return data, timestamps


def stat(fname):
    """
    Count the valid data.
    For a given dataset, due to missing data, the # of timeslots is smaller than 48*(lastday-firstday+1)
    @Input: filename
    @Output: like below

    ==========stat==========
    data shape: (7220, 2, 32, 32)
    # of days: 162, from 2015-11-01 to 2016-04-10
    # of timeslots: 7776
    # of timeslots (available): 7220
    missing ratio of timeslots: 7.2%
    max: 1250.000, min: 0.000
    ==========stat==========

    """

    def get_nb_timeslot(f):
        """
        Count the number of timeslot of given data.
        Assuming each day is complete with 48 observations.
        @Input: filename
        @Output: 
            nb+timeslot: number of timeslot that the given file is suppose to have if no missing data.
            time_s_str: string format of the start of the first day in record.
            time_e_str: string format of the end of the last day in record.
        """
        s = f['date'][0]
        e = f['date'][-1]
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
        time_s_str, time_e_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, time_s_str, time_e_str

    with h5py.File(fname) as f:
        nb_timeslot, time_s_str, time_e_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / 48)
        mmax = f['data'][:].max()
        mmin = f['data'][:].min()
        stat = '=' * 10 + 'stat' + '=' * 10 + '\n' + \
               'data shape: %s\n' % str(f['data'].shape) + \
               '# of days: %i, from %s to %s\n' % (nb_day, time_s_str, time_e_str) + \
               '# of timeslots: %i\n' % int(nb_timeslot) + \
               '# of timeslots (available): %i\n' % f['date'].shape[0] + \ # number of timeslot available in record
               'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
               'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
               '=' * 10 + 'stat' + '=' * 10
        print(stat)


def remove_incomplete_days(data, timestamps, T=48):
    """
    Remove a certain day which has not 48 timestamps
    @Input: full data and full timestamps
    @Output: data and timestamps with complete 48 timeslots for each day
    """
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1: 
            i += 1
        # only check the start of each day is sufficient
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T: 
            days.append(timestamps[i][:8]) # save days with complete data
            i += T
        # if the timestamp after T timeslots matches with T, the day has complete data
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


def process_dataset(T=48, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
                 len_test=None, preprocess_name='preprocessing.pkl',
                 weekday_data=True, meteorol_data=True, holiday_data=True):
    """
    Load all 4-year data, generate all (x,y) pairs for model input and conduct traine-test split.
    @Output:
        X_train: [XC_train, XP_train, XT_train, ex_feature_train] with shape [[(#train,32,32,6)]*3,(#train,extdata_dim)]
        Y_train: Y_train with shape (#train,32,32,2)
        X_test: [XC_test, XP_tests, XT_test, ext_feature_test] with shape [[(#test,32,32,6)]*3,(#test,extdata_dim)]
        Y_test: Y_test with shape (#train,32,32,2)
        mmn: MinMaxNormalization object containing the scaling parameters (min, max)
        extdata_dim: the dimension of external variables in total
        timestamp_train: timestamps used for the train set (matched to y)
        timestamp_test: timestamps used for the test set (matched to y)
    """
    assert (len_closeness + len_period + len_trend > 0)
    # load data
    # 13 - 16
    data_all = []
    timestamps_all = list()
    for year in range(13, 17):
        fname = os.path.join(
            DATAPATH, 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        stat(fname)
        data, timestamps = load_stdata(fname)
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax normalization
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('data_train shape: ', data_train.shape)
    # data in test is unknown, minmax normalization should consider train set only
    mmn = MinMaxNormalization()
    mmn.fit(data_train) 
    data_all_mmn = [mmn.transform(d) for d in data_all]
    fpkl = open(os.path.join(DATAPATH, CACHEPATH, preprocess_name), 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)  # save the [-1,1] normalization parameters
    fpkl.close()

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    # For each year, convert the original dataset to the dataset in the format of model input.
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
            len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y 
    
    # Extract and generate external variables for all available timestamp in timestamps_Y
    ext_feature = []
    if weekday_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        ext_feature.append(time_feature)
    if holiday_data:
        # load holiday
        holiday_feature = load_holiday(timestamps_Y)
        ext_feature.append(holiday_feature)
    if meteorol_data:
        # load meteorol data
        meteorol_feature = load_meteorol(timestamps_Y)
        ext_feature.append(meteorol_feature)

    ext_feature = np.hstack(ext_feature) if len(
        ext_feature) > 0 else np.asarray(ext_feature)
    extdata_dim = ext_feature.shape[1] if len(
        ext_feature.shape) > 1 else None
    if extdata_dim < 1:
        extdata_dim = None
    if weekday_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'external feature: ', ext_feature.shape)

    XC = np.vstack(XC)  # shape = (15072,6,32,32)
    XP = np.vstack(XP)  # shape = (15072,6,32,32)
    XT = np.vstack(XT)  # shape = (15072,6,32,32)
    Y = np.vstack(Y)  # shape = (15072,2,32,32)

    XC = np.transpose(XC,[0,2,3,1]) # shape = (15072,32,32,6)
    XP = np.transpose(XP,[0,2,3,1]) # shape = (15072,32,32,6)
    XT = np.transpose(XT,[0,2,3,1]) # shape = (15072,32,32,6)
    Y = np.transpose(Y,[0,2,3,1]) # shape = (15072,32,32,2)
    
    # note that 15072 may not be accurate

    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)

    if extdata_dim is not None:
        ext_feature_train, ext_feature_test = ext_feature[:-len_test], ext_feature[-len_test:]
        X_train.append(ext_feature_train)
        X_test.append(ext_feature_test)

    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, extdata_dim, timestamp_train, timestamp_test


def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    '''
    Save the dataset after using process_dataset if it is used for the first time to save time.
    @Input: output from process_dataset
    @Output: a h5 file
    '''
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))
    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()


def read_cache(fname):
    '''
    Read in previously saved post-processed data to be used for model directly.
    '''
    mmn = pickle.load(open(os.path.join(DATAPATH, CACHEPATH, 'preprocessing.pkl'), 'rb'))
    f = h5py.File(fname, 'r')
    num = int(f['num'][()])
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(num):
        X_train.append(f['X_train_%i' % i][:])
        X_test.append(f['X_test_%i' % i][:])
    Y_train = f['Y_train'][:]
    Y_test = f['Y_test'][:]
    external_dim = f['external_dim'][()]
    timestamp_train = f['T_train'][:]
    timestamp_test = f['T_test'][:]
    f.close()
    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test


def load_data(len_closeness, len_period, len_trend, len_test, weekday_data=True, meteorol_data=True, holiday_data=True):
    '''
    Load post-processed data directly or generate the data for the first time and save it.
    '''
    fname = os.path.join(DATAPATH, CACHEPATH, 'TaxiBJ_C{}_P{}_T{}.h5'.format(len_closeness, len_period, len_trend))
    if os.path.exists(fname):
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
            fname)
        print("load %s successfully" % fname)
    else:
        print("post-processed %s not found" % fname)
        print("generating %s for the first time" % fname)
        if os.path.isdir(CACHEPATH) is False:
            os.mkdir(CACHEPATH)
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
            process_dataset(len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,
                         len_test=len_test, weekday_data=True, meteorol_data=True, holiday_data=True)
        cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test)
        print("%s generated and saved successfully" % fname)
    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test