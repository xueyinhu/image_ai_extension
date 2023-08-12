from tensorflow.python.keras import Sequential, layers


def getConvBlockM(inc, rate, ks=(3, 3), st=(1, 1), pd_mode='same', dl=(1, 1), gr=1, ub=False, be=True):
    s = Sequential()
    s.add(layers.Conv2D(inc * rate, ks, st, pd_mode, dilation_rate=dl, groups=gr, use_bias=ub))
    if be:
        s.add(layers.BatchNormalization())
        s.add(layers.ELU())
    return s


def getConvBlock(inc, rate, s, ks=3, st=1, pd_mode='same', dl=1, gr=1, ub=False, be=True):
    if s:
        return Sequential([
            getConvBlockM(inc, rate, (ks, 1), (st, 1), pd_mode, (dl, 1), gr, ub, be),
            getConvBlockM(inc, rate, (1, ks), (1, st), pd_mode, (1, dl), gr, ub, be)
        ])
    else:
        return getConvBlockM(inc, rate, (ks, ks), (st, st), pd_mode, (dl, dl), gr, ub, be)

