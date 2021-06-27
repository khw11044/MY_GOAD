import scipy.io
import numpy as np
import pandas as pd
import torchvision.datasets as dset
import os

class Data_Loader:

    def __init__(self, n_trains=None):
        self.n_train = n_trains
        self.urls = [
        "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
        "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names"
        ]

    def norm_kdd_data(self, train_real, val_real, val_fake, cont_indices,val_datas,test_datas):
        symb_indices = np.delete(np.arange(train_real.shape[1]), cont_indices)
        mus = train_real[:, cont_indices].mean(0)
        sds = train_real[:, cont_indices].std(0)
        sds[sds == 0] = 1

        def get_norm(xs, mu, sd):
            bin_cols = xs[:, symb_indices]
            cont_cols = xs[:, cont_indices]
            cont_cols = np.array([(x - mu) / sd for x in cont_cols])            # standard scaling
            return np.concatenate([bin_cols, cont_cols], 1)

        train_real = get_norm(train_real, mus, sds)
        val_real = get_norm(val_real, mus, sds)
        val_fake = get_norm(val_fake, mus, sds)
        test_real = val_real[-50:]
        test_fake = val_fake[-50:]
        val_real = val_real[:-50]
        val_fake = val_fake[:-50]
        return train_real, val_real,val_fake,val_datas, test_real,test_fake,test_datas


# MinMaxScaler
    def cn7_MinMaxScaling_data(self, train_real, val_real, val_fake, val_datas,test_datas):
        def get_norm(xs):
            xs_min = xs.min(axis=0)
            xs_max = xs.max(axis=0)
            scaled_xs = (xs - xs_min) / (xs_max- xs_min)
            return scaled_xs

        train_real = get_norm(train_real)
        val_real = get_norm(val_real)
        val_fake = get_norm(val_fake)
        test_real = val_real[-20:]
        test_fake = val_fake[-20:]
        val_real = val_real[:-20]
        val_fake = val_fake[:-20]
        return train_real, val_real,val_fake,val_datas, test_real,test_fake,test_datas

    def norm_data(self, train_real, val_real, val_fake, val_datas,test_datas):
        mus = train_real.mean(0)
        sds = train_real.std(0)
        sds[sds == 0] = 1

        def get_norm(xs, mu, sd):
            return np.array([(x - mu) / sd for x in xs])

        train_real = get_norm(train_real, mus, sds)
        val_real = get_norm(val_real, mus, sds)
        val_fake = get_norm(val_fake, mus, sds)
        test_real = val_real[-50:]
        test_fake = val_fake[-50:]
        val_real = val_real[:-50]
        val_fake = val_fake[:-50]
        return train_real, val_real,val_fake,val_datas, test_real,test_fake,test_datas

    def norm(self, data, mu=1):
        return 2 * (data / 255.) - mu

    def get_dataset(self, dataset_name, c_percent=None, true_label=1):
        if dataset_name == 'cifar10':
            return self.load_data_CIFAR10(true_label)
        elif dataset_name == 'thyroid':
            return self.Thyroid_train_valid_data()
        elif dataset_name == 'arrhythmia':
            return self.Arrhythmia_train_valid_data() 
        elif dataset_name == 'cn7' or dataset_name == 'cn7_demo':
            return self.Cn7_train_valid_data()
        elif dataset_name == 'kdd':
            return self.KDD99_train_valid_data()
        elif dataset_name == 'kdd_down' or dataset_name == 'kdd_demo':
            return self.down_KDD99_train_valid_data()
        elif dataset_name == 'kddrev':
            return self.KDD99Rev_train_valid_data()
        elif dataset_name == 'ckdd':
            return self.contaminatedKDD99_train_valid_data(c_percent)


    def load_data_CIFAR10(self, true_label):
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)

        trainset = dset.CIFAR10(root, train=True, download=True)
        train_data = np.array(trainset.data)
        train_labels = np.array(trainset.targets)

        testset = dset.CIFAR10(root, train=False, download=True)
        test_data = np.array(testset.data)
        test_labels = np.array(testset.targets)

        train_data = train_data[np.where(train_labels == true_label)]
        x_train = self.norm(np.asarray(train_data, dtype='float32'))
        x_test = self.norm(np.asarray(test_data, dtype='float32'))
        return x_train, x_test, test_labels


    def Arrhythmia_train_valid_data(self):
        data = scipy.io.loadmat("data/arrhythmia.mat")
        samples = data['X']  # 518
        labels = ((data['y']).astype(np.int32)).reshape(-1)

        norm_samples = samples[labels == 0]  # 452 norm
        anom_samples = samples[labels == 1]  # 66 anom

        n_train = len(norm_samples) // 2
        x_train = norm_samples[:n_train]  # 226 train

        val_real = norm_samples[n_train:]
        val_fake = anom_samples

        test_real = val_real[-50:]
        test_fake = val_fake[-50:]
        # val_real = val_real[:-50]
        # val_fake = val_fake[:-50]

        # + 추가 
        # val_datas을 만들어준다. val_real과 val_fake가 test_data를 위해 자른 상태로 val_datas를 만들어준다.
        x_val_fscore = np.concatenate([val_real[:-50],val_fake[:-50]])
        y_val_fscore = np.concatenate([np.zeros(len(val_real[:-50])), np.ones(len(val_fake[:-50]))])
        y_val_fscore_reshape = y_val_fscore.reshape(-1,1)
        val_datas = np.hstack((x_val_fscore,y_val_fscore_reshape))

        # + 추가 
        x_test_fscore = np.concatenate([test_real,test_fake])
        y_test_fscore = np.concatenate([np.zeros(len(test_real)), np.ones(len(test_fake))])
        y_test_fscore_reshape = y_test_fscore.reshape(-1,1)
        test_datas = np.hstack((x_test_fscore,y_test_fscore_reshape))
        return self.norm_data(x_train,val_real,val_fake,val_datas, test_datas)


    def Thyroid_train_valid_data(self):
        data = scipy.io.loadmat("data/thyroid.mat")
        samples = data['X']  # 3772
        labels = ((data['y']).astype(np.int32)).reshape(-1)

        norm_samples = samples[labels == 0]  # 3679 norm
        anom_samples = samples[labels == 1]  # 93 anom

        n_train = (len(norm_samples) * 2) // 3
        x_train = norm_samples[:n_train]  # 1839 train

        val_real = norm_samples[n_train:]
        val_fake = anom_samples

        test_real = val_real[-50:]
        test_fake = val_fake[-50:]
        # val_real = val_real[:-50]
        # val_fake = val_fake[:-50]

        # + 추가 
        # val_datas을 만들어준다. val_real과 val_fake가 test_data를 위해 자른 상태로 val_datas를 만들어준다.
        x_val_fscore = np.concatenate([val_real[:-50],val_fake[:-50]])
        y_val_fscore = np.concatenate([np.zeros(len(val_real[:-50])), np.ones(len(val_fake[:-50]))])
        y_val_fscore_reshape = y_val_fscore.reshape(-1,1)
        val_datas = np.hstack((x_val_fscore,y_val_fscore_reshape))

        # + 추가 
        x_test_fscore = np.concatenate([test_real,test_fake])
        y_test_fscore = np.concatenate([np.zeros(len(test_real)), np.ones(len(test_fake))])
        y_test_fscore_reshape = y_test_fscore.reshape(-1,1)
        test_datas = np.hstack((x_test_fscore,y_test_fscore_reshape))

        return self.norm_data(x_train,val_real,val_fake,val_datas, test_datas)
        

    def Cn7_train_valid_data(self):
        pd_samples = pd.read_csv('data/cn7_samples.csv')
        pd_labels = pd.read_csv('data/cn7_labels.csv')

        labels = pd_labels['PassOrFail'].values
        samples = np.array(pd_samples)


        norm_samples = samples[labels == 0]  # 3679 norm
        anom_samples = samples[labels == 1]  # 93 anom

        n_train = (len(norm_samples) * 2) // 3
        x_train = norm_samples[:n_train]  # 1839 train

        val_real = norm_samples[n_train:]
        val_fake = anom_samples

        test_real = val_real[-20:]
        test_fake = val_fake[-20:]

        # + 추가 
        # val_datas을 만들어준다. val_real과 val_fake가 test_data를 위해 자른 상태로 val_datas를 만들어준다.
        x_val_fscore = np.concatenate([val_real[:-20],val_fake[:-20]])
        y_val_fscore = np.concatenate([np.zeros(len(val_real[:-20])), np.ones(len(val_fake[:-20]))])
        y_val_fscore_reshape = y_val_fscore.reshape(-1,1)
        val_datas = np.hstack((x_val_fscore,y_val_fscore_reshape))

        # + 추가 
        x_test_fscore = np.concatenate([test_real,test_fake])
        y_test_fscore = np.concatenate([np.zeros(len(test_real)), np.ones(len(test_fake))])
        y_test_fscore_reshape = y_test_fscore.reshape(-1,1)
        test_datas = np.hstack((x_test_fscore,y_test_fscore_reshape))

        return self.cn7_MinMaxScaling_data(x_train,val_real,val_fake,val_datas, test_datas)


# kdd dataset =================================================================================================================================

# preprocessing 
    def KDD99_preprocessing(self):
        df_colnames = pd.read_csv(self.urls[1], skiprows=1, sep=':', names=['f_names', 'f_types'])
        df_colnames.loc[df_colnames.shape[0]] = ['status', ' symbolic.']
        df = pd.read_csv(self.urls[0], header=None, names=df_colnames['f_names'].values)
        df_symbolic = df_colnames[df_colnames['f_types'].str.contains('symbolic.')]
        df_continuous = df_colnames[df_colnames['f_types'].str.contains('continuous.')]
        samples = pd.get_dummies(df.iloc[:, :-1], columns=df_symbolic['f_names'][:-1])

        smp_keys = samples.keys()
        cont_indices = []
        for cont in df_continuous['f_names']:
            cont_indices.append(smp_keys.get_loc(cont))

        labels = np.where(df['status'] == 'normal.', 1, 0)
        pd_labels = pd.DataFrame(labels)
        pd_cont_indices = pd.DataFrame(cont_indices)
        samples.to_csv('data/samples.csv',index=None)
        pd_labels.to_csv('data/labels.csv',index=None)                 # --> pd_labels[0].values
        pd_cont_indices.to_csv('data/cont_indices.csv',index=None)     # --> pd_cont_indices[0].values.tolist()
        return np.array(samples), np.array(labels), cont_indices


    def down_KDD99_preprocessing(self):
        samples = pd.read_csv('data/kdd_samples.csv')
        # samples.drop(['Unnamed: 0'], axis = 1, inplace = True)
        pd_labels = pd.read_csv('data/kdd_labels.csv')
        # pd_labels.drop(['Unnamed: 0'], axis = 1, inplace = True)
        pd_cont_indices = pd.read_csv('data/kdd_cont_indices.csv')
        # pd_cont_indices.drop(['Unnamed: 0'], axis = 1, inplace = True)

        labels = pd_labels['0'].values
        cont_indices = pd_cont_indices['0'].values.tolist()

        return np.array(samples), np.array(labels), cont_indices


# data 나누기 
    def KDD99_train_valid_data(self):
        samples, labels, cont_indices = self.KDD99_preprocessing()
        
        anom_samples = samples[labels == 1]  # attacknorm : 396743
        norm_samples = samples[labels == 0]  # norm: 97278 

        n_norm = norm_samples.shape[0]
        ranidx = np.random.permutation(n_norm)
        n_train = n_norm // 2

        x_train = norm_samples[ranidx[:n_train]]

        # n_test, n_valid
        norm_test = norm_samples[ranidx[n_train:]]

        val_real = norm_test
        val_fake = anom_samples

        test_real = val_real[-50:]
        test_fake = val_fake[-50:]
        # val_real = val_real[:-50]
        # val_fake = val_fake[:-50]

        # + 추가 
        # val_datas을 만들어준다. val_real과 val_fake가 test_data를 위해 자른 상태로 val_datas를 만들어준다.
        x_val_fscore = np.concatenate([val_real[:-50],val_fake[:-50]])
        y_val_fscore = np.concatenate([np.zeros(len(val_real[:-50])), np.ones(len(val_fake[:-50]))])
        y_val_fscore_reshape = y_val_fscore.reshape(-1,1)
        val_datas = np.hstack((x_val_fscore,y_val_fscore_reshape))

        # + 추가 
        x_test_fscore = np.concatenate([test_real,test_fake])
        y_test_fscore = np.concatenate([np.zeros(len(test_real)), np.ones(len(test_fake))])
        y_test_fscore_reshape = y_test_fscore.reshape(-1,1)
        test_datas = np.hstack((x_test_fscore,y_test_fscore_reshape))

        return self.norm_kdd_data(x_train,val_real,val_fake,cont_indices,val_datas, test_datas)     # val_real,val_fake는 아직 안나눠짐 val_datas는 이미 나눠짐

    def down_KDD99_train_valid_data(self):
        samples, labels, cont_indices = self.down_KDD99_preprocessing()
        anom_samples = samples[labels == 1]  # norm: 97278

        norm_samples = samples[labels == 0]  # attack: 396743

        n_norm = norm_samples.shape[0]
        ranidx = np.random.permutation(n_norm)
        n_train = n_norm // 2

        x_train = norm_samples[ranidx[:n_train]]
        norm_test = norm_samples[ranidx[n_train:]]

        val_real = norm_test
        val_fake = anom_samples

        test_real = val_real[-50:]
        test_fake = val_fake[-50:]
        # val_real = val_real[:-50]
        # val_fake = val_fake[:-50]

        # + 추가 
        # val_datas을 만들어준다. val_real과 val_fake가 test_data를 위해 자른 상태로 val_datas를 만들어준다.
        x_val_fscore = np.concatenate([val_real[:-50],val_fake[:-50]])
        y_val_fscore = np.concatenate([np.zeros(len(val_real[:-50])), np.ones(len(val_fake[:-50]))])
        y_val_fscore_reshape = y_val_fscore.reshape(-1,1)
        val_datas = np.hstack((x_val_fscore,y_val_fscore_reshape))

        # + 추가 
        x_test_fscore = np.concatenate([test_real,test_fake])
        y_test_fscore = np.concatenate([np.zeros(len(test_real)), np.ones(len(test_fake))])
        y_test_fscore_reshape = y_test_fscore.reshape(-1,1)
        test_datas = np.hstack((x_test_fscore,y_test_fscore_reshape))

        return self.norm_kdd_data(x_train,val_real,val_fake,cont_indices,val_datas, test_datas)     # val_real,val_fake는 아직 안나눠짐 val_datas는 이미 나눠짐




    def demo_KDD99_preprocessing(self):
        demo_samples = pd.read_csv('data/demo_samples.csv')
        demo_samples.drop(['Unnamed: 0'], axis = 1, inplace = True)
        pd_labels = pd.read_csv('data/demo_labels.csv')
        pd_labels.drop(['Unnamed: 0'], axis = 1, inplace = True)
        pd_cont_indices = pd.read_csv('data/demo_cont_indices.csv')
        pd_cont_indices.drop(['Unnamed: 0'], axis = 1, inplace = True)

        demo_labels = pd_labels['0'].values
        demo_cont_indices = pd_cont_indices['0'].values.tolist()

        return np.array(demo_samples), np.array(demo_labels), demo_cont_indices




    def demo_KDD99_test_data(self):
        samples, labels, cont_indices = self.demo_KDD99_preprocessing()
        anom_samples = samples[labels == 1]  # norm: 97278

        norm_samples = samples[labels == 0]  # attack: 396743

        n_norm = norm_samples.shape[0]
        ranidx = np.random.permutation(n_norm)
        n_train = n_norm // 2

        x_train = norm_samples[ranidx[:n_train]]
        norm_test = norm_samples[ranidx[n_train:]]

        val_real = norm_test
        val_fake = anom_samples

        # + 추가 
        x_test_fscore = np.concatenate([val_real,val_fake])
        y_test_fscore = np.concatenate([np.zeros(len(val_real)), np.ones(len(val_fake))])
        y_test_fscore_reshape = y_test_fscore.reshape(-1,1)
        val_datas = np.hstack((x_test_fscore,y_test_fscore_reshape))

        return self.norm_kdd_data(x_train, val_real, val_fake, cont_indices,val_datas)



# kdd99 data ================================================================================

    def KDD99Rev_train_valid_data(self):
        samples, labels, cont_indices = self.KDD99_preprocessing()

        norm_samples = samples[labels == 1]  # norm: 97278

        # Randomly draw samples labeled as 'attack'
        # so that the ratio btw norm:attack will be 4:1
        # len(anom) = 24,319
        anom_samples = samples[labels == 0]  # attack: 396743

        rp = np.random.permutation(len(anom_samples))
        rp_cut = rp[:24319]
        anom_samples = anom_samples[rp_cut]  # attack:24319

        n_norm = norm_samples.shape[0]
        ranidx = np.random.permutation(n_norm)
        n_train = n_norm // 2

        x_train = norm_samples[ranidx[:n_train]]
        norm_test = norm_samples[ranidx[n_train:]]

        val_real = norm_test
        val_fake = anom_samples
        return self.norm_kdd_data(x_train, val_real, val_fake, cont_indices)


    def contaminatedKDD99_train_valid_data(self, c_percent):
        samples, labels, cont_indices = self.KDD99_preprocessing()

        ranidx = np.random.permutation(len(samples))
        n_test = len(samples)//2
        x_test = samples[ranidx[:n_test]]
        y_test = labels[ranidx[:n_test]]

        x_train = samples[ranidx[n_test:]]
        y_train = labels[ranidx[n_test:]]

        norm_samples = x_train[y_train == 0]  # attack: 396743
        anom_samples = x_train[y_train == 1]  # norm: 97278
        n_contaminated = int((c_percent/100)*len(anom_samples))

        rpc = np.random.permutation(n_contaminated)
        x_train = np.concatenate([norm_samples, anom_samples[rpc]])

        val_real = x_test[y_test == 0]
        val_fake = x_test[y_test == 1]
        return self.norm_kdd_data(x_train, val_real, val_fake, cont_indices)


