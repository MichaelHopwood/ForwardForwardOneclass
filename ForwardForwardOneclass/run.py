import os
import numpy as np

from base.utils import prepare_dataset, scoring
from forwardforward_general_oneclass_autodiff import GeneralOneclass

import matplotlib.pyplot as plt

def pipeline(model, train, test, val, viz_opt_landscape=False):
    name = model.name
    trainX, trainY = train[:, :-1], train[:, -1]
    valX, valY = val[:, :-1], val[:, -1]
    print("")
    print("Model: %s\n" % name + "-" * 40)
    model.fit(trainX, trainY, valX, valY, viz_opt_landscape=viz_opt_landscape)

    train_pred, train_proba = model.predict(trainX, return_probs=True)
    acc_tr, f1_tr, auc_tr, cm_tr = scoring(trainY, train_pred)
    print("Train Accuracy=%.4f | F1=%.4f | AUC=%.4f" % (acc_tr, f1_tr, auc_tr))
    print(cm_tr)

    plt.figure()
    _,bins = np.histogram(train_proba)
    plt.hist(train_proba[trainY==0], alpha=0.6, bins=bins, label='0')
    plt.hist(train_proba[trainY==1], alpha=0.6, bins=bins, label='1')
    plt.axvline(model.threshold, lw=3, linestyle='--', color='k')
    plt.legend()
    plt.savefig(os.path.join(model.save_folder, 'train_prob_histogram.png'))
    plt.close()
    
    testY = test[:, -1]
    test_pred, test_proba = model.predict(test[:, :-1], return_probs=True)
    acc_te, f1_te, auc_te, cm_te = scoring(testY, test_pred)
    print("Test Accuracy=%.4f | F1=%.4f | AUC=%.4f" % (acc_te, f1_te, auc_te))
    print(cm_te)

    plt.figure()
    _,bins = np.histogram(test_proba)
    plt.hist(test_proba[testY==0], bins=bins, alpha=0.6, label='0')
    plt.hist(test_proba[testY==1], bins=bins, alpha=0.6, label='1')
    plt.axvline(model.threshold, lw=3, linestyle='--', color='k')
    plt.legend()
    plt.savefig(os.path.join(model.save_folder, 'test_prob_histogram.png'))
    plt.close()

    return (acc_tr, f1_tr, auc_tr, cm_tr), (acc_te, f1_te, auc_te, cm_te)


def evaluate(datafile, m, nn_size, n_iters=50, **kwargs):
    acc_train, acc_test = [], []
    f1_train, f1_test = [], []
    auc_train, auc_test = [], []
    for seed in range(n_iters):
        train, test, val, labels = prepare_dataset(f"scripts/Forward-Forward-Network-main/{datafile}", do_normalize=True)
        model = m(nn_size, seed=seed, **kwargs)

        try:
            (acc_tr, f1_tr, auc_tr, cm_tr), (acc_te, f1_te, auc_te, cm_te) = pipeline(model, train, test, val)
        except:
            continue

        acc_train.append(acc_tr)
        acc_test.append(acc_te)
        f1_train.append(f1_tr)
        f1_test.append(f1_te)
        auc_train.append(auc_tr)
        auc_test.append(auc_te)

    print('acc_train')
    print( pd.Series(acc_train).describe() )
    print('acc_test')
    print( pd.Series(acc_test).describe() )
    print('f1_train')
    print( pd.Series(f1_train).describe() )
    print('f1_test')
    print( pd.Series(f1_test).describe() )
    print('auc_train')
    print( pd.Series(auc_train).describe() )
    print('auc_test')
    print( pd.Series(auc_test).describe() )

    np.savetxt(f'{model.save_folder}//acc_train.npy', acc_train)
    np.savetxt(f'{model.save_folder}//acc_test.npy', acc_test)
    np.savetxt(f'{model.save_folder}//f1_train.npy', f1_train)
    np.savetxt(f'{model.save_folder}//f1_test.npy', f1_test)
    np.savetxt(f'{model.save_folder}//auc_train.npy', auc_train)
    np.savetxt(f'{model.save_folder}//auc_test.npy', auc_test)

def read_summary(nm, folder):

    try:
        acc_train = np.loadtxt(f'{folder}//acc_train.npy')
        acc_test = np.loadtxt(f'{folder}//acc_test.npy')
        f1_train = np.loadtxt(f'{folder}//f1_train.npy')
        f1_test = np.loadtxt(f'{folder}//f1_test.npy')
        auc_train = np.loadtxt(f'{folder}//auc_train.npy')
        auc_test = np.loadtxt(f'{folder}//auc_test.npy')

        acc_test = pd.Series(acc_test)
        f1_test = pd.Series(f1_test)
        auc_test = pd.Series(auc_test)

        print(nm + " & \multicolumn{1}{c|}{" f"{round(100*acc_test.mean(),2)} ($\pm$ {round(100*acc_test.std(),2)} )" "} & " f"{round(100*acc_test.max(),2)}" " & \multicolumn{1}{c|}{" f"{round(f1_test.mean(),4)} ($\pm$ {round(f1_test.std(),4)})" "} & " f"{round(f1_test.max(),4)} " "& \multicolumn{1}{c|}{" f"{round(auc_test.mean(),4)} ($\pm$ {round(auc_test.std(),4)}) " "} & " f"{round(auc_test.max(),4)}" r" \\" )

        # print( f"{nm} {round(100*acc_test.mean(),2)}(\pm{round(100*acc_test.std(),2)}) {round(100*acc_test.max(),2)} {round(f1_test.mean(),4)}(\pm{round(f1_test.std(),4)}) {round(f1_test.max(),4)} {round(auc_test.mean(),4)} (\pm{round(auc_test.std(),4)}) {round(auc_test.max(),4)}" )
    except:
        pass

    return acc_test

if __name__ == "__main__":
    import pandas as pd

    print(os.getcwd())

    # dataname = 'iris'
    # datafile = dataname + '.csv'

    dataname = "data_banknote_authentication"
    datafile = dataname + '.txt'

    ### TEMP
    # main_folder = 'results-bp'
    # save_folder = os.path.join(main_folder, dataname)

    # for loss_nm, loss_type in zip(
    #                     ['Goodness', 'GoodnessAdjusted', 'HB-SVDD', 'SVDD', 'LS-SVDD'],
    #                     # ['LS-SVDD'],
    #                     ['origgoodness', 'goodness', 'hbsvdd', 'svdd', 'lssvdd']
    #                     # ['lssvdd']
    #                         ):
    #     for nn_height in [10,25,50,100]:
    #         for nn_width in [1,2]:
    #             dims = [4] + [nn_height]*nn_width
    #             evaluate(datafile, GeneralOneclass, dims, nntype='bp', save_folder=save_folder, loss_type=loss_type)
    ### TEMP ^

    # nn_height = 100
    # nn_width = 2
    loss_type = 'goodness'

    dims = [4,40,40,40]  #+ [nn_height]*nn_width

    main_folder = 'results'
    save_folder = os.path.join(main_folder, dataname)
    train, test, val, labels = prepare_dataset(f"ForwardForwardOneclass/{datafile}", do_normalize=True)
    pipeline(GeneralOneclass(dims, save_folder=save_folder, nntype='ff', seed=1, loss_type=loss_type), train, test, val, viz_opt_landscape=False)
    main_folder = 'results-bp'
    save_folder = os.path.join(main_folder, dataname)
    train, test, val, labels = prepare_dataset(f"ForwardForwardOneclass/{datafile}", do_normalize=True)
    pipeline(GeneralOneclass(dims, save_folder=save_folder, nntype='bp', seed=1, loss_type=loss_type), train, test, val, viz_opt_landscape=False)

    # ### TEMP ^

    sys.exit()


    loss_names = []
    nn_dims = []
    accuracies = []

    ordered_dims = []
    for nn_height in [10,25,50,100]:
        for nn_width in [1,2]:
            dims = [4] + [nn_height]*nn_width
            ordered_dims.append(dims)
    ordered_dims = pd.Series(ordered_dims).astype(str)

    for loss_nm, loss_type in zip(
                        ['Goodness', 'GoodnessAdjusted', 'HB-SVDD', 'SVDD', 'LS-SVDD'],
                        # ['LS-SVDD'],
                        ['origgoodness', 'goodness', 'hbsvdd', 'svdd', 'lssvdd']
                        # ['lssvdd']
                            ):
        for nn_height in [10,25,50,100]:
            for nn_width in [2]:

                dims = [4] + [nn_height]*nn_width

                # train, test, val, labels = prepare_dataset(f"scripts/Forward-Forward-Network-main/{datafile}", do_normalize=True)
                # pipeline(GeneralOneclass(dims, save_folder=save_folder, nntype='ff', loss_type=loss_type), train, test, val, viz_opt_landscape=False)

                # evaluate(datafile, GeneralOneclass, dims, nntype='ff', save_folder=save_folder, loss_type=loss_type)

                nm = f"{loss_nm} ({','.join([str(d) for d in dims])})"
                acc_test = read_summary(nm, GeneralOneclass(dims, save_folder=save_folder, nntype='bp', seed=0, loss_type=loss_type).save_folder)

                loss_names.extend([loss_nm]*len(acc_test))
                nn_dims.extend([dims]*len(acc_test))
                accuracies.extend(acc_test)

        print('\hline')
    
    df = pd.DataFrame()
    df['loss_name'] = loss_names
    df['NN_Dims'] = nn_dims
    df['Accuracy'] = accuracies
    df.Accuracy = 100. * df.Accuracy.astype(float)
    df.NN_Dims = df.NN_Dims.astype(str)
    print(df)

    print("FF AVG ACCURACY", df.Accuracy.mean())

    import seaborn as sns
    plt.figure(figsize=(8,4))
    sns.boxenplot(data=df, x="loss_name", y="Accuracy")
    plt.savefig(f'{main_folder}//lossname_vs_accuracy.png')
    plt.close()

    plt.figure(figsize=(12,4))
    sns.boxenplot(data=df, x="NN_Dims", y="Accuracy")
    plt.savefig(f'{main_folder}//nndims_vs_accuracy.png')
    plt.close()


    df['num_layers'] = [len(n)-1 for n in nn_dims]
    print( df.groupby('num_layers').mean() ) 

    main_folder = 'results-bp'
    save_folder = os.path.join(main_folder, dataname)

    loss_names = []
    nn_dims = []
    accuracies = []

    for loss_nm, loss_type in zip(
                        ['Goodness', 'GoodnessAdjusted', 'HB-SVDD', 'SVDD', 'LS-SVDD'],
                        #['LS-SVDD'],
                        ['origgoodness', 'goodness', 'hbsvdd', 'svdd', 'lssvdd']
                        #['lssvdd']
                            ):
        for nn_height in [10,25,50,100]:
            for nn_width in [2]:

                dims = [4] + [nn_height]*nn_width

               # train, test, val, labels = prepare_dataset(f"scripts/Forward-Forward-Network-main/{datafile}", do_normalize=True)
               # pipeline(GeneralForwardForwardOneclass(dims, save_folder=save_folder, nntype='bp', loss_type=loss_type), train, test, val, viz_opt_landscape=True)

               # evaluate(datafile, GeneralOneclass, dims, nntype='bp', save_folder=save_folder, loss_type=loss_type)

                nm = f"{loss_nm} ({','.join([str(d) for d in dims])})"
                acc_test = read_summary(nm, GeneralOneclass(dims, save_folder=save_folder, nntype='bp', seed=0, loss_type=loss_type).save_folder)

                loss_names.extend([loss_nm]*len(acc_test))
                nn_dims.extend([dims]*len(acc_test))
                accuracies.extend(acc_test)
        print('\hline')
    
    df_bp = pd.DataFrame()
    df_bp['loss_name'] = loss_names
    df_bp['NN_Dims'] = nn_dims
    df_bp['Accuracy'] = accuracies
    df_bp.Accuracy = 100. * df_bp.Accuracy.astype(float)
    df_bp.NN_Dims = df_bp.NN_Dims.astype(str)
    print(df_bp)

    import seaborn as sns
    plt.figure(figsize=(8,4))
    sns.boxenplot(data=df_bp, x="loss_name", y="Accuracy")
    plt.savefig(f'{main_folder}//lossname_vs_accuracy.png')
    plt.close()

    plt.figure(figsize=(12,4))
    sns.boxenplot(data=df_bp, x="NN_Dims", y="Accuracy")
    plt.savefig(f'{main_folder}//nndims_vs_accuracy.png')
    plt.close()




    # plt.figure(figsize=(8,4))
    # fig,ax = plt.subplots(1,1,figsize=(12,4))
    dfs = []
    for loss_nm in ['Goodness', 'GoodnessAdjusted', 'HB-SVDD', 'SVDD', 'LS-SVDD']:
        dfiter = df[df.loss_name==loss_nm]
        dfiter = dfiter.groupby("NN_Dims").max()#.reset_index()

        df_bp_iter = df_bp[df_bp.loss_name==loss_nm]
        df_bp_iter = df_bp_iter.groupby("NN_Dims").max()#.reset_index()

        dfjoin = dfiter.join(df_bp_iter, how='inner', lsuffix='_ff', rsuffix='_bp')
        
        dfjoin['Accuracy_FF - Accuracy_BP'] = ( dfjoin['Accuracy_ff'] - dfjoin['Accuracy_bp'] )
        dfjoin = dfjoin.reset_index()

        dfjoin['loss_name'] = loss_nm
        dfs.append(dfjoin)
    plt.figure(figsize=(10,4))
    dfbarplot = pd.concat(dfs)
    print(dfbarplot)
    print(ordered_dims)
    dfbarplot_sorted = dfbarplot.sort_values(by="NN_Dims", key=lambda column: column.map(lambda e: ordered_dims.tolist().index(e)), inplace=True)
    print(dfbarplot_sorted)
    ax = sns.barplot(data=dfbarplot, x='NN_Dims', y='Accuracy_FF - Accuracy_BP', hue='loss_name', n_boot=0)
    
    # ax = sns.barplot(data=df_bp.groupby(["loss_name", "NN_Dims"]).max().reset_index(), x='NN_Dims', y='Accuracy', hue='loss_name', n_boot=0, hatch='+', alpha=0.5)
    # ax2 = sns.barplot(data=df.groupby(["loss_name", "NN_Dims"]).max().reset_index(), x='NN_Dims', y='Accuracy', hue='loss_name', n_boot=0, hatch='/', ax=ax, alpha=0.5)

    # _df = df_bp.groupby(["loss_name", "NN_Dims"]).max().reset_index()
    # _df['trainer'] = 'ff'
    # _df1 = df.groupby(["loss_name", "NN_Dims"]).max().reset_index()
    # _df1['trainer'] = 'bp'
    # _df = pd.concat([_df, _df1])
    # print(_df)
    # ax = sns.barplot(data=_df[_df.loss_name == 'LS-SVDD'], x='NN_Dims', y='Accuracy', hue='trainer', n_boot=0)

    #dfavgnndims = dfbarplot.groupby('NN_Dims').mean()
    # plt.plot(dfavgnndims.index, dfavgnndims['Accuracy_FF - Accuracy_BP'], lw=3, color='k')
    sns.move_legend(ax, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f'{main_folder}//ff_bp_comparisons_accuracy_on_nndims.png')
    plt.close()

    print(dfbarplot['Accuracy_FF - Accuracy_BP'].mean())
    print(dfbarplot.loc[dfbarplot['loss_name'] != 'LS-SVDD']['Accuracy_FF - Accuracy_BP'].mean())


    df_bp['num_layers'] = [len(n)-1 for n in nn_dims]
    print( df_bp.groupby('num_layers').mean() ) 


    print("FF AVG ACCURACY", df.Accuracy.mean())
    print("BP AVG ACCURACY", df_bp.Accuracy.mean())

