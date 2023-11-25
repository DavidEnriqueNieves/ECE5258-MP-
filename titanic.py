from pathlib import Path
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mnr import MNR, ConstLrMNR
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple, Self
import pandas as pd
from argparse import ArgumentParser
import numpy as np
from enum import Enum

class Splits(Enum):
    TRAIN = "Train"
    TEST = "Test"
    VAL = "Validation"

    def get_values(self) -> List[str]:
        return [self.TRAIN.value, self.TEST.value, self.VAL.value]
    def get_splits() -> Self:
        return [Splits.TRAIN, Splits.TEST, Splits.VAL]




def print_stats(df : pd.DataFrame, split_name : str):
    print(f"Train size: {len(df)}")
    # statistics regarding training data:
    # gather mean, std, min, max for each column

    summary_table : PrettyTable = PrettyTable()
    print(f"Statistics for training data:")
    summary_table.add_column("Column", ["Mean", "Std", "Min", "Max", "Num unique values"])
    for col in df.columns:
        summary_table.add_column(col, [df[col].mean(), df[col].std(), df[col].min(), df[col].max(), len(df[col].unique())])
        # plot a histogram for each one



    print(f"Statistics for {split_name} data:")
    print(summary_table)

def plot_summary_hists(df : pd.DataFrame, train_df : pd.DataFrame, test_df : pd.DataFrame, val_df : pd.DataFrame ) -> None:
    for col in df.columns:
        fig, axs = plt.subplots(3, 1)
        fig.subplots_adjust(hspace=0.5)
        col_str = str(col).replace(" ", "_")
        hist_path = Path(f"./plots/titanic_splits_{col}_hist.png")
        hist_title = f"Histogram Split for Column {col} of Titanic Dataset"
        i = 0
        for split_name, split_df in zip(["Train", "Test", "Validation"], [train_df, test_df, val_df]):
            axs[i].hist(split_df[col], label=split_name, alpha=0.5)
            axs[i].set_ylabel("Frequency")
            axs[i].set_title(f"{split_name} {col} Histogram")
            i+=1
        fig.suptitle(hist_title)
        fig.savefig(hist_path)

def get_design_matrices(train_df : pd.DataFrame, test_df, val_df ) -> Dict[Splits, Tuple[np.array, np.array]]:
    design_matrix_map : Dict[Splits, Tuple[np.array, np.array]] = {}
    dfs = [train_df, test_df, val_df]
    for (df, split_name) in zip(dfs, Splits.get_splits()):
        X = np.array(df.drop("Survived", axis=1))
        print(f"X_{split_name} Shape is {X.shape}")
        X = np.append(X, np.ones((len(X), 1)), axis=1)
        X = X.T
        print(f"X_{split_name} Shape is now {X.shape}")
        y = np.array(df["Survived"])
        print(f"y_{split_name} Shape is {y.shape}")
        design_matrix_map[split_name] = (X, y)
    return design_matrix_map

def MNR_sanity_test():
    C : int = 2
    D : int = 3
    test_mnr = MNR(C, D)
    preds = np.array([[0.001, 0.9], [0.01, 0.99], [0.99, 0.1]])
    labels = np.array([[1, 0], [1, 0], [0, 1]])
    ace = test_mnr.measure_ace(preds, labels)
    # avg_01_loss = test_mnr.measure_zero_one_loss(preds, labels)
    # print(f"ACE is {ace}")
    # print(f"Average 0-1 Loss is {avg_01_loss}")
    # assert int(avg_01_loss) == 1

def plot_one_run(design_matrix_map, C, D, epochs=100_000, initial_lr=0.01, mod=1000):
    mnr = ConstLrMNR(C, D)
    # convert t to a one-hot vector
    t = np.array([np.array([1, 0]) if x == 0 else np.array([0, 1]) for x in design_matrix_map[Splits.TRAIN][1]])
    t = t.T

    weights, ace_list = mnr.train(design_matrix_map[Splits.TRAIN][0], t, epochs, initial_lr, mod)

    # plot the ACE vs epoch
    fig, axs = plt.subplots(1, 1)
    fig.subplots_adjust(hspace=0.5)
    axs.plot(ace_list)
    axs.set_ylabel("ACE")
    axs.set_xlabel("Epoch")
    plot_path = Path(f"./plots/titanic_ace_vs_epoch?epochs={epochs}init_lr={initial_lr}.png")
    plot_title = f"ACE vs Epoch for Titanic Dataset with \n Epochs={epochs} and Initial Learning Rate={initial_lr}"
    fig.suptitle(plot_title)
    fig.savefig(Path(plot_path))
    return weights, ace_list

def save_confusion_matrix(model,C : int, s_preds : np.array, t : np.array, epochs : int, initial_lr : float, split_name : str ):
    # generating a confusion matrix
    conf_mat = np.zeros((C, C))

    mod = 100
    i = 0
    for (s, t) in zip(s_preds, t):
        if i % mod == 0:
            print(f"Predicted: {np.argmax(s)} Actual: {np.argmax(t)}")
        predicted = np.argmax(s)
        actual = np.argmax(t)
        conf_mat[predicted][actual] += 1

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat.T)
    disp.plot()
    model = ConstLrMNR(C, D)
    plot_title = f"Confusion Matrix for predictions of {model.model_name} on Titanic Dataset \n ({split_name} set with Epochs={epochs} and Initial Learning Rate={initial_lr}"
    plt.suptitle(plot_title)
    plt.savefig(Path(f"./plots/titanic_confusion_matrix?epochs={epochs}init_lr={initial_lr}split={split_name}.png"))

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--debug", action="store_true", help="debug mode")
    args =  argparser.parse_args()
    if args.debug:
        import debugpy
        debug_port = 5678
        print(f"Waiting for debugger to attach on {debug_port}")
        debugpy.listen(debug_port)
        debugpy.wait_for_client()
    
    print(f"##############################")
    print(f"#  3A")
    print(f"##############################")

    data_path = Path("./data/Titanic-MP3.csv")

    df : pd.DataFrame = None
    with data_path.open() as f:
        df = pd.read_csv(f)
    print(df.head())

    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Columns: {df.columns}")
    dropped_cols = ["PassengerId", "Name", "Ticket", "Cabin"]

    print("Dropping the following columns")
    df.drop(dropped_cols, axis=1, inplace=True)
    print(f"Columns are now: {df.columns}")
    print(f"Number of columns: {len(df.columns)}")

    cols_to_onehot : List[str] = ["Sex", "Pclass", "Embarked", "Survived"]

    unique_val_arrs : List[List] = []

    # remove all rows with NaN values
    df.dropna(inplace=True)

    for col in cols_to_onehot:
        print(f"One-hot encoding column {col}")
        unique_col_values = df[col].unique()
        print(f"Unique values: {unique_col_values}")
        unique_val_arrs.append(unique_col_values)

        one_hot_lambda = lambda x: 1 if x == unique_col_values[0] else 0
        vectorize_func = np.vectorize(one_hot_lambda)
        df[col] = vectorize_func(df[col])

        # assert len(df[col].unique()) == 2, f"Column {col} has more than 2 unique values"
        # assert len(unique_col_values) == 2, f"Column {col} has more than 2 unique values"
    
    print(df.head())

    num_rows = len(df)

    # shuffle the rows
    df = df.sample(frac=1)

    # split into train, test, and validation
    train_df = df.iloc[:int(num_rows * 1/3)]
    test_df = df.iloc[int(num_rows * 1/3):int(num_rows * 2/3)]
    val_df = df.iloc[int(num_rows * 2/3):]
    assert len(train_df) + len(test_df) + len(val_df) == len(df), f"Lengths don't add up: {len(train_df)} + {len(test_df)} + {len(val_df)} != {len(df)}"
    assert not train_df.index.isin(test_df.index).any(), "Train and test overlap"
    assert not train_df.index.isin(val_df.index).any(), "Train and val overlap"
    assert not test_df.index.isin(val_df.index).any(), "Test and val overlap"
    # make sure train test and validation don't have the same rows
    assert len(train_df.index.intersection(test_df.index)) == 0, "Train and test overlap"



    print_stats(train_df, "Train")
    print_stats(test_df, "Test")
    print_stats(val_df, "Validation")
    
    plot_summary_hists(df, train_df, test_df, val_df)

    print("##############################")
    print("#  3B")
    print("##############################")

    design_matrix_map : Dict[Splits, Tuple[np.array, np.array]] = get_design_matrices(train_df, test_df, val_df)

    C : int = 2 # number of classes (two classes: survived or not)
    D =  len(df.columns) - 1 # subtract one since we don't count the survival column, as it is the output
    print(f"D is {D}")
    print(f"C is {C}")

    
    MNR_sanity_test()

    epochs=100_000
    initial_lr=0.001
    mod=1000

    weights, ace_list = plot_one_run(design_matrix_map, C, D, epochs=epochs, initial_lr=initial_lr, mod=100)
    ace_list = np.array(ace_list)
    best_epoch = np.argmin(ace_list)
    best_ace = ace_list[best_epoch]
    print(f"Best ACE is {best_ace} at epoch {best_epoch}")
    print(f"Evaluating best weights on training and test sets with zero-one loss...")

    W_best = weights[best_epoch]

    zero_one_losses : PrettyTable = PrettyTable()
    for split in Splits.get_splits():
        print(f"Split: {split.value}")
        s_preds = ConstLrMNR.predict(design_matrix_map[split][0], W_best)
        zero_one_loss = ConstLrMNR.measure_zero_one_loss(s_preds.T, design_matrix_map[split][1].T)
        save_confusion_matrix(ConstLrMNR, C, s_preds.T, design_matrix_map[split][1].T, epochs, initial_lr, split.value)
        print(f"{split.value} 0-1 Loss is {zero_one_loss}")
        zero_one_losses.add_row([split.value, zero_one_loss])
    
    print(f"Zero-One Losses for Titanic Dataset with Epochs={epochs} and Initial Learning Rate={initial_lr}")
    print(zero_one_losses)