"""Main functions for handling h5ad objects for training and testing NNs."""

import torch
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader


def scanpy_to_dataloader(file_path: str = None,
                         scanpy_object: sc.AnnData = None,
                         train_only: bool = False,
                         test_no_valid: bool = False,
                         batch_size: int = 128,
                         workers: int = 12,
                         log_transform: bool = False,
                         log_base: int = None,
                         log_method: str = "scanpy",
                         verbose=0,
                         raw_x=True):

    """Function to read in (or use an existing) H5AD files to make dataloaders.

    Args:
        file_path: A string that is the path to the .h5ad file.
        scanpy_object: An existing scanpy object that should be used for the
          dataloaders.
        test_no_valid: A boolean to check for a test split only if no validation
          set is available.
        batch_size: An integer indicating the batch size to be used for the
          Pytorch dataloader.
        workers: Number of workers to load/lazy load in data.
        log_transform: Whether we want to take log transorm of the data or not.
        log_base: The log base we want to use. If None, we will use natural log.
        log_method: If we want to take the log using scanpy or PyTorch.
        verbose: Verbosity option indicated as a boolean.
        raw_x:bool: This is a dataset- and platform-dependant variable. This
          option enables using the "raw" X matrix, as defined in Seurat. Useful
          for when preprocessing in R and running N-ACT in PyTorch.

    Returns:
        This function will return two dataloaders:

        (1) A Training data loader consisting of the data (at batch[0]) and
        labels (at batch[1]).

        (2) A Testing data loader consisting of the data (at batch[0]) and
        labels (at batch[1])

    Raises:
        ValueError: If neither a path to an h5ad file or an existing scanpy
          object is provided.
        ValueError: If log base falls outside of the implemented ones.

    """
    if scanpy_object is None and file_path is not None:
        print("==> Reading in Scanpy/Seurat AnnData")
        adata = sc.read(file_path)
    elif scanpy_object is not None and file_path is None:
        adata = scanpy_object
    else:
        raise ValueError("Pleaes either provide a path to a h5ad file, or"
                         " provide an existing scanpy object.")

    if raw_x:
        print("    -> Trying adata.raw.X instead of adata.X!")
        try:
            adata.X = adata.raw.X
        except Exception as e:
            print(f"    -> Failed with message: {e}")
            print("    -> Reverting to adata.X if possible")

    if log_transform and log_method == "scanpy":
        print("    -> Doing log(x+1) transformation with Scanpy")
        sc.pp.log1p(adata, base=log_base)

    print("    -> Splitting Train and Validation Data")
    # train
    train_adata = adata[adata.obs["split"].isin(["train"])]
    # validation or test set
    if train_only:
        valid_adata = None

    elif test_no_valid:
        valid_adata = adata[adata.obs["split"].isin(["test"])]

    else:
        valid_adata = adata[adata.obs["split"].isin(["valid"])]

    # turn the cluster numbers into labels
    print("==> Using cluster info for generating train and validation labels")
    y_train = [int(x) for x in train_adata.obs["cluster"].to_list()]

    if not train_only:
        y_valid = [int(x) for x in valid_adata.obs["cluster"].to_list()]

    print("==> Checking if we have sparse matrix into dense")
    try:
        norm_count_train = np.asarray(train_adata.X.todense())
        if not train_only:
            norm_count_valid = np.asarray(valid_adata.X.todense())
    except:
        print("    -> Seems the data is dense")
        norm_count_train = np.asarray(train_adata.X)
        if not train_only:
            norm_count_valid = np.asarray(valid_adata.X)

    train_data = torch.torch.from_numpy(norm_count_train)
    if not train_only:
        valid_data = torch.torch.from_numpy(norm_count_valid)

    if log_transform and log_method == "torch":
        print("    -> Doing log(x+1) transformation with torch")
        if log_base is None:
            train_data = torch.log(1 + train_data)
            if not train_only:
                valid_data = torch.log(1 + valid_data)
        elif log_base == 2:
            train_data = torch.log2(1 + train_data)
            if not train_only:
                valid_data = torch.log2(1 + valid_data)
        elif log_base == 10:
            train_data = torch.log2(1 + train_data)
            if not train_only:
                valid_data = torch.log2(1 + valid_data)
        else:
            raise ValueError(
                "    -> We have only implemented log base e, 2 and 10 for torch"
            )

    data_and_labels = []
    validation_data_and_labels = []
    for i in range(len(train_data)):
        data_and_labels.append([norm_count_train[i], y_train[i]])
        # since validation will always be less than equal to train size
        try:
            validation_data_and_labels.append([norm_count_valid[i], y_valid[i]])
        except:
            pass

    if verbose:
        print(f"==> sample of the training data: {train_data}")
        if test_no_valid:
            print(f"==> sample of the test data: {valid_data}")
        else:
            if not train_only:
                print(f"==> sample of the validation data: {valid_data}")
            print()

    train_data_loader = DataLoader(data_and_labels,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   sampler=None,
                                   batch_sampler=None,
                                   num_workers=workers,
                                   collate_fn=None,
                                   pin_memory=True)
    if not train_only:
        valid_data_loader = DataLoader(validation_data_and_labels,
                                       batch_size=len(valid_data),
                                       shuffle=True,
                                       sampler=None,
                                       batch_sampler=None,
                                       num_workers=workers,
                                       collate_fn=None,
                                       pin_memory=True)

        return train_data_loader, valid_data_loader
    if train_only:
        return train_data_loader
