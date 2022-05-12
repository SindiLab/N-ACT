# std libs
import os
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report as class_rep

# torch libs
import torch
import torch.nn as nn

def count_parameters(model):
    """ 
    Count the total number of parameters in a model
    
    Params
    ------
        model: torch model
            A pytorch model which will be initilized with xavier weights

    Returns
    -------
        the number of *trainable* parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def detailed_count_parameters(model):
    """
    Count the total number of parameters in a model in detail, printed in a pretty table
    
    Params
    ------
        model: torch model
            A pytorch model which will be initilized with xavier weights

    Returns
    -------
        the number of *trainable* parameters in a model. It will also print the table
    
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    

def init_weights_xavier_uniform(model):
    """
    Initializing the weights of a model with Xavier uniform distribution
    
    Params
    ------
        model
            A pytorch model which will be initilized with xavier weights

    Returns
    -------
        the updated weights of the model
    """
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
        try:
            model.bias.data.fill_(0.01)
        except Exception:
            # our model doesnt have biases (or something bad is going on)
            pass
        
def init_weights_xavier_normal(model):
    """
    Initializing the weights of a model with Xavier normal distribution
    
    Params
    ------
        model
            A pytorch model which will be initilized with xavier weights

    Returns
    -------
        the updated weights of the model
    """
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_normal_(model.weight)
        try:
            model.bias.data.fill_(0.01)
        except Exception:
            # our model doesnt have biases (or something bad is going on)
            pass
        
def load_model(model, pretrained_path):
    """
    Loading pre-trained weights of a model
    
    Params
    ------
        model
            A pytorch model which will be updated with the pre-trained weights
        pretrained_path:str
            The path to where the .pth file is saved

    Returns
    -------
        model
            The updated model
        trained_epoch: int
            The number of epochs the model was trained
    """
    weights = torch.load(pretrained_path)
    try:
        trained_epoch = weights['epoch']
    except:
        trained_epoch = 50 #this is an arbitrary number that NACT is trained to... this is to have the epoch information
    pretrained_dict = weights['Saved_Model'].state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return model, trained_epoch 


def save_checkpoint_classifier(model, epoch:int, iteration:int, prefix:str="", dir_path:str=None):
    """
    Saving pre-trained model for inference

    Params
    ------
        model
            PT model which we want to save
        epoch:int
            The current epoch number (will be used in the filename)
        iteration:
            Current iteration (will be used in the filename)
        prefix (optional):str
            Prefix to the filename
        dir_path (optional):str
            Path to save the pre-trained model

    Returns
    -------
        None.
    """

    if not dir_path:
        dir_path = "./NACT-Weights/"

    model_out_path = dir_path + prefix +f"model_epoch_{epoch}_iter_{iteration}.pth"
    state = {"epoch": epoch ,"Saved_Model": model}
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    torch.save(state, model_out_path)
    print(f"Classifier Checkpoint saved to {model_out_path}")

    
def save_best_classifier(model, prefix="", dir_path = None):
    """
    Saving the best pre-trained model for inference

    Params
    ------
        model-> PT model which we want to save
        iteration -> current iteration (will be used in the filename)
        prefix (optional)-> a prefix to the filename
        dir_path (optional)-> path to save the pre-trained model
    Returns
    -------
        None.
    """

    if not dir_path:
        dir_path = "./ClassifierWeights/"

    model_out_path = dir_path + prefix +f"model_Best.pth"
    state = {"Saved_Model": model}
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    torch.save(state, model_out_path)
    print(f"Current best model saved to {model_out_path}")
        
def evaluate_classifier(valid_data_loader, cf_model, 
                        classification_report:bool = False,
                        device:str=None):
    """
    Evaluating the performance of the network on validation/test dataset

    Params
    ------
        valid_data_loader: Torch dataloader
            A dataloader of the validation or test dataset
        cf_model: Torch model 
            The model which we want to use for validation
        classification_report:bool
            If you want to enable classification report
        device:str 
            If you want to run the evaluation on a specific device

    Returns
    -------
        None

    """
    ##### Potential bug: if a user has GPUs but does not want to use them by default!
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("==> Evaluating on Validation Set:")
    total = 0;
    correct = 0;
    # for sklearn metrics
    y_true = np. array([])
    y_pred = np. array([])
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_data_loader):
            features, labels = data
            labels = labels.to(device)
            outputs, alphas, _ = cf_model(features.float().to(device), training=False)
            _, predicted = torch.max(outputs.squeeze(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # get all the labels for true and pred so we could use them in sklearn metrics
            y_true = np.append(y_true,labels.detach().cpu().numpy())
            y_pred = np.append(y_pred,predicted.detach().cpu().numpy())
            
    print(f'    -> Accuracy of classifier network on validation set: {(100 * correct / total):4.4f} %' )
    # calculating the precision/recall based multi-label F1 score
    macro_score = f1_score(y_true, y_pred, average = 'macro' )
    w_score = f1_score(y_true, y_pred,average = 'weighted' )
    print(f'    -> Non-Weighted F1 Score on validation set: {macro_score:4.4f} ' )
    print(f'    -> Weighted F1 Score on validation set: {w_score:4.4f} ' )
    if classification_report:
        print(class_rep(y_true,y_pred))
    
    return y_true, y_pred, macro_score


def transfer_CellTypes(scanpy_data, cf_model, inplace:bool=True, device:str=None):
    """
    Evaluating the performance of the network on validation/test dataset

    Params
    ------
        valid_data_loader
            A dataloader of the validation or test dataset
        cf_model
            The model which we want to use for validation
        inplace:bool
            If we want to make the modifications directly to the passed on object
        device:str 
            If we want to run the evaluation on a specific device

    Returns
    -------
        None

    """
    ##### This could be a bug if a user has GPUs but it not using them!

    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    print("==> Checking if we have sparse matrix into dense")
    try:
        np_data = np.asarray(scanpy_data.X.todense());
    except:
        print("    -> Seems the data is dense")
        np_data = np.asarray(scanpy_data.X);

    print("==> Making predictions:")
    
    with torch.no_grad():
            features = torch.from_numpy(np_data)
            outputs, alphas, _ = cf_model(features.float().to(device), training=False)
            _, predicted = torch.max(outputs.squeeze(), 1)
            
    nact_labels = predicted.detach().cpu().numpy()
    if inplace:
        scanpy_data.obs['NACT_Labels'] = nact_labels
        scanpy_data.obs['NACT_Labels'] = scanpy_data.obs['NACT_Labels'].astype('category')
    
    print(">-< Done")
    return nact_labels
    
    
    
#######################

#------------ Debug utils
## Removed for public release

#######################