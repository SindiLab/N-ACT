import torch
import collections
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from itertools import chain
from anndata import AnnData
import matplotlib.pyplot as plt

class AttentionQuery():
    def __init__(self, scanpy_object:AnnData, split_test:bool= False, which_split:str='test'):
        """
        Params
        ------
        scanpy_object: annData 
            The scanpy object that contains the attention weights and predicted cells
        
        split_test: bool
            If the user wants to run on the entire data, or just a split (e.g. test)
            
        which_split: str
            If we want to split the AnnData, which split should it be
            
        Returns
        -------
        None. The object attributes will be set in place
        """
        if split_test:
            print("==> Splitting the data to 'test' only:")
            self.data = scanpy_object[scanpy_object.obs.split==which_split]
        else:
            self.data = scanpy_object
            
        self.split = split_test;
        self.internal_flag = False
        
    def AssignAttention(self, model=None, local_scObj:AnnData=None, attention_type:str='additive', 
                        inplace:bool=False, correct_preditions_only = True, use_raw_X=True ,verbose=True):
        """
        Assign Attention score to the scanpy object, globally or for a specific dataframe
        
        Params
        ------
        model
            The model we want to use to make predictions
              
        local_scObj: AnnData
            If we want to run a local scanpy object as opposed to the one set for the object
        
        attention_type: str
            The type of attention the fed model was trained. This will be 'additive' for the most part
            
        inplace: bool
            Wheather we want changes to be inplace, or on a copy (that will be returned)
        
        Returns
        -------
        self.data (or test_data) 
            The scanpy object with predictions and attention weights as annotations
                
        self.att_df
            A dataframe that contains Cells by Genes, with the attention weights for each gene as the value of the DF
 
        """
        assert model != None, "You must provide a model for making predictions"
        # set the flag 
        self.attention_weights = True;
        
        if local_scObj is None:
            test_data = self.data.copy()
            if verbose:
                    print("*Caution*: The method is running on the entire data. If this is not what you want, provide scanpy object")
        else:
            test_data = local_scObj;
        
        if use_raw_X:
            # if the counts are sparse:
            try:
                test_tensor = torch.from_numpy(test_data.raw.X.todense())
            except:
                test_tensor = torch.from_numpy(test_data.raw.X)
        else:
            # if the counts are sparse:
            try:
                test_tensor = torch.from_numpy(test_data.X.todense())
            except:
                test_tensor = torch.from_numpy(test_data.X)
        
        if verbose:
            print("==> Calling forward:")
        model.eval()
        
        if verbose:
            print("    -> Making predictions")
            
        with torch.no_grad():
            logits, score, attentive_genes = model(test_tensor.float(), training=False)
            _, predicted = torch.max(logits.squeeze(), 1)
        
        # if this call is for the entirety of the data we have, then we should assign
        ## otherwise we would get a dimension error if we are considering a subset
        predicted=predicted.detach().cpu().numpy()
        score = score.detach().cpu().numpy()
        if attention_type == 'multi-headed':
            score = score.reshape(int(score.shape[0]/8), 8)
        attentive_genes= attentive_genes.detach().cpu().numpy() 

        if local_scObj is None:
            self.predicted=predicted
            self.score = score
            self.attentive_genes= attentive_genes
            if verbose:
                print("    -> Assigning attention weights globally: ")
            test_data.obsm['attention'] = score #attentive_genes
            predicted_str = [f'{i}' for i in predicted]
            test_data.obs['prediction'] = predicted_str
            test_data.obs['prediction'] = test_data.obs['prediction'].astype('str') # changed to str from category since it was causing issues
            # adding a check for the correct data type in the cluster column
            if test_data.obs['cluster'].dtype not 'str':
                test_data.obs['cluster'] = test_data.obs['cluster'].astype('str')
            
        if correct_preditions_only:
            print("    -> **Returning only the correct predictions**")
            test_data = test_data[test_data.obs['cluster']==test_data.obs['prediction']]
            # we will use this adata later on instead if we only want to look at the correct predictions
            self.correct_pred_adata = test_data
            self.internal_flag = True;
        
        if verbose:
            print("    -> Creating a [Cells x Attention Per Gene'] DataFrame")
        
        try:
            att_df = pd.DataFrame(test_data.obsm['attention'].values, index=test_data.obs.index, columns = test_data.var.gene_ids.index)
        except:
            print("    -> Could not locate obs['gene_ids'] attribute. Defaulting to .var instead")
            att_df = pd.DataFrame(test_data.obsm['attention'], index=test_data.obs.index, columns = test_data.var.index)
        
        if local_scObj is None:
            self.att_df = att_df
        
        if inplace:
            if verbose:
                print("    -> Making all changes inplace and returning input data with changes")
            self.data = test_data;
            return self.self.data, att_df
        
        else:
            if verbose:
                print("    -> Returning the annData with the attention weights")
            return test_data, att_df
        
        
    def GetTopN(self, n:int=10, dataframe:pd.DataFrame=None, correct_preditions_only:bool=False,
                rank_mode:str=None, verbose:bool=True):
        """
        Get the top n genes for the entire dataset
        
        Params
        ------
        n: int
            How many top values we want to keep
            
        dataframe: DataFrame
            The dataframe we want to find the top n Genes in
            
        correct_preditions_only:bool
            Whether we want to use only the correct predicitons or all predictions 
        
        rank_mode: str
            The mode we want to use for ranking top n genes
            
        verbose: bool
            If we want to print out dialogue
            
        Returns
        -------
        top_genes_transpose(transpose): pd.DataFrame
            The dataframe containing the top n genes with the shape cells x genes
        
        """
        
        if not hasattr(self, 'attention_weights'):
            print("Please first set the attention weights by calling AssignAttention()")
            return 0;
        
        if verbose:
            print(f"==> Getting Top {n} genes for all cells in the original data")
            print("    -> Be cautious as this may not be cluster specific. If you want cluster specific, call GetTopN_PerClust()")
        
        # make it to be genes x cells
        if dataframe is None:
            att_df_trans = self.att_df.T
            
        else:
            att_df_trans = dataframe.T
            
        # find n largest based on the mode
        ## getting the top n gene expression after averaging over all the cells
        if rank_mode is not None:
            if verbose:
                print(f"    -> Ranking mode: {rank_mode}")
            if rank_mode.lower() == 'mean':
                top_genes_transpose = att_df_trans.loc[att_df_trans.sum(axis=1).nlargest(n, keep='all').index]
                
            elif rank_mode.lower() == 'nlargest':
                top_genes_transpose = att_df_trans.nlargest(n, columns = att_df_trans.columns, keep='all');
                
            else:
                print(">/< Current mode not implemented. Please choose between 'mean' or None for now.")
        
        else:   
            top_genes_transpose = att_df_trans.nlargest(n, columns = att_df_trans.columns, keep='all');

        # return the correct order, which is cells x genes
        if dataframe is None:
            self.top_genes_df = top_genes_transpose
            return self.top_genes_df
        
        else:
            return top_genes_transpose.T
    
    def GetTopN_PerClust(self, n:int=25, model=None, mode:str='TFIDF', top_n_rank_method:str='mean'):
        """
        Get the top n genes for each indivual cluster, and return as 
        
        Params
        ------
        n: int
            How many top values we want to keep
            
        model
            The model we want to use to make predictions
            
        mode: str
            The mode we want to use for identifying top genes and normalizing the values
            
        top_n_rank_method:
            The mode we want to use for ranking top n genes (the mode used for GetTopN method)
        
            
        Returns
        -------
        ** 'mode' dependent **
        self.clust_to_att_dict: dict
            A dictionary mapping each cluster to the attention scores
            
        self.clust_sums_dict: dict
            A dictionary containing the sum of gene attention scores for each cluster
            
        self.top_n_names_dict: dict
            A dictionary mapping containing the name of top n genes for each cluster
                   
        """
        assert model != None, "You must provide a model for making predictions"
        
        print(f"==> Top {n} genes will be selected in {mode} mode")
        
        # dictionary to map clusters to their attention weights, no ranking or filtering
        self.clust_to_att_dict = {};
        
        # dictionary to mapping clusters to series containing their summed attention weights per gene
        self.clust_sums_dict = {};
        
        # dictionary to mapping clusters to dataframes containing the top n genes and their attention weights
        self.top_n_df_dict = {};
        
        # dictionary to mapping clusters to a dataframe containing the top n genes based on their summed attention weights
        self.top_n_names_dict = {};
        
        if self.internal_flag:
            data_to_use = self.correct_pred_adata
        else:
            data_to_use = self.data
        
        print(f"==> Getting Top {n} genes for each cluster in the data")
        iter_list = list(data_to_use.obs.cluster.unique());
        iter_list.sort()
        
        for i in iter_list:
            print(f"    -> Cluster {i}:")
            # get data for the current cluster
            curr_clust = data_to_use[data_to_use.obs.cluster==i]
            print(f"    -> Cells in current cluster: {curr_clust.shape[0]}")
            
            # get the cell x attention per gene df for the current cluster
            curr_att_df = self.att_df.loc[curr_clust.obs.index]
            
            # map the clusters to the attention dataframe
            self.clust_to_att_dict[f'Cluster_{i}'] = curr_att_df;
            
            # get the top n gene dataframe based on the mode
            if mode.lower()=='tfidf':

                self.clust_sums_dict[f'Cluster_{i}'] = curr_att_df.T.sum(axis=1)
                
            else:
                self.top_n_df_dict[f'Cluster_{i}'] = self.GetTopN(n=n, dataframe=curr_att_df, 
                                                                  verbose = False, 
                                                                  rank_mode=top_n_rank_method)
            
            # get the top n gene names in ranked order (highest expression to lowest)
            self.top_n_names_dict[f'Cluster_{i}'] = curr_att_df.T.sum(axis=1).nlargest(n).index
            
            # self.top_n_names_dict[f'Cluster_{i}'] = self.clust_sums_dict[f'Cluster_{i}'].columns.values
            print(f"    >-< Done with Cluster {i}:")
            print()
        
        if mode.lower()=='tfidf':
            return self.clust_to_att_dict, self.clust_sums_dict, self.top_n_names_dict
        
        else:
            # wrong needs to be changed!
            return self.clust_to_att_dict, self.top_n_df_dict, self.top_n_names_dict

    
    def GeneTFIDF(self, top_n_genestrings_dict=None):
        """
        Calculate TF-IDF values for all genes in top n names ranked dictionary
        ***
        @Oscar: This needs to be cleaned up a bit :) 
        ***
        
        """
        
        if top_n_genestrings_dict is None:
            try:
                att_dict = self.top_n_genestrings_dict.copy()
                print("==> Since no dictionary was provided, we will use gene names dictionary as default")
            except:
                print("==> Please either provide a top n gene name list, or set the attribute 'self.top_n_genestrings_dict'")
        else:
            att_dict = top_n_genestrings_dict
        
        topNGenes = pd.DataFrame.from_dict(att_dict)
        
        
        def tf(genes):
            """
            calculate the gene frequency of a list of genes
            """
            freq = {}
            total = len(genes)

            # count occurrences 
            for gene in genes:
                if gene in freq:
                    freq[gene] += 1
                else:
                    freq[gene] = 1

            # generate frequencies 
            for gene in freq:
                freq[gene] = freq[gene] / total
            return freq 
        
        def idf(dictionaries):
            """
            calculate the inverse document frequency of genes in a list dictionaries
            """
            freq = {}
            for document in dictionaries:
                for gene in document:
                    if gene in freq:
                        freq[gene] += 1
                    else:
                        freq[gene] = 1
            for gene in freq:
                freq[gene] = np.log(len(dictionaries) / freq[gene])
            return freq
        
        tf_idf = {}

        # calculate gene frequencies
        print(f"==> Calculating gene frequencies")
        tf_vals = topNGenes.apply(tf, axis=1)

        # calculate inverse document frequencies
        print(f"==> Calculating inverse document frequencies")
        idf_vals = idf(tf_vals)

        # calculate tf-idf values for all genes in ranked_topN_strdf
        print(f"==> Calculating tf-idf values for all ranked genes")
        for document in range(len(tf_vals)):
            for gene in tf_vals[document]:
                tf_idf[gene] = tf_vals[document][gene] * idf_vals[gene]
        return tf_idf
        

    def MakeValuesUnique(self, top_n_dictionary:dict=None, threshold:int=None):
        """
        Class method to make all the values in a dictionary unique with respect to all other values
        
        Params
        ------
        top_n_dictionary:dict
            The dictionary containing the top n gene names for all populations (or smaller populations)
        
        threshold: int
            The threshold for removing common genes: if a gene occurs in threshold many populations, it will be removed. 
        
        Returns
        -------
        att_dict:dict
            The modified top_n_dictionary dictionary based on thresholding
        
        """
        
        if top_n_dictionary is None:
            print("==> Since no dictionary was provided, we will use gene names dictionary as default")
            att_dict = self.top_n_names_dict.copy()
        else:
            att_dict = top_n_dictionary
            
        # concat all the gene names into a list
        all_genes = list(chain.from_iterable(att_dict.values()))
        
        if threshold is None:
            # do not threshold the allowed overlaps
            print("    -> No thresholding... setting overlap bound to inf")
            threshold = np.inf
        
        # find duplicates that appear as many times as the threshold
        duplicate_list = [item for item, count in collections.Counter(all_genes).items() if count > threshold]
        print(f"==> Found {len(duplicate_list)} many duplicates that appear in more than {threshold} cluster(s)")

        for key in att_dict.keys():
            att_dict[key] = [item for item in att_dict[key] if item not in duplicate_list]

        if top_n_dictionary is None:
            self.global_unique_gene_names = att_dict
            
        return att_dict
    
    #------ getters and setters ------
    def GetScanpyObj(self):
        """
        Getter method for returning Scanpy object at any given time
        """
        return self.data
    
    def SetScanpyObj(self, new_data):
        """
        Setter method for setting Scanpy object at any given time
        """
        self.data = new_data