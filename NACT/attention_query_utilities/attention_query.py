"""Class and function implementation regarding extracting attention values."""

from anndata import AnnData
import collections
from itertools import chain
import numpy as np
import pandas as pd
import torch
from ..utilities import sparse_to_dense


class AttentionQuery():
    """ Class implementation for extracting and querying attention weights.

    This class contains the core methods for extracting cluster-specific
    attention weights, which are used for querying and interpretability.

    Attributes:
        data: The scanpy object that we want to make predictions on.
          Note: The dataframe can be changed or retrived via the defined
          "setter" and "getter" methods.
        split: The split of the scanpy data we want to use, e.g. "test" split.
        attention_weights: The attnetion weights extracted from the model.
        predicted: Model predictions over the chosen split.
        score: The gene score matrix.
        attentive_genes: Top genes with the highest weights (attention values).
        top_genes_df: A dataframe ranked based on the top attentive genes.

    """

    def __init__(self,
                 scanpy_object: AnnData,
                 split_test: bool = False,
                 which_split: str = "test"):
        """Initializer of the AttentionQuery class.

        Args:

            scanpy_object: The scanpy object that contains the attention weights
              and predicted cells.
            split_test: A boolean indicating whether the user wants to get
              attention weights for the entire data (when "False"), or just a
              split (when set to "True").
            which_split: The name of the split that we are interested in and
              which exists in the AnnData.

        """
        if split_test:
            print("==> Splitting the data to 'test' only:")
            self.data = scanpy_object[scanpy_object.obs.split == which_split]
        else:
            self.data = scanpy_object

        self.split = split_test
        self._correct_predictions_only_flag = False

    def assign_attention(self,
                         model=None,
                         local_scanpy_obj: AnnData = None,
                         attention_type: str = "additive",
                         inplace: bool = False,
                         correct_predictions_only=True,
                         use_raw_x=True,
                         verbose=True):
        """The method to assign attention score to a scanpy object.

        Args:
            model: The model we want to use to make predictions and extract
              attention weights from.
            local_scanpy_obj: An AnnData object that would locally replace the
              scanpy object that was set in the constructor for the object.
            attention_type: The type of attention the inputted model was
              trained with. This will be 'additive' in most cases (even when
              NACT included projection blocks).
            inplace: Wheather we want changes to be inplace, or on a copy (in
              which case it will be returned.

        Returns:
            The method will return:

            (1) A scanpy object with predictions and
            attention weights as annotations (will either be the).

            (2) A dataframe consisting of genes as columns and attention weights
                as the corresponding row values.

        Raises:
            ValueError: An error occured during reading the trained model.

        """
        if model is None:
            raise ValueError("Please provide a model for making predictions and"
                             " extracting attention weights.")
        # set the flag
        self.attention_weights = True

        if local_scanpy_obj is None:
            test_data = self.data.copy()
            if verbose:
                print("*Caution*: The method is running on the entire data."
                      " If this is not what you want, provide scanpy"
                      "object.")
        else:
            test_data = local_scanpy_obj

        if use_raw_x:
            test_tensor = torch.from_numpy(sparse_to_dense(test_data.raw))
        else:
            test_tensor = torch.from_numpy(sparse_to_dense(test_data))

        if verbose:
            print("==> Calling forward:")

        model.eval()

        if verbose:
            print("    -> Making predictions")

        with torch.no_grad():
            logits, score, attentive_genes = model(test_tensor.float(),
                                                   training=False)
            _, predicted = torch.max(logits.squeeze(), 1)

        # If this call is for the entirety of the data we have, then we should
        # assign all cells, otherwise we would get a dimension error if we are
        # considering a subset.
        predicted = predicted.detach().cpu().numpy()
        score = score.detach().cpu().numpy()
        if attention_type == "multi-headed":
            score = score.reshape(int(score.shape[0] / 8), 8)
        attentive_genes = attentive_genes.detach().cpu().numpy()

        if local_scanpy_obj is None:
            self.predicted = predicted
            self.score = score
            self.attentive_genes = attentive_genes
            if verbose:
                print("    -> Assigning attention weights globally: ")
            test_data.obsm["attention"] = score  #attentive_genes
            predicted_str = [f"{i}" for i in predicted]
            test_data.obs["prediction"] = predicted_str
            test_data.obs["prediction"] = test_data.obs["prediction"].astype(
                "str"
            )  # changed to str from category since it was causing issues
            # adding a check for the correct data type in the cluster column
            if not test_data.obs["cluster"].dtype == str:
                test_data.obs["cluster"] = test_data.obs["cluster"].astype(
                    "str")

        if correct_predictions_only:
            print("    -> **Returning only the correct predictions**")
            test_data = test_data[test_data.obs["cluster"] ==
                                  test_data.obs["prediction"]]
            # We will use this adata later on instead if we only want to look
            # at the correct predictions.
            self.correct_pred_adata = test_data
            self._correct_predictions_only_flag = True

        if verbose:
            print("    -> Creating a [Cells x Attention Per Gene] DataFrame")

        try:
            att_df = pd.DataFrame(test_data.obsm["attention"].values,
                                  index=test_data.obs.index,
                                  columns=test_data.var.gene_ids.index)
        except Exception as _:
            print(
                "    -> Could not locate obs['gene_ids'] attribute. Defaulting"
                " to .var instead")
            att_df = pd.DataFrame(test_data.obsm["attention"],
                                  index=test_data.obs.index,
                                  columns=test_data.var.index)

        if local_scanpy_obj is None:
            self.att_df = att_df

        if inplace:
            if verbose:
                print(
                    "    -> Making all changes inplace and returning input data"
                    " with changes")
            self.data = test_data
            return self.self.data, att_df

        else:
            if verbose:
                print("    -> Returning the annData with the attention weights")
            return test_data, att_df

    def get_top_n(self,
                  n: int = 10,
                  dataframe: pd.DataFrame = None,
                  rank_mode: str = None,
                  verbose: bool = True):
        """Class method for getting the top n genes for the entire dataset.

        Args:
            n: An integer indicating the number of top genes desired.
            dataframe: The dataframe we want to find the top n Genes in.
            rank_mode: The mode we want to use for ranking top "n" genes.
            verbose: If we want to print out a complete dialogue.

        Returns:
            A dataframe containing the top n genes with the shape cells x genes.

        Raises:
           NotImplementedError: An error occured if "rank_mode" argument is not
             one of the existing modes.

        """

        if not hasattr(self, "attention_weights"):
            print("Please first set the attention weights by calling"
                  "AssignAttention()")
            return 0

        if verbose:
            print(f"==> Getting Top {n} genes for all cells in the original"
                  " data")
            print("    -> Be cautious as this may not be cluster specific."
                  " If you want cluster specific, pleae call"
                  " 'get_top_n_per_cluster()' method.'")

        # make it to be genes x cells
        if dataframe is None:
            att_df_trans = self.att_df.T

        else:
            att_df_trans = dataframe.T

        # Finding n largest genes (features based on the mode:
        if rank_mode is not None:
            if verbose:
                print(f"    -> Ranking mode: {rank_mode}")
            if rank_mode.lower() == "mean":
                top_genes_transpose = att_df_trans.loc[att_df_trans.sum(
                    axis=1).nlargest(n, keep="all").index]

            elif rank_mode.lower() == "nlargest":
                top_genes_transpose = att_df_trans.nlargest(
                    n, columns=att_df_trans.columns, keep="all")

            else:
                raise NotImplementedError(f"Your provided mode={rank_mode} has"
                                        "not been implemented yet. Please"
                                        " choose between 'mean' or 'None' for"
                                        "now.")

        else:
            top_genes_transpose = att_df_trans.nlargest(
                n, columns=att_df_trans.columns, keep="all")

        # return the correct order, which is cells x genes
        if dataframe is None:
            self.top_genes_df = top_genes_transpose
            return self.top_genes_df

        else:
            return top_genes_transpose.T

    def get_top_n_per_cluster(self,
                              n: int = 25,
                              model=None,
                              mode: str = "TFIDF",
                              top_n_rank_method: str = "mean"):
        """Get the top n genes for each indivual cluster.

        Args:
            n: An integer indicating the number of top genes to keep.
            model: The model we want to use to make predictions
            mode: The mode we want to use for identifying top genes and
              normalizing the values
            top_n_rank_method: The mode we want to use for ranking top n genes
              (the mode used for get_top_n method different than "mode"
              argument) for this method.

        Returns:

        This method's reutns are "mode" dependent, and will return three
        objects:

        (1) A dictionary mapping each cluster to the attention scores.

        (2) A dictionary containing the sum of gene attention scores for each
            cluster.

        (3) A dictionary mapping containing the name of top n genes for each
            cluster.

        Raises:
            ValueError: An error occured during reading the trained model.

        """
        if model is None:
            raise ValueError("Please provide a model for making predictions and"
                             " extracting attention weights.")

        print(f"==> Top {n} genes will be selected in {mode} mode")

        # Dictionary to map clusters to their attention weights, no ranking
        # or filtering.
        self.clust_to_att_dict = {}
        # Dictionary to map clusters to series containing their summed attention
        # weights per gene.
        self.clust_sums_dict = {}
        # Dictionary to mapping clusters to dataframes containing the top n
        # genes and their attention weights.
        self.top_n_df_dict = {}
        # dictionary to mapping clusters to a dataframe containing the top n
        # genes based on their summed attention weights.
        self.top_n_names_dict = {}

        if self._correct_predictions_only_flag:
            data_to_use = self.correct_pred_adata
        else:
            data_to_use = self.data

        print(f"==> Getting Top {n} genes for each cluster in the data")
        iter_list = list(data_to_use.obs.cluster.unique())
        iter_list.sort()

        for i in iter_list:
            print(f"    -> Cluster {i}:")
            # get data for the current cluster
            curr_clust = data_to_use[data_to_use.obs.cluster == i]
            print(f"    -> Cells in current cluster: {curr_clust.shape[0]}")

            # Getting the cell x attention per gene df for the current cluster.
            curr_att_df = self.att_df.loc[curr_clust.obs.index]
            # Mapping the clusters to the attention dataframe.
            self.clust_to_att_dict[f"Cluster_{i}"] = curr_att_df
            # Getting the top n gene dataframe based on the mode.
            if mode.lower() == "tfidf":
                self.clust_sums_dict[f"Cluster_{i}"] = curr_att_df.T.sum(axis=1)
            else:
                self.top_n_df_dict[f"Cluster_{i}"] = self.get_top_n(
                    n=n,
                    dataframe=curr_att_df,
                    verbose=False,
                    rank_mode=top_n_rank_method)

            # Gettting the top n gene names in ranked order (frp, highest
            # expression to lowest).
            self.top_n_names_dict[f"Cluster_{i}"] = curr_att_df.T.sum(
                axis=1).nlargest(n).index

            print(f"    >-< Done with Cluster {i}:")
            print()

        if mode.lower() == "tfidf":
            return (self.clust_to_att_dict, self.clust_sums_dict,
                    self.top_n_names_dict)

        else:
            # TODO: Check the ranking and ordering on the returns
            return (self.clust_to_att_dict, self.top_n_df_dict,
                    self.top_n_names_dict)

    def make_values_unique(self,
                           top_n_dictionary: dict = None,
                           threshold: int = None):
        """ Class method to make all the values in a dictionary unique.

        Args:
            top_n_dictionary: The dictionary containing the top n gene names for
              all populations (or smaller populations)
            threshold: The threshold for removing common genes: if a gene occurs
              in threshold many populations, it will be removed.

        Returns:
            The modified *unique* top_n_dictionary dictionary based on the
            threshold provided.

        Raises:
            None.
        """

        if top_n_dictionary is None:
            print("==> Since no dictionary was provided, we will use gene names"
                  " dictionary as default")
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
        duplicate_list = [
            item for item, count in collections.Counter(all_genes).items()
            if count > threshold
        ]
        print(f"==> Found {len(duplicate_list)} many duplicates that appear in"
              " more than {threshold} cluster(s)")

        for key in att_dict.keys():
            att_dict[key] = [
                item for item in att_dict[key] if item not in duplicate_list
            ]

        if top_n_dictionary is None:
            self.global_unique_gene_names = att_dict

        return att_dict

# Getter and setter methods for the class.

    def get_scanpy_object(self):
        """Getter method for returning Scanpy object at any given time.

        Args:
            None.

        Returns:
            The current scanpy data set as the internal attribute.

        Raises:
            None
        """
        return self.data

    def set_scanpy_object(self, new_data):
        """Setter method for setting Scanpy object at any given time

        Args:
            new_data: A new scanpy object to replace the previously passed data.

        Return:
            None. Method will set the new dataset using the passed scanpy
            object.

        Raises:
            None.

        """
        self.data = new_data


# TODO: @Oscar: could you please take care of the clean up and docstrin of this
# method please?

    def gene_tfidf(self, top_n_genestrings_dict: dict = None):
        """Calculate TF-IDF values for all genes in a top-n ranked dictionary.

        ***
        @Oscar: This needs to be cleaned up a bit :)
        ***

        """

        if top_n_genestrings_dict is None:
            try:
                att_dict = self.top_n_genestrings_dict.copy()
                print("==> Since no dictionary was provided, we will use gene"
                      " names dictionary as default")
            except Exception as _:
                print("==> Please either provide a top n gene name list, or set"
                      "the attribute 'self.top_n_genestrings_dict'")
        else:
            att_dict = top_n_genestrings_dict

        top_n_genes = pd.DataFrame.from_dict(att_dict)

        def tf(genes):
            """Calculate the gene frequency of a list of genes
            ***
            @Oscar: This needs to be cleaned up a bit and maybe moved to a
            different scope : )
            ***
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
            """Calculate the inverse document frequency of genes in a list of
               dictionaries.

            ***
            @Oscar: This needs to be cleaned up a bit and maybe moved to a
            different scope : )
            ***
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
        print("==> Calculating gene frequencies")
        tf_vals = top_n_genes.apply(tf, axis=1)

        # calculate inverse document frequencies
        print("==> Calculating inverse document frequencies")
        idf_vals = idf(tf_vals)

        # calculate tf-idf values for all genes in ranked_topN_strdf
        print("==> Calculating tf-idf values for all ranked genes")
        for document in range(len(tf_vals)):
            for gene in tf_vals[document]:
                tf_idf[gene] = tf_vals[document][gene] * idf_vals[gene]
        return tf_idf
