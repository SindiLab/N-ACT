import os 
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

class Abdelaal_Prepro():
    def __init__(self, min_cell:int=0, raw_data=None, raw_data_path:str=None):
        """
        A class for preprocessing scRNAseq datasets stated in \
        Abdelaal, T., Michielsen, L., Cats, D. et al. A comparison of automatic cell identification methods for single-cell RNA sequencing data. Genome Biol 20, 194 (2019). https://doi.org/10.1186/s13059-019-1795-z
        
        Params
        ------
        min_cell:int
            The minimum number of cells that a gene must be detected in so that it is not filtered out. E.g., if min_cell = 3, then any gene that is detected in less than 3 cells will be removed.
        raw_data:AnnData
            An existing AnnData object which we want to modify. If we want to load in data for the first time (i.e. an existing object does not exist), this should remain None.
        raw_data_path:str
            The path to a scanpy-compatible data file that we want to load. This path should be provided only if we do not want to modify an existing object.
        
        Returns
        -------
        None. All done in place.

        """
        self.mincell = min_cell
        if not raw_data:
            if raw_data_path == None:
                raise ValueError("Either data or path to raw data must be provided")
            else:
                self.rawdata = sc.read_h5ad(raw_data_path)
        else:
            print("Taking the input for <rawdata> as data object")
            self.rawdata = raw_data
        

    def filter_genes(self, plot:bool = False):
        """
        Filter genes with at least min_detection cells
        
        Params
        ------
        plot:bool
            Whether to plot the distribution of the filtered genes or not
        """
        
        sums = np.sum(self.rawdata.X.todense() > self.mincell, axis=0)
        sums = np.array(sums).flatten().reshape(-1)
        
        keep_genes = sums >= 1
        
        print(f"Remaining genes after filtering: {np.sum(keep_genes)}")
        
        filt_genes = self.rawdata[:,keep_genes]

        if plot:

            plt.figure(figsize=(8,6))
            plt.style.use('seaborn')
            n, bins, patches = plt.hist(sums, bins = 100, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.9)

            n = n.astype('int')

            for i in range(len(patches)):
                patches[i].set_facecolor(plt.cm.RdYlBu_r(n[i]/max(n)))

            plt.xlabel('# cells in which gene is expressed')
            plt.ylabel('# of genes')
            plt.yscale('symlog')
            plt.title('Gene detection across cells')
            
            
        # now save the things we did to the object
        self.rawdata = filt_genes;
    
        return filt_genes

    def mad_cells(self):
        """
        Filter out cells when the total number of detected genes 
        is bellow three MAD from the median number of detected genes per cell
        
        Params
        ------
        None.
        
        Returns
        -------
        None. All modifications are done inplace.
        """
        try:
            # if it is not dense
            data = np.array(self.rawdata.X.todense())
        except ValueError:
            # if it is  dense
            print("Input needs to be anndata object")
        
        total_detected_genes = np.sum(np.array(self.rawdata.X.todense()), axis=1)


        # calculate the median number of detected genes per cell
        med = np.median(total_detected_genes)

        # calculate the median absolute deviation (mad) across all cells in the log scale
        mad = np.median(np.abs(total_detected_genes - med))
        log_mad = np.log10(mad)

        # get the absolute deviation from the median of each point 
        abs_dev = np.abs(total_detected_genes - med) / log_mad

        keep_cells = abs_dev > log_mad * 3

        print(f"Remaining cells after filtering: {np.sum(keep_cells)}")
        
        cells_filt = self.rawdata[keep_cells,:]
        
        # now save the things we did to the object
        self.rawdata = cells_filt;
        
        return cells_filt
    
    
    def get_highly_variable_genes(self, n_top_genes:int=5000, subset:bool=True, flavor='seurat_v3', span=0.3, n_bins=20,
               inplace=True):
        """
        Method to identify highly-variable genes (HGV). The method can either subset the data to have only HGVs, 
        or add index booleans as an additional attribute to the raw data
        
        Params (based on scanpy)
        ------
        n_top_genes: int
            Number of highly-variable genes to keep. Mandatory if flavor='seurat_v3'
        subset:bool
            Inplace subset to highly-variable genes if True otherwise merely indicate highly variable genes
        flavor: str 
            Choose the flavor for identifying highly variable genes. Possible choices: {‘seurat’, ‘cell_ranger’, ‘seurat_v3’}
            For the dispersion based methods in their default workflows, Seurat passes the cutoffs whereas Cell Ranger passes n_top_genes.
        span:float
            The fraction of cells used when estimating the variance in the loss model fit if flavor='seurat_v3'
        n_bins:int
            Number of bins for binning the mean gene expression. Normalization is done with respect to each bin. If just a single gene falls into a bin, the normalized dispersion is artificially set to 1.
        
        Return
        ------
        None, unless inplace=False, in which case will return the preprocessed data
        """
        print(f"Getting HGVs. Subsetting the data based on HGV is set to {subset}")
        # we are returning in case inplace=False, otherwise it will return None but will modify self.rawdata inplace
        return sc.pp.highly_variable_genes(self.rawdata, n_top_genes=n_top_genes, 
                                           subset=subset, flavor=flavor, span=span, n_bins=n_bins, 
                                           inplace=inplace)
    def save(self, path = None, prefix=""):
        """
        Save the modified data and the associated parameters in a JSON file.
        
        Params
        ------
        path:str
            Path to the directory we want to save the data+params in.
        prefix:str
            Prefix to be added to the save string for the file. Good option would be the name of the dataset.
        
        Returns
        -------
        None.
        
        """
        
        if path:
            save_path = path;
        else:
            dir_path = "./AbdelaalPreprocessedData/"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            save_path = dir_path + prefix + "_Abdelaal_processed.h5ad";
            
        print(f"Saving data and parameters to folder {save_path}");
        
        # write out the modified scanpy object
        self.rawdata.write(save_path)
        print("Done.")
  
        ####### For now, not worrying about saving the parameters #####
#         # save all the parameters in a json file for reproducibility
#         hparam = dict();
        
#         hparam['preprocessed'] = \
#         {
#             'total_count': self.cells_count,
#             'genes_no': self.genes_count,
#             'split_seed': self.seed,
#             'scale': self.scale
#         }        
        
#         with open(os.path.join(self.save_path, 'preprocessing_parameters.json'), 'w') as fp:
# #             json.dump(hparam, fp, sort_keys=True, indent=4)
#             json.dump(hparam, fp)