README for /Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations

InputData:
    Has the original data created on Rivanna and scp here.

Preprocessing:
    Takes in InputData and generates useble final files for highest and lowest separately.
    note--we don't use all the data, but made decisions prior and this is reflected in
    the preprocessing files as well as some processing statistics, eg len of final files
    questions used, subject frequencies, etc.. It also includes a replicate of the left_right
    min/maxers analysis for sanity and it was needed to add this info to the cleaning process.
    USE THESE .pkl FILES as input to the simulations.

Left_Right_MinMaxers_Analysis:
    The dev and final version of the miners maxers analysis.  Can use to cross ref with
    preprocessing.

Lowest/Highest:
    These are the simulation filesets.

Priors:
    These are pre-made priors for both the high and lows. Goes frmo 20 to 200 as mean
    of the Poisson with N=1000.  The three files are pickles from lists and are related
    directly by their index:
    catch_prior_index_out[i] --> is the mean value of the prior
    catch_all_prior_over_all_t_out[i] --> is the median value of the posterior over t
    catch_all_dist_prior_over_all_mean[i] --> is the raw distribution of samples for the prior
    
Theory:
    This is for showing some of the key theoretical aspects of the mathematical model.
    
Dev:
    This is for dev.
    AND, this is where study 2 work resides.

#EOF