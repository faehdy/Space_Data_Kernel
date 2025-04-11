All the relevant Scripts for the Project in CC2 can be found under Grace/Project/src

init.ipynb: Downloads the Data from Simon Stählers shared polybox folder

requirements.txt: Lists all the packages needed.

explore_GLDAS: Explore the GLDAS data and preprocess it to have one single file for entire canada.

LWE_Preprocessing: Interpolates the grace data for gaps using linear for only the modes 2, 3, 5. Therefore extracting the glacial-iceshield-anomalies. 

LWE_grid reduction: Reduces the LWE Mascons to a grid with the desired size (in our case 5°x5°)

correlation_Mas_Sm.py : Calculates the correlation for a given location between the mascon and the according Soilmoisture data.

Wildfire preprocessing: Makes a grid of 1deg for every month which includes the cumulative area of every wildfire which was in that month within the cell.

water_combiner: Combines the mascons and the soil moisture in a certain way and rescales the grid which resulted from that to 1°

correlator water wildfire: Correlates the preprocessed (combination of Grace and GLDAS) water data to the wildfire grided data



Workflow:
1. Clone this repository

2.1 Set up a venv and install the packages which are listed in requirements.txt 

2.2 Download all the data which are needed using the init.ipynb from the polybox shared by simon stähler (Backup Link: https://polybox.ethz.ch/index.php/f/4028862749)
If the data folder already exists delete it and rerun the code.

3. Run preprocessing/LWE_Preprocessing.ipynb for the interpolation of the missing mascons months, and to filter out the 1 mode of the PCA. 

4. Run the preprocessing/LWE_grid_reduction.ipynb to get the gridded data on the 5x5° grid

5. Run preprocessing/Explore_GLDAS.ipynb to combine all the individual Months to one big file with all the cells and all the Months

6. Run the correlations/correlation_Mas_SM.py Script to get the correlation values between the Soilmoisture data and the Mascons, the plots and the summary csv.

7. Run the preprocessing/water_combiner.ipynb which combines the Mascons and the SM data to one file and downsamples it to 5° Resolution

8. Run Plot_the_fires.ipynb to get an overview over the wildfire data

9. Run the preprocessing/Wildfire_preprocessing.ipynb to get a grid of 5° for each Month which includes the cumulative burned area

10. Run water_wildfire_correl.py to get the correlation and cross correlation for the wildfire area and the water. !! Attention: For processing speed reasons the script stores the matches to the Cache folder which is created. If different data is analysed make sure to delete the cache before rerunning the script!!

(11.) For further analysis the water input file in water_wildfire can be changed to a not combined file (only Grace or only GLDAS).




