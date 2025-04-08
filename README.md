explore_GLDAS: Explore the GLDAS data and preprocess it to have one single file for entire canada.

correlator water wildfire: Correlates the preprocessed (combination of Grace and GLDAS) water data to the wildfire grided data

correlation_Mas_Sm.py : Calculates the correlation for a given location between the mascon and the according Soilmoisture data.

Wildfire preprocessing: Makes a grid of 1deg for every month which includes the cumulative area of every wildfire which was in that month within the cell.

water_combiner: Combines the meascons and the soil moisture in a certain way and rescales the grid which resulted from that to 1°



Workflow:
1. Download all the relevant GLDAS Files from the NASA Website

2. Run Explore GLDAS to combine all the individual Months to one big file with all the cells and all the Months

3. Run the correlation_Mas_SM Script to get the correlation values between the Soilmoisture data and the Mascons

4. Run the water combiner which combines the Mascons and the SM data to one file and downsamples it to 1° Resolution

5. Run Plot_the_wildfires to get an overview over the wildfire data

6. Run the wildfire_Preprocessing.ipynb to get a grid of 1° for each Month which includes the cumulative burned area

7. Run even_more_eff_w_w_correlator.py !!!!!!CHANGE_NAME!!!!!! to get the correlation and cross correlation for the wildfire area and the water




