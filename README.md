explore_GLDAS: Explore the GLDAS data and preprocess it to have one single file for entire canada.
correlator water wildfire: Correlates the preprocessed (combination of Grace and GLDAS) water data to the wildfire grided data
correlation_Mas_Sm.py : Calculates the correlation for a given location between the mascon and the according Soilmoisture data.
Wildfire preprocessing: Makes a grid of 1deg for every month which includes the cumulative area of every wildfire which was in that month within the cell.
water_combiner: Combines the meascons and the soil moisture in a certain way and rescales the grid which resulted from that to 1°

