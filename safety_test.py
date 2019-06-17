import safety_lib
import numpy as np
import pandas as pd
import time


part_0 = pd.read_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\features\part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
part_1 = pd.read_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\features\part-00001-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
part_2 = pd.read_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\features\part-00002-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
part_3 = pd.read_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\features\part-00003-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
part_4 = pd.read_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\features\part-00004-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
part_5 = pd.read_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\features\part-00005-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
part_6 = pd.read_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\features\part-00006-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
part_7 = pd.read_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\features\part-00007-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
part_8 = pd.read_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\features\part-00008-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
part_9 = pd.read_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\features\part-00009-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
data_label = pd.read_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\labels\part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv")


if __name__=='__main__':
    start_time = time.time()
    print ("Part1:Data cleaning and preprocessing.")
    print (" -reading dataset...")
    dataset_features = pd.concat([part_0,part_1,part_2,part_3,part_4,part_5,part_6,part_7,part_8,part_9]) 
    #remove duplicated data_label with different label
    data_label = data_label.drop_duplicates(keep='first')
    data_label = data_label.drop_duplicates(subset='bookingID',keep=False)
    #merge dataset with label
    dataset_features_label = pd.merge(dataset_features, data_label,left_on = ['bookingID'],right_on =['bookingID'] , how = 'left')
    dataset_features_label = dataset_features_label.dropna(subset = ['label']) 
    print("--- %s seconds ---" % (time.time() - start_time)) 

    print (" -filtering outlier and normalizing data...")
    #filter low accuracy data based on IQR
    outlier = safety_lib.reject_outliers_IQR(dataset_features_label)
    outlier_df = safety_lib.find_bearing_diff(outlier)        
    #use mean to check for signal offset during speed and bearing_diff = 0
    offset_df = safety_lib.offset_signal(outlier_df, ['acceleration_x','acceleration_y','acceleration_z'])

    print("--- %s seconds ---" % (time.time() - start_time))

#    print (" -running LPF for critical signals to filter unwanted spike signal")  
#    start_time = time.time()     
#    lpf_df = safety_lib.butter_lowpass_filter(offset_df,fc=0.6,fs=1,order=2,column=['acceleration_x_offset','acceleration_y_offset','acceleration_z_offset','gyro_x','gyro_y','gyro_z'])    
#    print("--- %s seconds ---" % (time.time() - start_time)) 

    #magnitude of acceleration and gyroscope
    print (" -generating magnitude of acceleration and gyroscope...")
    offset_df['a_mag'] = np.sqrt(offset_df['acceleration_x_offset']**2 + offset_df['acceleration_y_offset']**2 + offset_df['acceleration_z_offset']**2)
    offset_df['g_mag'] = np.sqrt(offset_df['gyro_x']**2 + offset_df['gyro_y']**2 + offset_df['gyro_z']**2)  
#    lpf_df['a_mag_lpf'] = np.sqrt(lpf_df['acceleration_x_offset_lpf']**2 + lpf_df['acceleration_y_offset_lpf']**2 + lpf_df['acceleration_z_offset_lpf']**2)
#    lpf_df['g_mag_lpf'] = np.sqrt(lpf_df['gyro_x_lpf']**2 + lpf_df['gyro_y_lpf']**2 + lpf_df['gyro_z_lpf']**2)  
#    lpf_df.to_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\after_lpf.csv", index = False)
#    offset_df.to_csv(r"C:\Users\nxf47752\Desktop\ChoonYan\CY_documents\DataScience\safety\after_offset_nolpf.csv", index = False)

    print ("Part2: Random forest modelling.")
    dataset = offset_df
    print (" -creating features and X y data...")
    features_result = safety_lib.features(dataset, 'bookingID')
    X, y = safety_lib.create_data(data_label,features_result, 'bookingID')
    print (" -running train test...")
    cmatrix, classification, accuracy = safety_lib.Train_Test(X, y)
    print (" RESULT:\n")
    print (" -confusion_matrix:\n", cmatrix)
    print (" -classification_report:\n", classification)
    print (" -accuracy_score:", accuracy)
    print ("\n")

    print("--- %s seconds ---" % (time.time() - start_time))  
  











 






