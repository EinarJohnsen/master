import pandas as pd
import pickle 

data = pd.read_csv("path_to_negative_data.csv",sep=";", encoding="latin-1")

print(len(data))

data = data.reset_index()
temp_data = data[['vegobjektid']]

print(len(temp_data))

pkl_file = open('downloaded_negative_weather_data.pkl', 'rb')    
                                                                                                                                                                                       
weather_data = pickle.load(pkl_file)                                                                                                                                                                                                          
pkl_file.close()

air_temps = []
pec_vals = []

counter = 0 
for x in temp_data.values: 
    has_same_values = []

    temp_air_temps = []
    temp_pec_vals = []

    try:
        weather_data[x[0]]
        
        has_at = False
        has_pec = False
        
        for y in weather_data[x[0]]:
            if y["sourceId"]+y["referenceTime"] not in has_same_values:
                has_same_values.append(y["sourceId"]+y["referenceTime"])
                
                for z in y["observations"]:
                    
                    if z["elementId"] == "air_temperature":
                        temp_air_temps.append(z["value"])
                        has_at = True 
                    
                    if z["elementId"] == "sum(precipitation_amount PT1H)":
                        temp_pec_vals.append(z["value"])
                        has_pec = True
                
        if has_at == False:
            air_temps.append(" ")
            #continue
        if has_pec == False:
            pec_vals.append(" ")
            #continue
    except:
        air_temps.append(" ")
        pec_vals.append(" ")

    try:
        air_temps.append(round(sum(temp_air_temps)/len(temp_air_temps), 3))
    except ZeroDivisionError:
        pass 

    try:
         pec_vals.append(round(sum(temp_pec_vals)/(len(temp_pec_vals)), 3))
    except ZeroDivisionError:
        pass 
    

print(len(air_temps), " air temps")
print(len(pec_vals), " pec vals")


data_1 = pd.DataFrame({"air_temp": air_temps})
data_2 = pd.DataFrame({"precipitation": pec_vals})

new_data = pd.concat([data,data_1, data_2],axis=1)

new_data.to_csv("new_file.csv", sep=";", encoding="latin-1")


