import pickle 
import pandas as pd

pkl_file = open('nedbor.pkl', 'rb')
nedbor_data = pickle.load(pkl_file)
pkl_file.close()

temp_data = pd.read_csv("original_data.csv",sep=";", encoding="latin-1")
temp_data2 = temp_data[['sample_type', 'vegobjektid']]
temp_data = temp_data.reset_index(drop=True)


for x in temp_data2:
    print(x)

data_in_order_air_temp = []
data_in_order_p = []

counter = 0 
for element in temp_data2.values:
    try:
        nedbor_data[str(element[1])+"_"+str(element[0])]
    except:
        data_in_order_air_temp.append("")
        data_in_order_p.append("")
        continue

    try:
        data_in_order_air_temp.append(round(sum(nedbor_data[str(element[1])+"_"+str(element[0])]["air_temperatures"])/(len(nedbor_data[str(element[1])+"_"+str(element[0])]["air_temperatures"])), 3))
    except ZeroDivisionError:
        data_in_order_air_temp.append("")
    try:
        data_in_order_p.append(round(sum(nedbor_data[str(element[1])+"_"+str(element[0])]["perticipation"])/(len(nedbor_data[str(element[1])+"_"+str(element[0])]["perticipation"])), 3))
    except ZeroDivisionError:
        data_in_order_p.append("")


print(len(data_in_order_air_temp))
print(len(data_in_order_p))

data_1 = pd.DataFrame({"air_temp": data_in_order_air_temp})
data_2 = pd.DataFrame({"precipitation": data_in_order_p})

new_data = pd.concat([temp_data,data_1, data_2],axis=1)

new_data.to_csv("new_csv_with_weather.csv", sep=";", encoding="latin-1")

