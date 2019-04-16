from census import Census
import pandas as pd
import shelve

c = Census('d599a7e3f4403a96f561574e64f6d8fd9bb34e20')

db = shelve.open('nb_zip/db')
nb_zip = db['uz']
db.close()

#ACS5 queries
acs5_dict = ['B01001_001E', # TOTAL_POP
             'B01001A_001E',# WHITE_POP
             'B01001B_001E',# BLACK_POP
             'B01001I_001E',# HISPANIC_POP
             'B05001_006E', # NON_CITIZEN
             'B17001_002E'] # POVERTY_RATE
             'B20002_001E', # MEDIAN_INCOME
             ['B25013_008E',# UNEDUCATED
              'B25013_003E'],
             ['C24010_023E',# POLICE_PRESENCE (law enforcement that live in zip code)
              'C24010_059E'],
             ['B21005_011E',# UNEMPLOYED
              'B21005_012E',
              'B21005_022E',
              'B21005_023E']]

unemployment_index = 9
poverty_index = acs5_dict.index('B17001_002E')


df = pd.read_csv('features_train.csv')

features_train = []
y = []
unique_zip = {}
for i in range(len(df.IMAGE_NAME)):
    container = []
    container.append(df.VAR1[i]) #SURELY THERES AN EASIER WAY... right?
    container.append(df.VAR2[i])
    container.append(df.VAR3[i])
    container.append(df.VAR4[i])
    container.append(df.VAR5[i])
    container.append(df.VAR6[i])
    container.append(df.VAR7[i])
    container.append(df.VAR8[i])
    container.append(df.VAR9[i])
    container.append(df.VAR10[i])
    container.append(df.VAR11[i])
    container.append(df.VAR12[i])
    container.append(df.VAR13[i])
    container.append(df.VAR14[i])
    container.append(df.VAR15[i])
    container.append(df.VAR16[i])
    container.append(df.VAR17[i])
    container.append(df.VAR18[i])
    container.append(df.VAR19[i])
    container.append(df.VAR20[i])
    container.append(df.ZIP[i])

    try:
        unique_zip[df.ZIP[i]]
    except:
        unique_zip[df.ZIP[i]] = []

    features_train.append(container)
    y.append(df.ALL_100_METERS_2016[i])

index_train = df.IMAGE_NAME

df = pd.read_csv('features_test.csv')

features_test = []
for i in range(len(df.IMAGE_NAME)):
    container = []
    container.append(df.VAR1[i]) #SURELY THERES AN EASIER WAY... right?
    container.append(df.VAR2[i])
    container.append(df.VAR3[i])
    container.append(df.VAR4[i])
    container.append(df.VAR5[i])
    container.append(df.VAR6[i])
    container.append(df.VAR7[i])
    container.append(df.VAR8[i])
    container.append(df.VAR9[i])
    container.append(df.VAR10[i])
    container.append(df.VAR11[i])
    container.append(df.VAR12[i])
    container.append(df.VAR13[i])
    container.append(df.VAR14[i])
    container.append(df.VAR15[i])
    container.append(df.VAR16[i])
    container.append(df.VAR17[i])
    container.append(df.VAR18[i])
    container.append(df.VAR19[i])
    container.append(df.VAR20[i])
    container.append(df.ZIP[i])

    try:
        unique_zip[df.ZIP[i]]
    except:
        unique_zip[df.ZIP[i]] = []

    features_test.append(container)

index_test = df.IMAGE_NAME

l = len(unique_zip)-1
i = 0
for zipcode in unique_zip:
    print(i, '/', l)
    container = []
    for query in acs5_dict:
        try:
            if query == 'B17001_002E':
                container.append(float(c.acs5.zipcode(query, str(zipcode))[0][query] / int(container[0])) * 100)
            else:
                container.append(int(c.acs5.zipcode(query, str(zipcode))[0][query]))
        except IndexError:
            na = [0] * len(acs5_dict)
            container = na
            break
        except TypeError:
            if query[0] == 'B25013_008E':
                uneducated_total = 0
                for q in query:
                    uneducated_total += int(c.acs5.zipcode(q, str(zipcode))[0][q])
                container.append(uneducated_total)
            elif query[0] == 'C24010_023E':
                police_presence = 0
                for q in query:
                    police_presence += int(c.acs5.zipcode(q, str(zipcode))[0][q])
                container.append(police_presence)
            elif query[0] == 'B21005_011E':
                unemployed = 0
                for q in query:
                    unemployed += int(c.acs5.zipcode(q, str(zipcode))[0][q])
                container.append(unemployed)
    unique_zip[zipcode] = container
    i+=1

print("Appending poverty within a quarter mile")
for zipcode in nb_zip:
    pv_rate = []
    if len(nb_zip[zipcode]) > 0:
        for z in nb_zip[zipcode]:
            pv_rate.append(unique_zip[z][poverty_index])
        unique_zip[zipcode].append(max(pv_rate))
    else:
        unique_zip[zipcode].append(0)


col_names = ['VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6', 'VAR7', 'VAR8',
             'VAR9', 'VAR10', 'VAR11', 'VAR12', 'VAR13', 'VAR14', 'VAR15',
             'VAR16', 'VAR17', 'VAR18', 'VAR19', 'VAR20', 'ZIP', 'TOTAL_POP',
             'WHITE_POP', 'BLACK_POP', 'HISPANIC_POP', 'NON_CITIZEN',
             'POVERTY_PER', 'MEDIAN_INCOME', 'UNEDUCATED', 'POLICE_PRESENCE',
             'UNEMPLOYED', 'POVERTY_QUARTER_MILE', 'ALL_100_METERS_2016']


for i in range(len(features_train)):
    z = features_train[i][-1]
    features_train[i] += unique_zip[z]
    features_train[i].append(y[i])

df = pd.DataFrame(features_train, index = index_train)
df.to_csv('features_train_1.csv', header = col_names)



for i in range(len(features_test)):
    z = features_test[i][-1]
    features_test[i] += unique_zip[z]

df = pd.DataFrame(features_test, index = index_test)
df.to_csv('features_test_1.csv', header = col_names[0:-1])



              
              
