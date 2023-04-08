import dataloader.plant_village_dataset as pvds

gen = pvds.PlantVillageDataGenerator('datasets', 'PlantVillage','Pepper__bell___Bacterial_spot',8,(224,224), 3, True)

X,y = gen.__getitem__(1)
print(X.shape)
print(y.shape)

for ele in X:
    print(ele)
