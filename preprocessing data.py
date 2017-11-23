from numpy import append
import pandas as pd
import numpy as np
import os
import shutil
import librosa
# Set number of columns to show in the notebook
pd.set_option('display.max_columns', 200)
# Set number of rows to show in the notebook
pd.set_option('display.max_rows', 50) 
# Make the graphs a bit prettier
pd.set_option('display.mpl_style', 'default') 

# Import MatPlotLib Package
import matplotlib.pyplot as plt
newdata = pd.read_csv('d:/annotations_final.csv', sep="\t")
print(newdata.head(5))
print(newdata.info())
print(newdata.columns)
print(newdata[["clip_id", "mp3_path"]])
clip_id, mp3_path = newdata[["clip_id", "mp3_path"]].as_matrix()[:,0], newdata[["clip_id", "mp3_path"]].as_matrix()[:,1]
synonyms = [['beat', 'beats'],
            ['chant', 'chanting'],
            ['choir', 'choral'],
            ['classical', 'clasical', 'classic'],
            ['drum', 'drums'],
            ['electro', 'electronic', 'electronica', 'electric'],
            ['fast', 'fast beat', 'quick'],
            ['female', 'female singer', 'female singing', 'female vocals', 'female vocal', 'female voice', 'woman', 'woman singing', 'women'],
            ['flute', 'flutes'],
            ['guitar', 'guitars'],
            ['hard', 'hard rock'],
            ['harpsichord', 'harpsicord'],
            ['heavy', 'heavy metal', 'metal'],
            ['horn', 'horns'],
            ['india', 'indian'],
            ['jazz', 'jazzy'],
            ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
            ['no beat', 'no drums'],
            ['no singer', 'no singing', 'no vocal','no vocals', 'no voice', 'no voices', 'instrumental'],
            ['opera', 'operatic'],
            ['orchestra', 'orchestral'],
            ['quiet', 'silence'],
            ['singer', 'singing'],
            ['space', 'spacey'],
            ['string', 'strings'],
            ['synth', 'synthesizer'],
            ['violin', 'violins'],
            ['vocal', 'vocals', 'voice', 'voices'],
            ['strange', 'weird']]
for synonym_list in synonyms:
    newdata[synonym_list[0]] = newdata[synonym_list].max(axis=1)
    newdata.drop(synonym_list[1:], axis=1, inplace=True)

print(newdata.info())
print(newdata.head())
print(newdata.drop('mp3_path', axis=1, inplace=True))
data = newdata.sum(axis=0)
print(data)
data.sort_values(axis=0, inplace=True)
topindex, topvalues = list(data.index[84:]), data.values[84:]
del(topindex[-1])
topvalues = np.delete(topvalues, -1)
print(topindex)
print(topvalues)
rem_cols = data.index[:84]
print(len(rem_cols))
newdata.drop(rem_cols, axis=1, inplace=True)
print(newdata.info())
backup_newdata = newdata
from sklearn.utils import shuffle
newdata = shuffle(newdata)
print(newdata.reset_index(drop=True))
print(newdata.info())
final_columns_names = list(newdata.columns)
del(final_columns_names[0])
print(final_columns_names)
final_matrix = pd.concat([newdata['clip_id'], newdata[final_columns_names]==1], axis=1)
root = 'D:/mtt'

#for id in range(25863):
    #print clip_id[id], mp3_path[id]
 #   src = root + "/" + mp3_path[id]
  #  dest = 'd:' + "/dataset_clip_id_mp3/" + str(clip_id[id]) + ".mp3"
   # shutil.copy2(src,dest)
    
mp3_available = []
melgram_available = []
for mp3 in os.listdir('d:/mtt/dataset_clip_id_mp3/'):
     mp3_available.append(int(os.path.splitext(mp3)[0]))
        
for melgram in os.listdir('d:/mtt/dataset_clip_id_melgram/'):
     melgram_available.append(int(os.path.splitext(melgram)[0]))
     
new_clip_id = final_matrix['clip_id']
print(set(list(new_clip_id)).difference(melgram_available))
final_matrix = final_matrix[final_matrix['clip_id']!= 35644]
final_matrix = final_matrix[final_matrix['clip_id']!= 55753]
final_matrix = final_matrix[final_matrix['clip_id']!= 57881]
print(final_matrix.info)
final_matrix.to_pickle('d:/mtt/final_Dataframe.pkl')
final_matrix=pd.read_pickle('d:/mtt/final_Dataframe.pkl')
training_with_clip = final_matrix[:19773]
training_with_clip.to_pickle('d:/mtt/training_with_clip.pkl')
print(training_with_clip.info)
validation_with_clip = final_matrix[19773:21294]
testing_with_clip = final_matrix[21294:]
validation_with_clip.to_pickle('d:/mtt/validation_with_clip.pkl')
testing_with_clip.to_pickle('d:/mtt/testing_with_clip.pkl')
training_clip_id = training_with_clip['clip_id'].values
validation_clip_id = validation_with_clip['clip_id'].values
testing_clip_id = testing_with_clip['clip_id'].values
os.chdir('d:/mtt/final_dataset/')
np.save('d:/mtt/train_y.npy', training_with_clip[final_columns_names].values)
np.save('d:/mtt/valid_y.npy', validation_with_clip[final_columns_names].values)
np.save('d:/mtt/test_y.npy', testing_with_clip[final_columns_names].values)

# Save the 'x' clip_id's. We will make the numpy array using this.
np.savetxt('d:/mtt/train_x_clip_id.txt', training_with_clip['clip_id'].values, fmt='%i')

np.savetxt('d:/mtt/test_x_clip_id.txt', testing_with_clip['clip_id'].values, fmt='%i')

np.savetxt('d:/mtt/valid_x_clip_id.txt', validation_with_clip['clip_id'].values, fmt='%i')
train_x = np.zeros((0,96,1366,1))
test_x = np.zeros((0,96,1366,1))
valid_x = np.zeros((0,96,1366,1))



root = 'd:/mtt/'
os.chdir(root + "/dataset_clip_id_melgram/")
for i,train_clip in enumerate(list(training_clip_id)):
    if os.path.isfile(str(train_clip) + '.npy'):
        #print i,train_clip
        
        melgram = np.load(str(train_clip) + '.npy')
        mel=melgram
        t=melgram
        t = t[np.newaxis,:]
        train_x=np.concatenate((train_x,t),axis=0)
        print(train_x.shape)
        np.save('d:/mtt/train_x.npy', train_x)
        train_x=np.load('d:/mtt/train_x.npy')
os.chdir('d:/mtt/')
#np.save('d:/mtt/train_x.npy', train_x)
print ("Training file created.")