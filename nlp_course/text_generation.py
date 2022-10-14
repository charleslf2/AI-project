import os
import tensorflow as tf
import numpy as np 


print("###################################################")
print("###################################################")
print("###################################################")

with open('data\shakespeare.txt', 'rb') as f:
    text=f.read().decode(encoding='utf-8')


print('dataset loaded.........')

# cup the text
text=text[11532 :]
#print(text[:250])
print('length of text : {} characters'.format(len(text)))


# text vecotorization

vocab=sorted(set(text))

print("{} unique charachters".format(len(vocab)))

char2idx= {unique:idx for idx, unique in enumerate(vocab)}
idx2char= np.array(vocab)

text_as_int=np.array([char2idx[char] for char in text])

#print('{')

#for char , _ in zip(char2idx, range(20)):
#    print(' {:4s}: {:3d}'.format(repr(char), char2idx[char]))

#print('...\n')

#print('{}====> charachter map to int ====>{}'
#    .format(repr(text[:13]), text_as_int[:30]))



# DIVIDED TEXT IN SMALL CHUNCKS

seq_length=100
examples_per_epoch=len(text) // (seq_length +1)
char_datasets=tf.data.Dataset.from_tensor_slices(text_as_int)

print('done')
for i in char_datasets.take(5):
    print(idx2char[i.numpy()])









