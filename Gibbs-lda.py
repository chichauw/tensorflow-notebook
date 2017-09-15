# -*- coding: utf-8 -*-
# Gibbs Sampling for LDA Example

from __future__ import division
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


vocab = ["money", "loan", "bank", "river", "stream"]

z_1 = np.array([1/3, 1/3, 1/3, .0, .0])
z_2 = np.array([.0, .0,1/3, 1/3, 1/3])

phi_actual = np.array([z_1, z_2]).T.reshape(len(z_2), 2)
print(phi_actual)

# number of documents
D = 16

# mean word length of documents
mean_length = 10

# sample a length for each document using Poisson
len_doc = np.random.poisson(mean_length, size=D)

# number of topics
T = 2


## Data Generation
docs = []
orig_topics = []
for i in range(D):
    z = np.random.randint(0, 2)
    if z == 0:
        words = np.random.choice(vocab, size=(len_doc[i]),p=z_1).tolist()
    else:
        words = np.random.choice(vocab, size=(len_doc[i]),p=z_2).tolist()
    orig_topics.append(z)
    docs.append(words)

print(" ".join(docs[0]))  # print document 1
print(" ".join(docs[1]))  # print document 2
print(" ".join(docs[2]))  # print document 3
print(orig_topics)


## Initialization
w_i = []
i = []
d_i = []
z_i = []
counter = 0

for doc_idx, doc in enumerate(docs):
    for word_idx, word in enumerate(doc):
        # pointer to word in vocab
        w_i.append(np.where(np.array(vocab)==word)[0][0])
        # counter
        i.append(counter)
        # pointer to which document it belongs to
        d_i.append(doc_idx)
        # initialize topic assignment
        z_i.append(np.random.randint(0, T))
        counter += 1

w_i = np.array(w_i)
d_i = np.array(d_i)
z_i = np.array(z_i)


## initialize our count matrices C^WT and C^DT

WT = np.zeros((len(vocab),T))
for idx, word_ in enumerate(vocab): 
    # for each topic, count the number of times 
    # it is assigned to a word
    topics = z_i[np.where(w_i == idx)] 
    for t in range(T):
        WT[idx,t] = sum(topics == t)
        

DT = np.zeros((D,T)) # create an empty matrix
for idx, doc_ in enumerate(range(D)):
    # for each topic, count the number of times 
    # it is assigned to a document
    topics = z_i[np.where(d_i == idx)]
    for t in range(T):
        DT[idx,t] = sum(topics == t)

# distribution for each word
WT_orig = WT.copy()
print "Oringal WT=", WT

# distributioni for each document
DT_orig = DT.copy()
print "Oringal DT=", DT

## Gibbs Sampling

phi_1 = np.zeros((len(vocab),100))
phi_2 = np.zeros((len(vocab),100))

# Sample Iterations
iters = 100

# The Dirichlet priors
beta = 1.
alpha = 1.

for step in range(iters):
    for current in i:
        # get document id and word id
        doc_idx = d_i[current] 
        w_idx = w_i[current]
                
        # count matrices - 1
        DT[doc_idx,z_i[current]] -= 1
        WT[w_idx,z_i[current]] -= 1
        
        # calculate new assignment 
        prob_word =  (WT[w_idx,:] + beta) / (WT[:,:].sum(axis=0) + len(vocab)* beta)
        prob_document = (DT[doc_idx,:] + alpha) / (DT.sum(axis=0) + D*alpha)
        prob = prob_word * prob_document
        
        #update z_i according to the probabilities for each topic
        z_i[current] = np.random.choice([0,1], 1, p=prob/prob.sum())[0]

        # update count matrices
        DT[doc_idx,z_i[current]] += 1
        WT[w_idx,z_i[current]] += 1
        
        # track phi 
        phi  = WT/(WT.sum(axis=0))
        phi_1[:,step] = phi[:,0]
        phi_2[:,step] = phi[:,1]

print "Final DT =",DT
print "Final WT =",WT

## Calculate posteriors
phi  = WT/(WT.sum(axis=0))
print "phi=", phi
theta = DT/DT.sum(axis=0)

# normalize to sum to 1
theta = theta/np.sum(theta, axis=1).reshape(16,1) 
print "theta=",theta
# Topics assigned to documents get the original document

print np.argmax(theta, axis=1) == orig_topics



## Plot

fig, axs = plt.subplots(1, 2, sharey=True)

for i in range(phi_1.shape[0]):
    axs[0].plot(phi_1[i,:20])
    axs[0].set_title('Topic 1')
    
for i in range(phi_2.shape[0]):
    axs[1].plot(phi_2[i,:20])
    axs[1].set_title('Topic 2')
    
fig.set_size_inches((8, 3))
sns.despine()

for ax in axs:
    ax.set_ylabel(r'$\phi_w$')
    ax.set_xlabel('Iteration')
    ax.set_yticks(np.arange(0, 0.5, 0.1))
    
fig.tight_layout()
fig.savefig('E:/fig1.pdf') 
## 
def plotHeatmap(data, ax):
    sns.heatmap(data, vmin=0, vmax=1, cmap='Blues', ax=ax, cbar=False, 
                annot=True, fmt=".3f", annot_kws={"size": 14})

phi_orig  = WT_orig/(WT_orig.sum(axis=0))

fig, axs = plt.subplots(1, 3, sharey=True)

plotHeatmap(phi_actual, axs[0])
axs[0].set_title(r"$\phi_{Original}$")

plotHeatmap(phi_orig, axs[1])
axs[1].set_title(r"$\phi_{Start}$")

plotHeatmap(phi, axs[2])
axs[2].set_title(r"$\phi_{End}$")


fig.set_size_inches((12, 3))

for ax in axs:
    ax.set_xticklabels(['Topic 1', 'Topic 2'])
    ax.set_yticklabels(vocab, rotation=360)


fig.tight_layout()
fig.savefig('E:/fig2.pdf') 