'''

PA-3: FastMap
Authors:
Amitabh Rajkumar Saini, amitabhr@usc.edu
Shilpa Jain, shilpaj@usc.edu
Sushumna Khandelwal, sushumna@usc.edu

Dependencies:
1. numpy : pip install numpy
2. matplotlib : pip install matplotlib

Output:
Return the mapped the data points to a 2D space just based on the symmetric distance between them.
The output will be the distances between the projections on two dimensionalities and a plot graph for the same


'''


import random
import numpy as np
import math
import matplotlib.pyplot as plt

class fastmap:
    def __init__(self,dmap,word,k):
        '''
        Constructor called when object  is invoked
        :param dmap: dictionary for storing fastmap-data with key as tuple indicationg the word and value as distance between those two words
        :param word:word list
        :param k:dimension
        :param embedding: Final output

        '''

        self.dmap=dmap   #(x,y):dist
        self.words=word #words
        self.k = k
        self.embedding = np.zeros((len(word)+1,self.k))  #[[x0,y0][x2,y2],[x3,y3]..]

    def get_farthest(self):
        '''
        Calculates the farthest points by not traversing all the points as discussed in class
        :return the two farthest points in the form of tuple
        '''

        rand_index = random.randint(0,len(self.dmap)-1)
        pivot = random.choice([list(self.dmap.keys())[rand_index][0], list(self.dmap.keys())[rand_index][1]])
        #print(pivot)
        prev = None
        while(True):
            maxi = float('-inf')
            next = None
            for i in range(1,len(self.words)+1):
                if i == pivot:
                    continue
                if (i,pivot) in self.dmap:
                    d=self.dmap[(i,pivot)]
                else:
                    d=self.dmap[(pivot,i)]

                if maxi < d:
                    maxi = d
                    next = i

            if prev == next:
                break

            prev = pivot
            pivot = next

        return (pivot,next)

    def calculate_embedding(self):
        '''
            Embeds the points on hyperplane for k =2 each time and updates the distance by subtracting the distnace calculated in
            previous iteration
            :returns the funcyion returns nothing
        '''

        k = 0
        #print(self.dmap)
        while k < self.k :
            far = self.get_farthest()
            if far not in self.dmap:
                far = (far[1],far[0])
            oa , ob = far
            for i in range(1,len(self.words)+1):
                if i == oa:
                    self.embedding[i][k]=0
                    continue
                elif i == ob:
                    self.embedding[i][k]=self.dmap[far]
                    continue
                pta = (i,oa) if (i,oa) in self.dmap else (oa,i)
                ptb = (i,ob) if (i,ob) in self.dmap else (ob,i)
                di = (self.dmap[far]**2 + self.dmap[pta]**2 - self.dmap[ptb]**2)/(2*self.dmap[far])
                self.embedding[i][k] = di
            self.update_dist(k)

            k+=1

    def plot(self, label_set):
        '''

        :param label_set:list containing word list
        :return: the function returns nothing just plots a graph representing the word list on 2d plane
        '''
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        self.embedding=self.embedding[1:]

        for i in range(1,len(self.embedding)):
            _x = self.embedding[i][0]
            _y = self.embedding[i][1]
            _label = label_set[i]
            plt.plot(_x, _y, 'b.', markersize=10)
            plt.annotate(
                _label, xy = (_x, _y), xytext = (30, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.1),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        plt.show()


    def update_dist(self,k):
        '''
        :param k: denoting the column in embedding
        :return: the updated distance by subtarcting teh distance in previous iteration
        '''
        for each in self.dmap.keys(): #(2,3)
            self.dmap[each] = math.sqrt((self.dmap[each]**2 - (self.embedding[each[0]][k]-self.embedding[each[1]][k])**2))

    def map_words(self):
        '''
        Combines the words and coordinates
        :return: the function returns nothing just prints the o/p on console
        '''
        map_words = zip(self.words,self.embedding[1:,:])
        #print(list(map_words))
        print("Word\t\tEmbedding")
        for each in map_words:
            print(each[0]+"\t\t"+str(each[1]))


def main():
    '''
    Runner Program
    :return: returns nothing
    '''
    d_map={}
    #Data loaded into a numpy array
    data=np.loadtxt('fastmap-data.txt')
    #Word-list loaded in list
    words=open('fastmap-wordlist.txt')
    words=words.read().split('\n')
    #k : dimension
    k = 2
    for i in data:
        d_map[(int(i[0]),int(i[1]))]=int(i[2])
    #object instantiation
    fmap = fastmap(d_map,words,k)
    fmap.calculate_embedding()
    fmap.map_words()
    fmap.plot(words)


if __name__ == "__main__":
    main()

