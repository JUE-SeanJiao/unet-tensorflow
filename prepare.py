import os
import csv


def create_csv(dirname):
    path = './data/'+dirname +'/'
    fnames = os.listdir(path)
    #name.sort(key=lambda x: int(x.split('.')[0]))
    #print(name)
    count = 0
    with open('data_'+dirname+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for fname in fnames:
            if fname.endswith('.png'):
                # print(n)
                count += 1
                # with open('data_'+dirname+'.csv','rb') as f:
                writer.writerow(['./data/' + dirname + '/' + fname, './data/' + dirname + 'annot/' + fname])
            else:
                pass
    print('{} {} examples.'.format(count, dirname))


if __name__ == "__main__":
    create_csv('train')
    create_csv('val')
