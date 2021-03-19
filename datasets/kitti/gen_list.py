import os
import random

def run(mode, save_path='split'):
    if mode == 'train':
        st = 0; ed = 8
    else:
        st = 9; ed = 10

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    f = open(os.path.join(save_path, '%s.csv' % mode), 'w')
    for id in range(st, ed+1):
        fnames = os.listdir("./data/%02d" % id)
        fnames.sort()
        k = random.randint(1, 7)
        num = len(fnames)
        for i in range(num):
            if i + k < num and i + 2*k < num:
                print("%02d,%s,%s,%s" % (id, fnames[i], fnames[i+k], fnames[i+2*k]), file=f)
    
    f.close()

if __name__ == '__main__':
    save_path = 'split'

    # train
    run('train', save_path)

    # test
    run('test', save_path)