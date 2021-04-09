import os

if __name__ == '__main__':
    path = r'Full path to class'
    counter = 1
    for file in os.listdir(path):
        src = file
        dst = "{}.jpg".format(counter)
        os.rename(path + '\\' + src, path + '\\' + dst)
        counter += 1
